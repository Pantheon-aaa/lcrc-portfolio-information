"""Scaled native SOCP with independently evaluated QP weak-duality intervals."""
from dataclasses import dataclass,field
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import clarabel
from core import feasible
from research_v2.risk_envelopes import tail_weights

@dataclass
class Result:
    w: np.ndarray
    lower: float
    upper: float
    status: str
    seconds: float=0.
    diagnostics: dict=field(default_factory=dict)
    @property
    def gap(self):return self.upper-self.lower

def simplex_caps(q,cap):
    q=np.minimum(np.maximum(q,0),cap)
    if q.sum()>1:q/=q.sum()
    rem=1-q.sum()
    if rem>0:
        free=np.maximum(cap-q,0)
        if free.sum()>=rem:q+=rem*free/free.sum()
    # Tiny residual is allocated to an atom with available room.
    j=np.argmax(cap-q) if q.sum()<1 else np.argmax(q)
    q[j]+=1-q.sum()
    return q

def solve(fam,p=None,mode='max',indices=None,epsilon=0.,tight=False,time_limit=30.):
    tic=perf_counter();ids=np.arange(fam.M) if indices is None else np.array(sorted(indices),int)
    if not len(ids):raise ValueError('empty problem')
    p=np.ones(fam.M)/fam.M if p is None else np.asarray(p,float)
    d=fam.d;m=len(ids);u=fam.unit;s=fam.scale;b=fam.base
    V=fam.V[ids];Q=fam.Q[ids];g=np.einsum('mij,j->mi',Q,b);g-=g.mean(1,keepdims=True)
    const=fam.costs(b)[ids]-V
    hinge=mode in ('tail','hinge');tie=mode=='tie'
    N=d+m+(1+m if hinge else (0 if tie else 1));ir=d;iz=d+m;iy=iz+1
    c=np.zeros(N);P=sp.csc_matrix((N,N))
    if tie:
        Qp=np.einsum('m,mij->ij',p,fam.Q);gp=Qp@b;gp-=gp.mean()
        P=sp.block_diag([sp.csc_matrix(Qp*s*s/u),sp.csc_matrix((N-d,N-d))],format='csc');c[:d]=gp*s/u
    elif mode=='max':c[iz]=1
    elif mode=='tail':c[iz]=1;c[iy:]=p[ids]/.5
    elif mode=='hinge':c[iy:]=p[ids]
    else:raise ValueError(mode)
    mats=[];rhs=[];cones=[]
    A=np.zeros((1,N));A[0,:d]=1
    mats.append(sp.csc_matrix(A));rhs.append(np.zeros(1));cones.append(clarabel.ZeroConeT(1))
    rr=[];cc=[];vv=[];bb=[]
    def add(cols,vals,bound):
        r=len(bb);rr.extend([r]*len(cols));cc.extend(cols);vv.extend(vals);bb.append(bound)
    for a in range(d):add([a],[-1],b[a]/s);add([a],[1],(fam.upper-b[a])/s)
    for j in range(m):
        if tie:add([ir+j],[1],epsilon/u)
        elif mode=='max':add([ir+j,iz],[1,-1],0)
        elif mode=='tail':add([ir+j,iz,iy+j],[1,-1,-1],0);add([iy+j],[-1],0)
        else:add([ir+j,iy+j],[1,-1],epsilon/u);add([iy+j],[-1],0)
    # Hinge's unused z is fixed, avoiding a free null direction.
    if mode=='hinge':add([iz],[1],0);add([iz],[-1],0)
    mats.append(sp.csc_matrix((vv,(rr,cc)),shape=(len(bb),N)));rhs.append(np.array(bb));cones.append(clarabel.NonnegativeConeT(len(bb)))
    soc_start=1+len(bb)
    for j in range(m):
        A=np.zeros((d+2,N));A[:2,:d]=g[j]*s/u;A[:2,ir+j]=-1
        A[2:,:d]=-np.sqrt(2/u)*s*np.linalg.cholesky(Q[j]).T
        bj=np.zeros(d+2);bj[0]=1-const[j]/u;bj[1]=-1-const[j]/u
        mats.append(sp.csc_matrix(A));rhs.append(bj);cones.append(clarabel.SecondOrderConeT(d+2))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=300
    settings.tol_gap_abs=1e-11 if tight else 1e-9;settings.tol_gap_rel=settings.tol_gap_abs;settings.tol_feas=1e-11
    settings.time_limit=max(.01,time_limit)
    ans=clarabel.DefaultSolver(P,c,sp.vstack(mats,format='csc'),np.concatenate(rhs),cones,settings).solve()
    w=feasible(b+s*np.array(ans.x[:d]),np.full(d,fam.upper))
    lo,hi=fam.regret_bounds(w)
    raw=np.array(ans.z[soc_start:]).reshape(m,d+2);q=np.maximum(0,raw[:,0]+raw[:,1])
    if tie:
        multipliers=q
        mass=1+multipliers.sum();mix=p.copy();mix[ids]+=multipliers
        oracle=fam.mean(mix/mass)
        L=mass*oracle['value_lb']-p@fam.VU-multipliers@(fam.VU[ids]+epsilon)
        U=p@hi
    else:
        if mode=='max':q=q/q.sum() if q.sum()>0 else np.ones(m)/m;U=float(hi[ids].max())
        elif mode=='tail':q=simplex_caps(q,p[ids]/.5);U=float(tail_weights(hi[ids],p[ids],.5)@hi[ids])
        else:q=np.minimum(q,p[ids]);U=float(p[ids]@np.maximum(hi[ids]-epsilon,0))
        mass=q.sum();mix=np.zeros(fam.M);mix[ids]=q
        L=mass*fam.mean(mix/mass)['value_lb']-q@fam.VU[ids] if mass>1e-15 else 0.
        if mode=='hinge':L-=epsilon*mass
    residual=max(abs(w.sum()-1),float(np.max(-w)),float(np.max(w-fam.upper)),0.)
    return Result(w,float(L),float(U),str(ans.status),perf_counter()-tic,
      dict(primal_residual=residual,solver_gap=float(ans.obj_val-ans.obj_val_dual)*u,dual_weights=q,
           model_indices=ids,mode=mode,unit=u,iterations=ans.iterations))

class Compatibility:
    def __init__(self,fam):self.fam=fam;self.cache={};self.calls=0
    def gamma(self,ids,tight=False):
        key=tuple(sorted(ids))
        if not key:raise ValueError('empty set')
        if key not in self.cache or tight:
            if len(key)==1:
                j=key[0];w=self.fam.oracle[j];_,hi=self.fam.regret_bounds(w)
                r=Result(w,0.,float(hi[j]),'analytic_singleton')
            else:r=solve(self.fam,indices=key,tight=tight)
            self.calls+=1
            old=self.cache.get(key)
            if old:
                if old.upper<r.upper:r.w=old.w;r.upper=old.upper
                r.lower=max(r.lower,old.lower)
            self.cache[key]=r
        return self.cache[key]
    def classify(self,ids,epsilon):
        r=self.gamma(ids)
        if r.upper<=epsilon:return 'compatible',r
        if r.lower>epsilon:return 'incompatible',r
        r=self.gamma(ids,tight=True)
        return ('compatible' if r.upper<=epsilon else ('incompatible' if r.lower>epsilon else 'indeterminate')),r
    def witness(self,ids,p,epsilon,center):
        r=solve(self.fam,p,'tie',ids,epsilon,tight=True)
        _,hi=self.fam.regret_bounds(r.w)
        if np.max(hi[list(ids)])>epsilon:
            # Convex interpolation to a certified interior witness; epsilon is unchanged.
            lo=0.;up=1.
            for _ in range(45):
                a=(lo+up)/2;w=(1-a)*center.w+a*r.w
                if self.fam.regret_bounds(w)[1][list(ids)].max()<=epsilon:lo=a
                else:up=a
            r.w=(1-lo)*center.w+lo*r.w;r.upper=float(p@self.fam.regret_bounds(r.w)[1]);r.diagnostics['tie_interpolation']=lo
        return r

def cvx_reference(fam,ids):
    import cvxpy as cp
    d=fam.d;x=cp.Variable(d);z=cp.Variable();b=fam.base;s=fam.scale;u=fam.unit
    w=b+s*x;g=np.einsum('mij,j->mi',fam.Q,b);g-=g.mean(1,keepdims=True)
    r=[.5*cp.quad_form(x,cp.psd_wrap(fam.Q[j]))*s*s/u+g[j]@x*s/u+(fam.costs(b)[j]-fam.V[j])/u for j in ids]
    prob=cp.Problem(cp.Minimize(z),[cp.sum(x)==0,w>=0,w<=fam.upper]+[v<=z for v in r])
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11)
    return float(prob.value*u),np.asarray(w.value)
