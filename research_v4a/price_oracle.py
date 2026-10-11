"""Native conic pricing, feasible witnesses and independent weak-duality bounds."""
from dataclasses import dataclass,field
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import clarabel
from core import feasible
from research_v3.compatible_qcqp import solve as geometry_solve,Result,simplex_caps
from research_v2.risk_envelopes import tail_weights
from .mean_budget import MeanBudget
from .common import digest

@dataclass
class PriceResult:
    witness_weights: np.ndarray
    cost_lower: float
    cost_upper: float
    epsilon_feasible: bool
    primal_residual: float
    dual_residual: float
    status: str
    diagnostics: dict=field(default_factory=dict)

class Geometry:
    def __init__(self,family,epsilon):self.fam=family;self.epsilon=float(epsilon);self.key=(family.hash,self.epsilon);self.cache={};self.cuts=[];self.calls=0
    def gamma(self,ids,deadline=np.inf,tight=False):
        ids=tuple(sorted(ids))
        if not ids:return Result(self.fam.base,0.,0.,'empty')
        if ids not in self.cache or tight:
            if perf_counter()>=deadline:raise TimeoutError('query deadline')
            if len(ids)==1:
                j=ids[0];r=Result(self.fam.oracle[j],0.,self.fam.errors[j],'singleton')
            else:r=geometry_solve(self.fam,indices=ids,tight=tight,time_limit=min(30,max(.001,deadline-perf_counter())))
            old=self.cache.get(ids)
            if old:
                if old.upper<r.upper:r.w=old.w;r.upper=old.upper
                r.lower=max(r.lower,old.lower)
            self.cache[ids]=r;self.calls+=1
        return self.cache[ids]
    def classify(self,ids,deadline=np.inf):
        r=self.gamma(ids,deadline)
        if r.upper<=self.epsilon:return 'compatible',r
        if r.lower>self.epsilon:return 'incompatible',r
        r=self.gamma(ids,deadline,True)
        return ('compatible' if r.upper<=self.epsilon else ('incompatible' if r.lower>self.epsilon else 'indeterminate')),r

def conic(mb,ids,epsilon,mode='price',budget=np.inf,rho=.5,tight=False,deadline=np.inf):
    f=mb.fam;p=mb.p;ids=np.asarray(ids,int);m=len(ids);d=f.d;u=f.unit;s=np.sqrt(u);b=mb.w
    extra=np.isfinite(budget);price=mode=='price';N=d+m+(0 if price else m+1);ir=d;iz=d+m;iy=iz+1
    c=np.zeros(N);P=sp.csc_matrix((N,N));Qbar=mb.Q
    if price:
        P=sp.block_diag([sp.csc_matrix(Qbar),sp.csc_matrix((N-d,N-d))],format='csc');grad=Qbar@b;grad-=grad.mean();c[:d]=grad/s
    else:
        c[iy:]=p[ids]/rho if mode=='tail' else p[ids]
        if mode=='tail':c[iz]=1
    mats=[];rhs=[];cones=[];A=np.zeros((1,N));A[0,:d]=1
    mats.append(sp.csc_matrix(A));rhs.append(np.zeros(1));cones.append(clarabel.ZeroConeT(1))
    rows=[];bounds=[]
    def add(cols,values,bound):
        z=np.zeros(N);z[cols]=values;rows.append(z);bounds.append(bound)
    for a in range(d):add([a],[-1],b[a]/s);add([a],[1],(f.upper-b[a])/s)
    for j in range(m):
        if price:add([ir+j],[1],epsilon/u)
        elif mode=='tail':add([ir+j,iz,iy+j],[1,-1,-1],0);add([iy+j],[-1],0)
        else:add([ir+j,iy+j],[1,-1],epsilon/u);add([iy+j],[-1],0)
    if mode=='hinge':add([iz],[1],0);add([iz],[-1],0)
    mats.append(sp.csc_matrix(np.array(rows)));rhs.append(np.array(bounds));cones.append(clarabel.NonnegativeConeT(len(rows)));soc_start=1+len(rows)
    Rlo,Rhi=f.regret_bounds(b)
    for k,j in enumerate(ids):
        g=f.Q[j]@b;g-=g.mean();A=np.zeros((d+2,N));A[:2,:d]=g/s;A[:2,ir+k]=-1;A[2:,:d]=-np.sqrt(2)*np.linalg.cholesky(f.Q[j]).T
        const=(Rlo[j]+Rhi[j])/2/u;z=np.r_[1-const,-1-const,np.zeros(d)]
        mats.append(sp.csc_matrix(A));rhs.append(z);cones.append(clarabel.SecondOrderConeT(d+2))
    if extra:
        g=Qbar@b;g-=g.mean();A=np.zeros((d+2,N));A[:2,:d]=g/s;A[2:,:d]=-np.sqrt(2)*np.linalg.cholesky(Qbar).T
        z=np.r_[1+budget/u,-1+budget/u,np.zeros(d)]
        mats.append(sp.csc_matrix(A));rhs.append(z);cones.append(clarabel.SecondOrderConeT(d+2))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
    settings.tol_gap_abs=1e-12 if tight else 1e-10;settings.tol_gap_rel=settings.tol_gap_abs;settings.tol_feas=1e-12
    settings.time_limit=min(30,max(.001,deadline-perf_counter()))
    ans=clarabel.DefaultSolver(P,c,sp.vstack(mats,format='csc'),np.concatenate(rhs),cones,settings).solve()
    if ans.x is None:raise RuntimeError(str(ans.status))
    w=feasible(b+s*np.array(ans.x[:d]),np.full(d,f.upper));raw=np.array(ans.z[soc_start:]).reshape(m+extra,d+2)
    q=np.maximum(0,raw[:m,0]+raw[:m,1]);nu=max(0,float(raw[-1,0]+raw[-1,1])) if extra else 0.
    if price:
        mix=p.copy();mix[ids]+=q;mass=mix.sum();a=f.mean(mix/mass);x=a['w']
        lower=mb.increment(x)[0]+q@(f.regret_bounds(x)[0][ids]-epsilon)-mass*a['gap'];upper=mb.increment(w)[1]
    else:
        q=simplex_caps(q,p[ids]/rho) if mode=='tail' else np.minimum(q,p[ids])
        mix=nu*p;mix[ids]+=q;mass=mix.sum();a=f.mean(mix/mass) if mass>1e-30 else mb.ma;x=a['w']
        lower=float(q@f.regret_bounds(x)[0][ids])+nu*(mb.increment(x)[0]-budget if extra else 0)-mass*a['gap']
        if mode=='hinge':lower-=epsilon*q.sum()
        hi=f.regret_bounds(w)[1][ids];upper=float(tail_weights(hi,p[ids],rho)@hi) if mode=='tail' else float(p[ids]@np.maximum(hi-epsilon,0))
    return w,float(lower),float(upper),dict(solver_status=str(ans.status),iterations=ans.iterations,dual_weights=q,budget_multiplier=nu,solver_primal_residual=float(ans.r_prim),solver_dual_residual=float(ans.r_dual))

class PriceCache:
    def __init__(self,mb,geometry):
        self.mb=mb;self.geometry=geometry;self.key=(mb.fam.hash,geometry.epsilon,digest(mb.p));self.cache={};self.calls=0
    def validate(self,f,p,epsilon):
        normalized=np.asarray(p)/np.sum(p)
        if self.key!=(f.hash,float(epsilon),digest(normalized)):raise ValueError('posterior price cache cannot cross posterior/family/epsilon')
    def get(self,ids,deadline=np.inf,tight=False):
        ids=tuple(sorted(ids));f=self.mb.fam;eps=self.geometry.epsilon
        if not ids:return PriceResult(self.mb.w,0.,0.,True,0.,self.mb.ma['dual_residual'],'empty_MA')
        if ids in self.cache and not tight:return self.cache[ids]
        state,center=self.geometry.classify(ids,deadline)
        if state=='incompatible':r=PriceResult(self.mb.w,np.inf,np.inf,False,0.,0.,'geometrically_incompatible')
        elif state=='indeterminate':r=PriceResult(center.w,0.,np.inf,False,0.,0.,'geometry_indeterminate')
        else:
            if perf_counter()>=deadline:raise TimeoutError('query deadline')
            w,lower,upper,diag=conic(self.mb,ids,eps,tight=tight,deadline=deadline)
            if f.regret_bounds(w)[1][list(ids)].max()>eps:
                lo=0.;hi=1.;target=w.copy()
                for _ in range(45):
                    a=(lo+hi)/2;x=(1-a)*center.w+a*target
                    if f.regret_bounds(x)[1][list(ids)].max()<=eps:lo=a
                    else:hi=a
                w=(1-lo)*center.w+lo*target;upper=self.mb.increment(w)[1];diag['interpolation']=lo
            valid=bool(f.regret_bounds(w)[1][list(ids)].max()<=eps)
            r=PriceResult(w,max(0,lower),upper if valid else np.inf,valid,max(abs(w.sum()-1),-w.min(),w.max()-f.upper,0.),diag['solver_dual_residual'],diag['solver_status'],diag)
        old=self.cache.get(ids)
        if old:
            r.cost_lower=max(old.cost_lower,r.cost_lower)
            if old.cost_upper<r.cost_upper:r.witness_weights=old.witness_weights;r.cost_upper=old.cost_upper;r.epsilon_feasible=old.epsilon_feasible
        self.cache[ids]=r;self.calls+=1;return r

def price_oracle(family,posterior,selected_models,epsilon):
    return PriceCache(MeanBudget(family,posterior),Geometry(family,epsilon)).get(selected_models,perf_counter()+30,tight=True)
