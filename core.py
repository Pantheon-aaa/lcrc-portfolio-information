"""Small CPU portfolio experiments. No evaluator truth is accepted by decision APIs."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
import cvxpy as cp


@dataclass
class DecisionInstance:
    returns_fit: np.ndarray
    observed_fit: np.ndarray
    returns_cal: np.ndarray
    observed_cal: np.ndarray
    budget_matrix: np.ndarray
    budget_rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    risk_unit: float
    prev_drifted_weights: object = None
    forecast_mu: object = None
    timestamp: str = 'synthetic'
    regime_id: str = 'base'


@dataclass
class DecisionResult:
    weights: np.ndarray
    risk_upper: object = None
    risk_lower: object = None
    numerical_gap: object = None
    confidence_level: object = None
    certificate_scope: str = 'none'
    solve_status: str = 'ok'
    fit_seconds: float = 0.
    likelihood_seconds: float = 0.
    solve_seconds: float = 0.
    qp_calls: int = 0
    scenario_count: int = 0
    cell_count: int = 0
    diagnostics: dict = field(default_factory=dict)


def feasible(x, upper):
    """Euclidean projection onto the capped unit simplex, with roundoff repair."""
    lo, hi = np.min(x-upper)-1., np.max(x)+1.
    for _ in range(65):
        mid=(lo+hi)/2
        if np.clip(x-mid,0,upper).sum()>1: lo=mid
        else: hi=mid
    w=np.clip(x-(lo+hi)/2,0,upper)
    rem=1-w.sum()
    j=np.argmax(upper-w if rem>0 else w)
    w[j]+=rem
    return w


def solve_qp(Q, mu=None, constraints=None, kappa=0., b=None):
    """QP oracle with independently evaluated weak-duality bounds.

    Main fast path: simplex + caps, Q SPD, kappa=0. General polyhedral
    constraints / fixed L1 cost use Clarabel and the same explicit dual.
    Bounds include a roundoff cushion, not directed interval arithmetic.
    """
    d=len(Q); mu=np.zeros(d) if mu is None else np.asarray(mu)
    con=constraints or {}; upper=np.broadcast_to(con.get('upper',1.),(d,)).copy()
    A=np.asarray(con.get('A',np.ones((1,d)))); a=np.asarray(con.get('a',np.ones(1)))
    G=np.asarray(con.get('G',np.r_[-np.eye(d),np.eye(d)]))
    h=np.asarray(con.get('h',np.r_[np.zeros(d),upper]))
    chol=cho_factor(Q,lower=True,check_finite=False)
    fast=not any(k in con for k in ('A','a','G','h')) and kappa==0
    z=np.zeros(d)
    if fast:
        inv=cho_solve(chol,np.c_[mu,np.ones(d)],check_finite=False)
        y=(inv[:,0].sum()-1)/inv[:,1].sum()
        w=inv[:,0]-y*inv[:,1]
        if w.min() < -1e-11 or np.max(w-upper)>1e-11:
            opt=minimize(lambda x:.5*x@Q@x-mu@x,feasible(w,upper),
                jac=lambda x:Q@x-mu,method='SLSQP',bounds=list(zip(np.zeros(d),upper)),
                constraints={'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones(d)},
                options={'ftol':1e-13,'maxiter':250})
            w=feasible(opt.x,upper)
            free=(w>1e-7)&(w<upper-1e-7)
            grad=Q@w-mu
            if free.any():
                y=-float(np.mean(grad[free]))
            else:
                # At a vertex no free coordinate identifies the budget
                # multiplier. KKT instead gives an interval: lower-active
                # coordinates require grad+y >= 0, upper-active <= 0.
                at_lower=w<1e-7; at_upper=w>upper-1e-7
                yl=float(np.max(-grad[at_lower])) if at_lower.any() else -np.inf
                yu=float(np.min(-grad[at_upper])) if at_upper.any() else np.inf
                y=(yl+yu)/2 if np.isfinite(yl+yu) else (yl if np.isfinite(yl) else yu)
        else: w=feasible(w,upper)
        grad=Q@w-mu+y
        s=np.r_[np.where(w<1e-7,np.maximum(grad,0),0),
                np.where(w>upper-1e-7,np.maximum(-grad,0),0)]
        y=np.array([y]); status='ok'
    else:
        x=cp.Variable(d)
        cons=[A@x==a,G@x<=h]
        obj=.5*cp.quad_form(x,cp.psd_wrap(Q))-mu@x
        if kappa:
            assert b is not None
            v=cp.Variable(d); cons += [x-b<=v,b-x<=v]; obj+=kappa*cp.sum(v)
        prob=cp.Problem(cp.Minimize(obj),cons)
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11,max_iter=250)
        if x.value is None: raise RuntimeError(prob.status)
        w=x.value; y=cons[0].dual_value; s=np.maximum(cons[1].dual_value,0)
        if kappa: z=np.clip(cons[2].dual_value-cons[3].dual_value,-kappa,kappa)
        status=prob.status
    c=-mu+z+A.T@y+G.T@s
    lb=-.5*c@cho_solve(chol,c,check_finite=False)-a@y-h@s
    if kappa: lb-=z@b
    ub=.5*w@Q@w-mu@w+(kappa*np.abs(w-b).sum() if kappa else 0)
    pad=5e-12*(1+abs(ub)+np.linalg.norm(Q,2)*np.dot(w,w))
    resid=max(float(np.max(np.abs(A@w-a))),float(np.max(G@w-h)),0.)
    return dict(w=w,value_lb=float(lb-pad),value_ub=float(ub+pad),
        primal_residual=resid,dual_residual=float(np.linalg.norm(Q@w-mu+z+A.T@y+G.T@s,np.inf)),
        gap=float(ub-lb+2*pad),status=status)


class OracleCache:
    def __init__(self,upper=1.): self.upper=upper; self.cache={}; self.calls=0
    def get(self,Q,mu=None):
        mu=np.zeros(len(Q)) if mu is None else mu
        key=(Q.tobytes(),mu.tobytes())
        if key not in self.cache:
            self.cache[key]=solve_qp(Q,mu,{'upper':self.upper}); self.calls+=1
        return self.cache[key]


def center(Qs, mus=None, upper=1., cache=None, absolute=False, tol=1e-6, maxiter=100):
    """Concave simplex dual, solved using SLSQP; each evaluation is one QP.

    A valid bound is returned even if optimization stops early. A direct
    QCQP is deliberately kept separate as the independent audit implementation.
    """
    Qs=np.asarray(Qs); M,d,_=Qs.shape
    mus=np.zeros((M,d)) if mus is None else np.asarray(mus)
    cache=cache or OracleCache(upper); start=cache.calls
    oracle=[cache.get(q,m) for q,m in zip(Qs,mus)]
    lbs=np.zeros(M) if absolute else np.array([o['value_lb'] for o in oracle])
    ubs=np.zeros(M) if absolute else np.array([o['value_ub'] for o in oracle])
    best={'U':np.inf,'L':-np.inf}; last={}
    def evaluate(pi):
        key=pi.tobytes()
        if key==last.get('key'): return last['val']
        pi=np.maximum(pi,0); pi/=pi.sum()
        q=np.einsum('m,mij->ij',pi,Qs); mu=pi@mus
        o=cache.get(q,mu); w=o['w']
        costs=.5*np.einsum('i,mij,j->m',w,Qs,w)-mus@w
        U=float(np.max(costs-lbs)); L=float(o['value_lb']-pi@ubs)
        if U<best['U']: best.update(U=U,w=w.copy())
        if L>best['L']: best.update(L=L,pi=pi.copy())
        vals=costs-(lbs+ubs)/2
        ans=(-float(pi@vals),-vals)
        last.update(key=key,val=ans); return ans
    initial=np.ones(M)/M
    initial_value,initial_gradient=evaluate(initial)
    scale=max(float(np.ptp(initial_gradient)),abs(initial_value),1e-5)
    def scaled(pi):
        value,gradient=evaluate(pi)
        return value/scale,gradient/scale
    opt=minimize(scaled,initial,jac=True,method='SLSQP',bounds=[(0,1)]*M,
        constraints={'type':'eq','fun':lambda p:p.sum()-1,'jac':lambda p:np.ones(M)},
        options={'ftol':min(tol/scale*.001,1e-13),'maxiter':maxiter})
    evaluate(opt.x)
    return dict(w=best['w'],U=best['U'],L=best['L'],gap=best['U']-best['L'],
        pi=best['pi'],qp_calls=cache.calls-start,iterations=int(opt.nit),
        status='converged' if best['U']-best['L']<=tol else 'gap_open')


def direct_center(Qs,mus=None,upper=1.,absolute=False):
    M,d,_=np.shape(Qs); mus=np.zeros((M,d)) if mus is None else mus
    vals=[0 if absolute else solve_qp(q,m,{'upper':upper})['value_ub'] for q,m in zip(Qs,mus)]
    w=cp.Variable(d); t=cp.Variable()
    cons=[cp.sum(w)==1,w>=0,w<=upper]
    cons += [.5*cp.quad_form(w,cp.psd_wrap(q))-m@w-v<=t for q,m,v in zip(Qs,mus,vals)]
    prob=cp.Problem(cp.Minimize(t),cons)
    prob.solve(solver='CLARABEL',tol_gap_abs=2e-11,tol_feas=2e-11,tol_gap_rel=2e-11,max_iter=300)
    if w.value is None: raise RuntimeError(prob.status)
    return float(t.value),feasible(w.value,np.broadcast_to(upper,(d,)))


class MaskLikelihood:
    def __init__(self,Y):
        masks=np.isfinite(Y); self.groups=[]; self.n=len(Y)
        for mask in np.unique(masks,axis=0):
            ids=np.flatnonzero(mask); rows=Y[np.all(masks==mask,axis=1)][:,ids]
            if len(ids): self.groups.append((ids,len(rows),rows.sum(0),rows.T@rows))
    def value_grad(self,Q,Es=None,mean=None):
        mean=np.zeros(len(Q)) if mean is None else mean
        ll=0.; grad=np.zeros(len(Es)) if Es is not None else None
        for ids,n,total,scatter in self.groups:
            S=Q[np.ix_(ids,ids)]; fac=cho_factor(S,lower=True,check_finite=False)
            mu=mean[ids]; scatter=scatter-np.outer(total,mu)-np.outer(mu,total)+n*np.outer(mu,mu)
            inv=cho_solve(fac,np.eye(len(ids)),check_finite=False)
            ll-=.5*(n*(len(ids)*np.log(2*np.pi)+2*np.log(np.diag(fac[0])).sum())+np.sum(inv*scatter.T))
            if Es is not None:
                B=.5*(inv@scatter@inv-n*inv)
                grad += np.einsum('rij,ji->r',Es[:,ids][:,:,ids],B)
        return float(ll),grad
    def __call__(self,Q): return self.value_grad(Q)[0]


@dataclass
class AffineFamily:
    base: np.ndarray
    directions: np.ndarray
    low: np.ndarray
    high: np.ndarray
    def Q(self,x): return self.base+np.einsum('r,rij->ij',x,self.directions)
    def vertices(self,low=None,high=None):
        lo=self.low if low is None else low; hi=self.high if high is None else high
        return np.array(list(product(*zip(lo,hi))))
    def spd_lower(self,lo,hi):
        return min(np.linalg.eigvalsh(self.Q(v))[0] for v in self.vertices(lo,hi))


def make_family(d=20,seed=812):
    rng=np.random.default_rng(seed); k=d//2
    diag=np.geomspace(.65,2.8,d); rng.shuffle(diag)
    base=np.diag(diag)
    # Observable heteroskedastic direction and two asymmetric cross-block directions.
    E0=np.diag(np.linspace(-1,1,d)); E0/=np.linalg.norm(E0,2)
    Es=[.14*E0]
    for r in range(2):
        a=rng.normal(size=k)+(.8 if r==0 else -.3)
        b=rng.normal(size=d-k)+.7
        H=np.zeros((d,d)); H[:k,k:]=np.outer(a,b); H+=H.T
        H/=np.linalg.norm(H,2); Es.append(.22*H)
    return AffineFamily(base,np.array(Es),-np.ones(3),np.ones(3))


def cell_bound(fam,like,lo,hi):
    c=(lo+hi)/2; h=(hi-lo)/2; Q=fam.Q(c)
    ll,grad=like.value_grad(Q,fam.directions)
    bound1=0.; quad=0.
    # PSD cone convexity makes vertex eigenvalue minima valid throughout a cell.
    for ids,n,total,scatter in like.groups:
        verts=fam.vertices(lo,hi)
        s=min(np.linalg.eigvalsh(fam.Q(v)[np.ix_(ids,ids)])[0] for v in verts)
        if s<=0: raise ValueError('non-SPD cell')
        E=fam.directions[:,ids][:,:,ids]
        norms=np.array([np.linalg.norm(e,2) for e in E]); radius=h@norms
        trace=float(np.trace(scatter))
        bound1+=.5*radius*(trace/s**2+n*len(ids)/s)
        # Uniform absolute Hessian bound for zero known observation mean.
        quad+=.5*radius**2*(.5*n*len(ids)/s**2+trace/s**3)
    bound=min(ll+bound1,ll+np.abs(grad)@h+quad)
    return ll,float(bound+1e-9*(1+abs(bound)))


def affine_mle(fam,like):
    grid=np.array(list(product(*[np.linspace(a,b,3) for a,b in zip(fam.low,fam.high)])))
    scores=np.array([like(fam.Q(x)) for x in grid])
    starts=grid[np.argsort(scores)[-5:]]
    best=(float(scores.max()),grid[scores.argmax()]); success=0
    for x in starts:
        def obj(z):
            val,g=like.value_grad(fam.Q(z),fam.directions); return -val,-g
        fit=minimize(obj,x,jac=True,method='L-BFGS-B',bounds=list(zip(fam.low,fam.high)),
                     options={'ftol':1e-13,'gtol':1e-7,'maxiter':200})
        success+=int(fit.success)
        if -fit.fun>best[0]: best=(-float(fit.fun),fit.x)
    # Exactly unobserved affine coordinates leave every observed submatrix
    # unchanged. Choose their box midpoint, rather than an arbitrary grid edge.
    # This is a likelihood tie convention, not a data-dependent shrinkage fit.
    unidentified=[r for r,e in enumerate(fam.directions) if all(
        np.max(np.abs(e[np.ix_(ids,ids)]))<1e-14 for ids,_,_,_ in like.groups)]
    estimate=best[1].copy()
    for r in unidentified: estimate[r]=(fam.low[r]+fam.high[r])/2
    assert abs(like(fam.Q(estimate))-best[0])<1e-7
    return estimate,{'loglik':best[0],'successful_starts':success,'grid_best':float(scores.max()),
                    'unidentified_coordinate_tie_rule':'box midpoint','unidentified_coordinates':unidentified}


def lcrc_finite(Qs,Y,upper=1.,delta=.05):
    like=MaskLikelihood(Y); ll=np.array([like(q) for q in Qs])
    threshold=logsumexp(ll)-np.log(len(ll))+np.log(delta)
    keep=ll>=threshold; ans=center(np.asarray(Qs)[keep],upper=upper)
    return ans,keep


def lcrc_affine(fam,Y,upper=1.,delta=.05,max_cells=32,risk_unit=.03):
    tic=perf_counter(); like=MaskLikelihood(Y); p=len(fam.low)
    anchors=np.array(list(product(*[np.linspace(a,b,3) for a,b in zip(fam.low,fam.high)])))
    threshold=logsumexp([like(fam.Q(x)) for x in anchors])-np.log(len(anchors))+np.log(delta)
    cells=[(fam.low.copy(),fam.high.copy())]; kept=[]; inner=[]; all_bounds=[]
    # Fixed budget partitions; no result-dependent epsilon stopping in comparisons.
    while cells:
        lo,hi=cells.pop(0); ll,ub=cell_bound(fam,like,lo,hi)
        all_bounds.append((lo.tolist(),hi.tolist(),ll,ub))
        c=(lo+hi)/2
        if ll>=threshold: inner.append(c)
        if ub<threshold: continue
        if len(cells)+len(kept)+1<max_cells:
            d=len(fam.base); P=np.eye(d)-np.ones((d,d))/d; b=np.ones(d)/d
            coeff=np.array([np.linalg.norm(P@e@b)+np.linalg.norm(P@e@P,2) for e in fam.directions])
            axis=int(np.argmax((hi-lo)*coeff)); left=hi.copy(); left[axis]=c[axis]
            right=lo.copy(); right[axis]=c[axis]
            cells.extend([(lo.copy(),left),(right,hi.copy())])
        else: kept.append((lo,hi))
    if not kept: raise RuntimeError('empty outer cover contradicts anchor-mixture construction')
    verts=np.unique(np.concatenate([fam.vertices(lo,hi) for lo,hi in kept]),axis=0)
    for v in verts:
        if like(fam.Q(v))>=threshold: inner.append(v)
    for v in anchors:
        if like(fam.Q(v))>=threshold: inner.append(v)
    elapsed=perf_counter()-tic; tic=perf_counter(); cache=OracleCache(upper)
    Qs=np.array([fam.Q(v) for v in verts])
    ans=center(Qs,upper=upper,cache=cache,tol=1e-4*risk_unit)
    inner_pts=np.unique(inner,axis=0) if inner else np.empty((0,p))
    inner_ans=center([fam.Q(v) for v in inner_pts],upper=upper,cache=cache,tol=1e-4*risk_unit) if len(inner_pts) else None
    volume=sum(np.prod(hi-lo) for lo,hi in kept)/np.prod(fam.high-fam.low)
    result=DecisionResult(ans['w'],ans['U'],None if inner_ans is None else inner_ans['L'],ans['gap'],
        1-delta,'affine_exact',ans['status'],likelihood_seconds=elapsed,solve_seconds=perf_counter()-tic,
        qp_calls=cache.calls,scenario_count=len(verts),cell_count=len(kept),diagnostics={
        'outer_set_bound':ans['U'],'outer_set_lower':ans['L'],
        'inner_set_bound':None if inner_ans is None else inner_ans['L'],
        'refinement_gap':None if inner_ans is None else ans['U']-inner_ans['L'],
        'volume_fraction':volume,'threshold':float(threshold),'inner_count':len(inner_pts),
        'cells':[(lo.tolist(),hi.tolist()) for lo,hi in kept],'cell_bounds':all_bounds})
    return result,Qs,like


def model_average(fam,like,order=5):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    grid=np.array(list(product(range(order),repeat=len(fam.low))))
    xs=(fam.low+fam.high)/2+nodes[grid]*(fam.high-fam.low)/2
    # Product uniform-prior quadrature weights, independent of cover refinement.
    lw=np.log(weights[grid]/2).sum(1)
    Qs=np.array([fam.Q(x) for x in xs]); ll=np.array([like(q) for q in Qs])+lw
    probs=np.exp(ll-logsumexp(ll)); return np.einsum('m,mij->ij',probs,Qs)


def em_cov(Y,maxiter=100):
    d=Y.shape[1]; diag=np.nanmean(Y**2,axis=0); Q=np.diag(np.maximum(diag,1e-5))
    masks=np.isfinite(Y); groups=[]
    for mask in np.unique(masks,axis=0):
        rows=Y[np.all(masks==mask,axis=1)]
        if mask.any(): groups.append((mask,rows))
    n=sum(len(rows) for _,rows in groups)
    for it in range(maxiter):
        total=np.zeros((d,d))
        for mask,rows in groups:
            obs=np.flatnonzero(mask); miss=np.flatnonzero(~mask); xo=rows[:,obs]
            full=np.zeros((len(rows),d)); full[:,obs]=xo
            if len(miss):
                B=np.linalg.solve(Q[np.ix_(obs,obs)],Q[np.ix_(obs,miss)]).T
                full[:,miss]=xo@B.T
                total[np.ix_(miss,miss)]+=len(rows)*(Q[np.ix_(miss,miss)]-B@Q[np.ix_(obs,miss)])
            total+=full.T@full
        new=total/n; eig,V=np.linalg.eigh((new+new.T)/2); new=(V*np.maximum(eig,1e-7))@V.T
        if np.linalg.norm(new-Q)/np.linalg.norm(Q)<1e-6: Q=new; break
        Q=new
    return Q,{'em_iterations':it+1,'em_converged':it+1<maxiter}


def sample_data(Q,nblocks,m,rng,distribution='gaussian'):
    d=len(Q); k=d//2; Z=rng.normal(size=(2*nblocks,d))
    if distribution=='t6': Z*=np.sqrt(4/rng.chisquare(6,size=(len(Z),1)))
    Y=Z@np.linalg.cholesky(Q).T
    mask=np.zeros_like(Y,dtype=bool)
    mask[:2*m:2,:]=True
    mask[2*m::2,:k]=True; mask[2*m+1::2,k:]=True
    Y[~mask]=np.nan; return Y
