"""Past-only bootstrap risk experiment, not a correctly specified Bayesian model."""
import time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc, norm
from core import MaskLikelihood, solve_qp, feasible
from ..new_generators import em_initialized, sqrt_spd
from ..scenario_inference import ScenarioSet, posterior_from_loglik
from ..risk_envelopes import RiskSpec
from ..risk_center import solve_center

POWERS=[.1,.25,.5,1.]
SHRINK=[0.,.1,.25,.5,1.]
PARAMS=[.01,.1,1.]

def regularize(Q):
    # Fixed conditioning, independent of test performance, on normalized returns.
    Q=.98*Q+.02*np.diag(np.diag(Q))
    e,V=np.linalg.eigh((Q+Q.T)/2)
    return (V*np.maximum(e,1e-6*np.trace(Q)/len(Q)))@V.T

def bootstrap_ids(n,rng):
    starts=rng.integers(0,n-9,size=int(np.ceil(n/10)))
    return np.concatenate([np.arange(x,x+10) for x in starts])[:n]

def fit_scenarios(past,size,seed):
    start=time.perf_counter(); d=past.shape[1]; k=d//2
    # Strong/weak split uses only observation patterns, not returns or future data.
    counts=np.isfinite(past).sum(0); first=np.argmax(np.isfinite(past),axis=0)
    order=np.lexsort((np.arange(d),first,-counts))
    x=past[:,order]; fit=x[:189]; score=x[189:]
    mean=np.nanmean(fit,axis=0); fit=fit-mean; score=score-mean
    nominal=regularize(em_initialized(fit,maxiter=40))
    A=nominal[:k,:k];B=nominal[k:,k:]
    invA=sqrt_spd(A,True);invB=sqrt_spd(B,True)
    C=invA@nominal[:k,k:]@invB; U,s,Vt=np.linalg.svd(C);s=np.clip(s,0,.85)
    rng=np.random.default_rng(seed)
    samples=[]; diagonal=[]
    # Shared bootstrap pilot calibrates every cross singular direction, not just two.
    for _ in range(16):
        qb=regularize(em_initialized(fit[bootstrap_ids(len(fit),rng)],nominal,maxiter=15))
        samples.append(qb)
        cb=sqrt_spd(qb[:k,:k],True)@qb[:k,k:]@sqrt_spd(qb[k:,k:],True)
        diagonal.append(np.diag(U.T@cb@Vt.T))
    sd=np.std(diagonal,axis=0,ddof=1)
    shapes={128:(8,16),256:(16,16),512:(16,32)}; ns,nj=shapes[size]
    z=norm.ppf(np.clip(qmc.Sobol(k,scramble=True,seed=seed+1).random_base2(int(np.log2(nj))),1e-9,1-1e-9))
    Q=[]
    for i in range(ns):
        qa=samples[i][:k,:k];qb=samples[i][k:,k:]; sa=sqrt_spd(qa);sb=sqrt_spd(qb)
        for j in range(nj):
            sj=np.clip(s+sd*z[j],-.9,.9)
            cross=sa@(U*sj)@Vt@sb
            Q.append(np.block([[qa,cross],[cross.T,qb]]))
    # Restore input security order before exposing the scene to any decision method.
    inverse=np.argsort(order); Q=np.array(Q)[:,inverse][:,:,inverse]
    nominal=nominal[inverse][:,inverse]; mean=mean[inverse]
    sc=ScenarioSet(Q,np.tile(z,(ns,1)),np.repeat(np.arange(ns),nj),np.full(size,-np.log(size)),
        source='nested bootstrap reference; past-only Gaussian/generalized score',upper=.2).prepare()
    like=MaskLikelihood(past[189:]-mean); ll=np.array([like(q) for q in Q])
    diag=dict(fit_seconds=time.perf_counter()-start,oracle_seconds=sc.oracle_seconds,
        cross_bootstrap_sd=sd.tolist(),group_order=order.tolist(),scenario_count=size,
        eigen_min=float(np.linalg.eigvalsh(Q).min()),fit_rows=189,score_rows=63,
        integration='bootstrap outer empirical measure; scrambled normal inner quadrature')
    return sc,ll,nominal,mean,diag

def specs(lam=.01,tau=.1,audit=False):
    out=[RiskSpec('MA','mean',rho=1),
         *[RiskSpec(f'{b}-Tail-{r}',rho=r,conditional=b=='B') for b in ('A','B') for r in (.25,.5)],
         RiskSpec('B-Max','max',conditional=True),RiskSpec('A-Max','max'),
         RiskSpec('C-LP-Max','max',penalty=lam),RiskSpec('C-Soft','soft',temperature=tau)]
    return [s for s in out if s.name in ('MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25')] if audit else out

def stable_solve(sc,post,spec,unit):
    if spec.kind=='max' and spec.penalty>0 and not spec.conditional and not np.any(sc.mu) and sc.kappa==0:
        # A maximum-likelihood atom has offset zero and true regret >= 0 for every w.
        # If an atom's GLOBAL regret upper bound is below its evidence offset, it
        # can never be the maximizer. Removing it is exact, not posterior truncation.
        offset=spec.penalty*unit*(post.loglik.max()-post.loglik)
        upper=.5*np.linalg.eigvalsh(sc.Q)[:,-1]*sc.upper-sc.oracle_lower
        keep=upper+1e-8*unit>=offset
        reduced=ScenarioSet(sc.Q[keep],sc.nodes[keep],np.zeros(keep.sum(),dtype=int),sc.log_reference[keep],
            mu=sc.mu[keep],upper=sc.upper,source=sc.source,oracle_lower=sc.oracle_lower[keep],oracle_upper=sc.oracle_upper[keep]).prepare()
        p=posterior_from_loglik(reduced,post.loglik[keep],power=post.diagnostics['power'])
        a=solve_center(reduced,p,spec,unit,time_limit=60.)
        full=np.zeros(len(sc.Q));full[keep]=a.adversary;a.adversary=full
        a.diagnostics.update(exact_dominated_atoms_removed=int((~keep).sum()),dominance_margin_unit=1e-8)
        return a
    return solve_center(sc,post,spec,unit,time_limit=60.)

def decisions(sc,ll,nominal,past,unit,power=1.,shrink=.25,lam=.01,tau=.1,audit=False):
    post=posterior_from_loglik(sc,ll,power=power); out=[]; d=len(nominal)
    # Strong plug-in baselines can use all 252 past days, including the score segment.
    nominal=regularize(em_initialized(past-np.nanmean(past,axis=0),maxiter=40))
    for name,w in [('EqualWeight',np.ones(d)/d),
                   ('InverseVol',feasible(1/np.sqrt(np.diag(nominal))/np.sum(1/np.sqrt(np.diag(nominal))),np.full(d,.2))),
                   ('EM-Shrink',solve_qp((1-shrink)*nominal+shrink*np.diag(np.diag(nominal)),constraints={'upper':.2})['w'])]:
        out.append(dict(method=name,weights=w.tolist(),status='baseline',gap=None,residual=float(abs(w.sum()-1)),seconds=0.,objective=None))
    if np.isfinite(past).sum(0).min()>=240:
        from sklearn.covariance import LedoitWolf
        complete=past[np.isfinite(past).all(1)]
        if len(complete)>=60:
            Q=LedoitWolf().fit(complete).covariance_; a=solve_qp(Q,constraints={'upper':.2})
            out.append(dict(method='LedoitWolf-CompleteRows',weights=a['w'].tolist(),status='baseline',
                            gap=float(a['value_ub']-a['value_lb']),residual=0.,seconds=0.,objective=None,
                            complete_rows=len(complete)))
    for spec in specs(lam,tau,audit):
        tic=time.perf_counter()
        try:
            a=stable_solve(sc,post,spec,unit)
            out.append(dict(method=spec.name,weights=a.weights.tolist(),status=a.status,gap=a.gap,
                residual=a.residual,seconds=a.seconds,objective=a.objective,lower=a.lower,upper=a.upper,
                adversary_ess=float(1/(a.adversary@a.adversary)),adversary_max=float(a.adversary.max()),
                alpha=np.exp(post.log_alpha).tolist(),diagnostics=a.diagnostics))
        except Exception as e:
            # Explicit failure; evaluator carries previous holdings, not a silent substitute.
            out.append(dict(method=spec.name,weights=None,status='exception',error=repr(e),seconds=time.perf_counter()-tic))
    return out,post.diagnostics

def predictive_log_score(sc,ll,mean,future,power):
    post=posterior_from_loglik(sc,ll,power=power)
    # One mixture model for the entire future evaluation block; no product of row mixtures.
    like=MaskLikelihood(future-mean)
    fl=np.array([like(q) for q in sc.Q])
    return float(logsumexp(post.log_joint+fl))/max(1,np.isfinite(future).sum())
