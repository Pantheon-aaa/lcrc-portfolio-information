"""Pilot-frozen, fixed-time evidence approval. No true model argument."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from .compatible_qcqp import Compatibility
from .dcl_master import DCL

@dataclass
class CandidateLibrary:
    baseline: np.ndarray
    weights: np.ndarray
    dangerous: list
    log_mixtures: list
    pilot_loglik: np.ndarray
    labels: list
    generation: dict

def prepare(fam,pilot_loglik,comp,epsilon):
    p=fam.posterior(pilot_loglik);wb=fam.mean(p)['w'];candidates=[];labels=[]
    for j in sorted(range(fam.M),key=lambda j:(-p[j],j))[:5]:
        w=fam.oracle[j]
        if np.linalg.norm(w-wb)>1e-8 and all(np.linalg.norm(w-z)>1e-8 for z in candidates):candidates.append(w);labels.append(f'oracle-{j}')
    dcl=DCL(comp,epsilon).solve(p)
    w=dcl['w']
    if np.linalg.norm(w-wb)>1e-8 and all(np.linalg.norm(w-z)>1e-8 for z in candidates):candidates.append(w);labels.append('DCL')
    dangers=[];mixtures=[];keep=[];kept_labels=[];changes=[]
    for w,label in zip(candidates,labels):
        diff=fam.costs(w)-fam.costs(wb);pad=1e-11*(1+np.max(np.abs(fam.costs(w))))
        improve=diff < -pad
        if not improve.any():continue
        # Conservatively include every model not certified safe, including near equality.
        dangerous=np.flatnonzero(diff > -pad)
        logweights=np.full(fam.M,-np.inf);logweights[improve]=np.log(np.maximum(p[improve],1e-300));logweights-=logsumexp(logweights)
        dangers.append(dangerous);mixtures.append(logweights);keep.append(w);kept_labels.append(label)
        changes.append(dict(entry=int(np.sum((wb<1e-7)&(w>1e-5))),exit=int(np.sum((wb>1e-5)&(w<1e-7))),
          near_cap=int(np.sum(w>fam.upper-1e-5)),cross_group=float(np.sum(w[fam.d//2:])-np.sum(wb[fam.d//2:]))))
    return CandidateLibrary(wb,np.asarray(keep).reshape(-1,fam.d),dangers,mixtures,np.asarray(pilot_loglik).copy(),kept_labels,
        dict(dcl_status=dcl['status'],geometry=changes,epsilon=epsilon))

def approve(fam,library,gate_loglik,t=.1,delta=.05):
    C=len(library.weights);threshold=np.log(C/(delta*t)) if C else np.inf
    approved=[];records=[]
    for c,(danger,logmix) in enumerate(zip(library.dangerous,library.log_mixtures)):
        # A single latent model generates the entire gate dataset.
        logq=logsumexp(logmix+gate_loglik)
        loge=logq-gate_loglik[danger]
        passed=not len(danger) or float(np.min(loge))>=threshold
        if passed:approved.append(c)
        records.append(dict(candidate=c,dangerous=danger,log_evalues=loge,approved=passed,threshold=threshold))
    W=np.vstack([library.baseline,library.weights[approved]])
    posterior=fam.posterior(library.pilot_loglik+gate_loglik);Q=np.einsum('m,mij->ij',posterior,fam.Q)
    if len(W)==1:weights=np.ones(1);status='baseline';gap=0.
    else:
        H=W@Q@W.T;scale=max(float(np.max(abs(H))),1e-8)
        opt=minimize(lambda a:(.5*a@H@a/scale,H@a/scale),np.ones(len(W))/len(W),jac=True,method='SLSQP',
          bounds=[(0,1)]*len(W),constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(len(W))},
          options={'ftol':1e-13,'maxiter':200})
        weights=np.maximum(opt.x,0);weights/=weights.sum();status=str(opt.message)
        grad=H@weights;gap=max(0,float(grad@weights-grad.min()))
    return dict(w=weights@W,approved=approved,records=records,convex_weights=weights,
       objective_gap=gap,status=status,error_budget=delta*t,candidate_budget=delta*t/max(C,1))

def two_model_action(ll,t,delta=.05):
    approved=float(ll[0]-ll[1])>=np.log(1/(delta*t))
    return (t/(3+2*t) if approved else 0.),approved

def scalar_decisions(p_good,t,ll,delta=.05):
    """Exact one-dimensional embedded GMV rules, including all CVaR kinks."""
    hG=3+2*t;hB=1.6;s=.7;xG=t/hG;V=.5*t*t/hG
    def regret(x):return np.array([.5*hG*(x-xG)**2,.5*hB*x*x+s*x])
    mean=max(0,(p_good*t-(1-p_good)*s)/(p_good*hG+(1-p_good)*hB))
    # R_G=R_B has exactly one relevant root in [0,xG]. Stable quadratic root.
    a=.5*(hG-hB);bb=-(t+s);cc=V
    cross=2*cc/(-bb+np.sqrt(bb*bb-4*a*cc))
    p=np.array([p_good,1-p_good]);qleft=min(1,p_good/.5);qright=max(0,1-(1-p_good)/.5)
    candidates=[0.,xG,cross]
    for q in (qleft,qright):candidates.append(np.clip((q*t-(1-q)*s)/(q*hG+(1-q)*hB),0,1))
    from research_v2.risk_envelopes import tail_weights
    tail=min(candidates,key=lambda x:tail_weights(regret(x),p,.5)@regret(x))
    logq=logsumexp(np.asarray(ll)+np.log(.5));keep=np.flatnonzero(np.asarray(ll)>=logq+np.log(delta))
    lcrc=cross if len(keep)==2 else (xG if keep[0]==0 else 0.)
    gate,unlock=two_model_action(ll,t,delta)
    return {'MA':mean,'CVaR':float(tail),'Minimax':cross,'MLE':xG if ll[0]>=ll[1] else 0.,
       'LCRC':lcrc,'B1':gate,'NoTrade':0.},regret
