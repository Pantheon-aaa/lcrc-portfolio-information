from dataclasses import dataclass, field
import hashlib
from time import perf_counter
import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc
from core import MaskLikelihood, solve_qp


@dataclass
class ScenarioSet:
    Q: np.ndarray
    nodes: np.ndarray
    groups: np.ndarray
    log_reference: np.ndarray
    mu: np.ndarray = None
    source: str = 'uniform-prior nested scrambled Sobol quadrature'
    upper: float = .2
    kappa: float = 0.
    previous: np.ndarray = None
    oracle_lower: np.ndarray = None
    oracle_upper: np.ndarray = None
    oracle_seconds: float = 0.
    _oracle_signature: str = field(default=None,init=False,repr=False)

    def prepare(self):
        self.Q = np.asarray(self.Q, dtype=float)
        if self.mu is None: self.mu = np.zeros(self.Q.shape[:2])
        assert np.isfinite(self.Q).all() and np.linalg.eigvalsh(self.Q).min() > 0
        assert self.upper*self.Q.shape[1] >= 1
        self.log_reference = np.asarray(self.log_reference)-logsumexp(self.log_reference)
        signature=hashlib.sha256()
        for value in (self.Q,self.mu,np.array([self.upper,self.kappa]),
                      np.array([]) if self.previous is None else self.previous):
            array=np.ascontiguousarray(value,dtype=float)
            signature.update(str(array.shape).encode());signature.update(array.tobytes())
        current_signature=signature.hexdigest()
        if self._oracle_signature is not None and self._oracle_signature!=current_signature:
            self.oracle_lower=None;self.oracle_upper=None
        if self.oracle_lower is None:
            tic=perf_counter()
            ans=[solve_qp(q,m,{'upper':self.upper},self.kappa,self.previous) for q,m in zip(self.Q,self.mu)]
            self.oracle_lower=np.array([a['value_lb'] for a in ans])
            self.oracle_upper=np.array([a['value_ub'] for a in ans])
            self.oracle_seconds=perf_counter()-tic
        self._oracle_signature=current_signature
        return self

    def costs(self,w):
        out=.5*np.einsum('i,mij,j->m',w,self.Q,w)-self.mu@w
        if self.kappa: out+=self.kappa*np.abs(w-self.previous).sum()
        return out


@dataclass
class PosteriorState:
    log_joint: np.ndarray
    loglik: np.ndarray
    groups: np.ndarray
    log_alpha: np.ndarray
    log_conditional: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def p(self): return np.exp(self.log_joint)


def posterior(scenarios,Y,log_prior=None,power=1.):
    like=MaskLikelihood(Y)
    ll=np.array([like(q) for q in scenarios.Q])
    return posterior_from_loglik(scenarios,ll,log_prior,power)


def posterior_from_loglik(sc,ll,log_prior=None,power=1.):
    prior=sc.log_reference if log_prior is None else log_prior
    lp=prior+power*np.asarray(ll); lp-=logsumexp(lp)
    ids=np.unique(sc.groups)
    assert np.array_equal(ids,np.arange(len(ids)))
    la=np.array([logsumexp(lp[sc.groups==s]) for s in ids])
    lc=lp-la[sc.groups]
    p=np.exp(lp)
    return PosteriorState(lp,np.asarray(ll),sc.groups,la,lc,{
        'ess':float(1/(p@p)),'max_mass':float(p.max()),'power':power,
        'underflow_nodes':int(np.sum(p==0)),
        'requires_resolution_audit':bool(1/(p@p)<10 or p.max()>.5),
        'interpretation':'approximate posterior' if power==1 else 'generalized posterior'})


def nested_nodes(size=128,seed=0,phi_dim=1,eta_dim=2):
    shapes={32:(4,8),128:(8,16),256:(16,16),512:(16,32),1024:(32,32),
            2048:(16,128),4096:(32,128),8192:(32,256)}
    ns,nj=shapes[size]
    phi=2*qmc.Sobol(phi_dim,scramble=True,seed=seed).random_base2(int(np.log2(ns)))-1
    # The same inner rule for every outer node ensures exact no-information identity.
    eta=2*qmc.Sobol(eta_dim,scramble=True,seed=seed+1).random_base2(int(np.log2(nj)))-1
    nodes=np.array([np.r_[s,j] for s in phi for j in eta])
    return nodes,np.repeat(np.arange(ns),nj),np.full(size,-np.log(size))


def prior_components(nodes):
    """Densities on [-1,1]^p; normalize quadrature separately for each component."""
    from scipy.stats import beta
    x=np.clip((nodes+1)/2,1e-10,1-1e-10)
    center=beta.logpdf(x,3,3).sum(1)-nodes.shape[1]*np.log(2)
    edge=logsumexp(np.stack([beta.logpdf(x,1,5).sum(1),beta.logpdf(x,5,1).sum(1)]),axis=0)-np.log(2)-nodes.shape[1]*np.log(2)
    skew=beta.logpdf(x,5,2).sum(1)-nodes.shape[1]*np.log(2)
    return np.stack([np.zeros(len(nodes)),center,edge,skew])


def component_logmasses(sc):
    x=prior_components(sc.nodes)+sc.log_reference
    return x-logsumexp(x,axis=1)[:,None]


def mixture_logprior(sc,weights):
    with np.errstate(divide='ignore'):
        return logsumexp(component_logmasses(sc)+np.log(weights)[:,None],axis=0)


def fit_eb(log_evidences):
    """Fit only mixture weights to observed development likelihoods, never truth."""
    from scipy.optimize import minimize
    E=np.asarray(log_evidences); E=E-E.max(axis=1)[:,None]; E=np.exp(E)
    def fn(a):
        z=E@a
        return -np.log(z).sum(),-(E/z[:,None]).sum(axis=0)
    res=minimize(fn,np.ones(E.shape[1])/E.shape[1],jac=True,method='SLSQP',
        bounds=[(1e-8,1)]*E.shape[1],constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(len(a))},
        options={'ftol':1e-10,'maxiter':500})
    a=np.maximum(res.x,0); a/=a.sum()
    return a,{'success':bool(res.success),'objective':float(fn(a)[0]),'datasets':len(E)}


def observation_operator(directions,masks):
    rows=[]
    for mask in np.unique(np.asarray(masks),axis=0):
        ids=np.flatnonzero(mask)
        if len(ids):
            tri=np.tril_indices(len(ids))
            rows.append(np.array([e[np.ix_(ids,ids)][tri] for e in directions]).T)
    return np.concatenate(rows) if rows else np.empty((0,len(directions)))


def fisher_information(Q,directions,masks,scales=None):
    p=len(directions); out=np.zeros((p,p))
    scales=np.ones(p) if scales is None else np.asarray(scales)
    for mask,count in zip(*np.unique(masks,axis=0,return_counts=True)):
        ids=np.flatnonzero(mask)
        if not len(ids): continue
        S=Q[np.ix_(ids,ids)]
        mats=np.array([np.linalg.solve(S,e[np.ix_(ids,ids)])*s for e,s in zip(directions,scales)])
        out+=.5*count*np.einsum('aij,bji->ab',mats,mats)
    # One public strong coordinate; eta block adjusted for its nuisance uncertainty.
    eff=out[1:,1:]-out[1:,:1]@np.linalg.pinv(out[:1,:1])@out[:1,1:]
    return out,eff
