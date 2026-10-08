from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.stats import qmc
from scipy.special import logsumexp
from core import make_family, sample_data, em_cov
from .scenario_inference import ScenarioSet,nested_nodes


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')


@dataclass
class BlockFamily:
    diagonal: np.ndarray
    trends: np.ndarray
    U: np.ndarray
    V: np.ndarray
    eta_rank: int = 2

    def Q(self,theta):
        k=len(self.diagonal)//2
        diag=self.diagonal*np.exp(.6*theta[0]*self.trends)
        C=(self.U*(.8*np.asarray(theta[1:])))@self.V.T
        Q=np.diag(diag); cross=np.sqrt(diag[:k,None])*C*np.sqrt(diag[None,k:])
        Q[:k,k:]=cross; Q[k:,:k]=cross.T
        return Q


def make_block_family(seed,d=20,rank=2):
    rng=np.random.default_rng(seed); k=d//2
    diag=np.geomspace(.35,3.5,d); rng.shuffle(diag)
    trends=np.r_[np.linspace(-1,.8,k),np.linspace(.9,-.8,d-k)]
    U=np.linalg.qr(rng.normal(size=(k,rank))+.5)[0]
    V=U.copy() if seed%2==0 and d-k==k else np.linalg.qr(rng.normal(size=(d-k,rank))+.3)[0]
    return BlockFamily(diag,trends,U,V,rank)


def family(kind,seed,d=20):
    return make_family(d,seed) if kind=='G1' else make_block_family(seed,d)


def scenarios(fam,size=128,seed=0,upper=.2):
    rank=getattr(fam,'eta_rank',2)
    nodes,groups,lr=nested_nodes(size,seed,eta_dim=rank)
    return ScenarioSet(np.array([fam.Q(x) for x in nodes]),nodes,groups,lr,upper=upper).prepare()


def importance_scenarios(fam,Y,size=256,seed=0,upper=.2):
    """Nested deterministic-mixture importance quadrature, with explicit density ratio.

    Proposal fits use only Y. A 25% uniform component retains full parameter support.
    Both proposal component identities and posterior evidence are used exactly once.
    """
    from scipy.optimize import minimize_scalar,minimize,brentq
    from scipy.stats import truncnorm
    from core import MaskLikelihood
    nodes0,groups,_=nested_nodes(size,seed,eta_dim=fam.eta_rank)
    ns=int(groups.max()+1);nj=size//ns;rank=fam.eta_rank
    # Marginal-block composite likelihood is used ONLY to fit a proposal.
    k=Y.shape[1]//2
    marginal=np.r_[np.c_[Y[:,:k],np.full_like(Y[:,k:],np.nan)],
                   np.c_[np.full_like(Y[:,:k],np.nan),Y[:,k:]]]
    marginal_like=MaskLikelihood(marginal);like=MaskLikelihood(Y)
    def outer_log(x):return marginal_like(fam.Q(np.r_[x,np.zeros(rank)]))
    opt=minimize_scalar(lambda x:-outer_log(x),bounds=(-1,1),method='bounded',options={'xatol':1e-7})
    mean=float(opt.x);step=.001
    curvature=-(outer_log(mean+step)-2*outer_log(mean)+outer_log(mean-step))/step**2
    sd=float(np.clip(1/np.sqrt(max(curvature,1e-6)),.015,.8))
    outer_dist=truncnorm((-1-mean)/sd,(1-mean)/sd,loc=mean,scale=sd)
    # Integrate the mixture CDF itself, rather than allocating just two uniform
    # nodes that can occasionally land on the narrow posterior peak.
    gl,wl=np.polynomial.legendre.leggauss(ns)
    phi=np.array([brentq(lambda x:.75*outer_dist.cdf(x)+.25*(x+1)/2-u,-1,1)
                  for u in (gl+1)/2])
    outer_weights=wl/2
    phi_density=.75*outer_dist.pdf(phi)+.25/2
    inner_u=qmc.Sobol(rank,scramble=True,seed=seed+1).random_base2(int(np.log2(nj)))
    records=[];logref=[];group_ids=[];diagnostics=[]
    joint=any(len(ids)>k for ids,_,_,_ in like.groups)
    # Block Gaussian cross-correlation scoring is independent of phi after whitening,
    # but we deliberately refit the proposal at every outer node, not share truth.
    previous=np.zeros(rank)
    for s,x in enumerate(phi):
        if not joint:
            eta=2*inner_u-1; density=np.full(nj,1/2**rank);mu=np.zeros(rank);sigma=np.ones(rank)
        else:
            def nll(e):return -like(fam.Q(np.r_[x,e]))
            starts=[previous,np.zeros(rank)]
            fits=[minimize(nll,a,method='L-BFGS-B',bounds=[(-1,1)]*rank,
                           options={'ftol':1e-10,'maxiter':60}) for a in starts]
            fit=min(fits,key=lambda z:z.fun);mu=fit.x.copy();previous=mu.copy();sigma=[]
            for j in range(rank):
                h=np.eye(rank)[j]*.002
                # Curvature probes remain in a larger SPD interval by shrinking at edges.
                c0=np.clip(mu,-.995,.995)
                hess=(nll(c0+h)-2*nll(c0)+nll(c0-h))/.002**2
                sigma.append(float(np.clip(1/np.sqrt(max(hess,1e-6)),.08,1.)))
            sigma=np.array(sigma);dist=truncnorm((-1-mu)/sigma,(1-mu)/sigma,loc=mu,scale=sigma)
            ni=3*nj//4
            eta=np.r_[dist.ppf(inner_u[:ni]),2*inner_u[ni:]-1]
            density=.75*np.prod(dist.pdf(eta),axis=1)+.25/2**rank
        for j,e in enumerate(eta):
            records.append(np.r_[x,e]);group_ids.append(s)
            logref.append(-(rank+1)*np.log(2)-np.log(phi_density[s])-np.log(density[j])+np.log(outer_weights[s])-np.log(nj))
        diagnostics.append({'phi':x,'eta_mode':mu.tolist(),'eta_scale':sigma.tolist()})
    nodes=np.array(records);sc=ScenarioSet(np.array([fam.Q(x) for x in nodes]),nodes,np.array(group_ids),np.array(logref),
        source='nested deterministic-mixture importance quadrature; uniform target prior / explicit fitted proposal',upper=upper).prepare()
    sc.proposal_diagnostics={'outer_mean':mean,'outer_sd':sd,'uniform_mixture_fraction':.25,'inner_fits':diagnostics,
        'reference_integral_estimate':float(np.exp(logsumexp(np.array(logref))))}
    return sc


def truth_points(seed,p=3):
    pts=2*qmc.Sobol(p,scramble=True,seed=seed).random_base2(3)-1
    return np.r_[np.zeros((1,p)),np.array([[0]+[.95]*(p-1),[0]+[-.95]*(p-1)]),pts[:5]]


def data_instance(fam,theta,n,m,seed):
    rng=np.random.default_rng(seed)
    return sample_data(fam.Q(theta),n,m,rng)


def factor_path(seed,d,variant,length=504):
    rng=np.random.default_rng(seed); B=rng.normal(size=(d,3))/np.sqrt(3)
    idio=rng.uniform(.25,.8,d); covariances=[]; returns=[]
    for t in range(length):
        state=(t//63)%2
        F=np.eye(3)
        if variant=='correlation': F[0,1]=F[1,0]=(.65 if state else -.25)
        if variant=='volatility': F*=2. if state else .6
        Q=B@F@B.T+np.diag(idio); covariances.append(Q)
        z=rng.normal(size=d)
        if variant=='t6': z*=np.sqrt(4/rng.chisquare(6))
        returns.append(np.linalg.cholesky(Q)@z)
    complete=np.asarray(returns); Y=complete.copy()
    if variant!='complete':
        starts=rng.integers(0,100,d)
        for j,s in enumerate(starts): Y[:s,j]=np.nan
        for t in range(length):
            if (t//14)%3==0: Y[t,:d//2]=np.nan
            if (t//14)%3==1: Y[t,d//2:]=np.nan
    if variant=='informative':
        for j in range(d): Y[complete[:,j]<-np.sqrt(np.mean(np.asarray(covariances)[:,j,j])),j]=np.nan
    return Y,complete,np.asarray(covariances)


def sqrt_spd(Q,inverse=False):
    e,V=np.linalg.eigh((Q+Q.T)/2); e=np.maximum(e,1e-7)
    return (V*(1/np.sqrt(e) if inverse else np.sqrt(e)))@V.T


def estimated_scenarios(past,size=128,seed=0):
    """Nested block bootstrap: no generating covariance/parameters enter this API."""
    fit=past[:126]; score=past[126:]; d=fit.shape[1]; k=d//2
    nominal=em_initialized(fit,maxiter=40)
    Sa=sqrt_spd(nominal[:k,:k]); Sb=sqrt_spd(nominal[k:,k:])
    C=sqrt_spd(nominal[:k,:k],True)@nominal[:k,k:]@sqrt_spd(nominal[k:,k:],True)
    U,s,Vt=np.linalg.svd(C,full_matrices=False); s=np.clip(s,0,.8)
    C=(U*s)@Vt
    nodes,groups,lr=nested_nodes(size,seed)
    rng=np.random.default_rng(seed); ns=groups.max()+1; nj=size//ns; Qs=[]
    # Block resampling estimates outer within-group risks independently of score half.
    for outer in range(ns):
        starts=rng.integers(0,len(fit)-10,size=13)
        ids=np.concatenate([np.arange(i,i+10) for i in starts])[:len(fit)]
        qa=em_initialized(fit[ids,:k],nominal[:k,:k],maxiter=25)
        qb=em_initialized(fit[ids,k:],nominal[k:,k:],maxiter=25)
        A=sqrt_spd(qa); B=sqrt_spd(qb)
        for j in range(nj):
            z=nodes[outer*nj+j,1:]
            ss=s.copy(); ss[:2]=np.clip(ss[:2]+.25*z,-.9,.9)
            cross=A@(U*ss)@Vt@B
            Q=np.block([[qa,cross],[cross.T,qb]])
            Qs.append(Q)
    sc=ScenarioSet(np.array(Qs),nodes,groups,lr,source='nested bootstrap reference, Gaussian score on separate past half',upper=.2).prepare()
    return sc,score,nominal


def em_initialized(Y,initial=None,maxiter=40):
    """Known-zero Gaussian EM; unobserved bootstrap columns retain an explicit
    past-only initialization, never a generated/true covariance value."""
    d=Y.shape[1]
    if initial is None:
        count=np.isfinite(Y).sum(0);total=np.nansum(Y**2,axis=0)
        variance=np.divide(total,count,out=np.full(d,np.nan),where=count>0)
        available=variance[np.isfinite(variance)&(variance>0)]
        scale=float(np.median(available)) if len(available) else 1.
        Q=np.diag(np.where(np.isfinite(variance),np.maximum(variance,1e-7),scale))
    else: Q=initial.copy()
    masks=np.isfinite(Y);groups=[]
    for mask in np.unique(masks,axis=0):
        if mask.any():groups.append((mask,Y[np.all(masks==mask,axis=1)]))
    n=sum(len(rows) for _,rows in groups)
    if n==0:return Q
    for _ in range(maxiter):
        total=np.zeros((d,d))
        for mask,rows in groups:
            obs=np.flatnonzero(mask);miss=np.flatnonzero(~mask);xo=rows[:,obs]
            full=np.zeros((len(rows),d));full[:,obs]=xo
            if len(miss):
                B=np.linalg.solve(Q[np.ix_(obs,obs)],Q[np.ix_(obs,miss)]).T
                full[:,miss]=xo@B.T
                total[np.ix_(miss,miss)]+=len(rows)*(Q[np.ix_(miss,miss)]-B@Q[np.ix_(obs,miss)])
            total+=full.T@full
        new=total/n;e,V=np.linalg.eigh((new+new.T)/2);new=(V*np.maximum(e,1e-7))@V.T
        change=np.linalg.norm(new-Q)/max(np.linalg.norm(Q),1e-10);Q=new
        if change<1e-6:break
    return Q
