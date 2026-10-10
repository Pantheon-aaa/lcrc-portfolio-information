"""Public finite model families; evaluator truth is absent from decision objects."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp
from core import solve_qp, MaskLikelihood
from .common import seed

@dataclass
class Family:
    Q: np.ndarray
    upper: float=1.
    name: str='finite'
    def __post_init__(self):
        self.Q=np.asarray(self.Q,float);self.M,self.d,_=self.Q.shape
        if np.linalg.eigvalsh(self.Q).min()<=0 or self.upper*self.d<1:raise ValueError('invalid family')
        ans=[solve_qp(q,constraints={'upper':self.upper}) for q in self.Q]
        self.oracle=np.array([a['w'] for a in ans]);self.VL=np.array([a['value_lb'] for a in ans]);self.VU=np.array([a['value_ub'] for a in ans])
        self.oracle_diagnostics=ans;self.V=(self.VL+self.VU)/2
        self.base=np.ones(self.d)/self.d
        self.unit=max(float(np.max(self.costs(self.base)-self.V)),1e-9)
        self.scale=np.sqrt(self.unit)
        self.chol=np.linalg.cholesky(self.Q)
        self.inverse=np.linalg.inv(self.Q);self.logdet=np.linalg.slogdet(self.Q)[1]
    def costs(self,w):return .5*np.einsum('i,mij,j->m',w,self.Q,w)
    def regret_bounds(self,w):
        f=self.costs(w);return np.maximum(0,f-self.VU),np.maximum(0,f-self.VL)
    def mean(self,p):return solve_qp(np.einsum('m,mij->ij',p,self.Q),constraints={'upper':self.upper})
    def likelihood(self,Y):return np.array([MaskLikelihood(Y)(q) for q in self.Q])
    def posterior(self,ll,prior=None):
        prior=np.ones(self.M)/self.M if prior is None else np.asarray(prior)
        logp=np.log(prior)+ll;return np.exp(logp-logsumexp(logp))

def e1(t):
    d=3;v=np.sqrt(1.5)*(np.eye(d)-np.ones((d,d))/d)
    H=np.array([3*(x[:,None]+x[None,:]) for x in v]);H[:,np.arange(d),np.arange(d)]=0
    return Family(np.eye(d)+t*H,name=f'E1-{t}')

def geometry(kind,d,M,split='development',family_id=0):
    rng=np.random.default_rng(seed('v3-family-v1',kind,d,M,split,family_id))
    if kind=='simplex':
        v=np.sqrt(d/(d-1))*(np.eye(d)-np.ones((d,d))/d)
        if M>d:
            extra=rng.normal(size=(M-d,d));extra-=extra.mean(1,keepdims=True);extra/=np.linalg.norm(extra,axis=1)[:,None]
            v=np.r_[v,extra]
        v=v[:M]
        H=np.array([d/(d-2)*(x[:,None]+x[None,:]) for x in v]);H[:,np.arange(d),np.arange(d)]=0
        H*=.12/max(np.linalg.norm(h,2) for h in H)
        Q=np.eye(d)+H
    elif kind=='asymmetric':
        H=rng.normal(size=(M,d,d));H=(H+H.transpose(0,2,1))/2;H[:,np.arange(d),np.arange(d)]=0
        H*=.25/max(np.linalg.norm(h,2) for h in H)
        D=np.linspace(.8,1.2,d);Q=np.diag(D)+H
    elif kind=='hidden_curvature':
        pairs=[(i,j) for i in range(d) for j in range(i+1,d)]
        A=np.zeros((d,len(pairs)))
        for k,(i,j) in enumerate(pairs):A[i,k]=A[j,k]=1
        N=null_space(A)
        if N.shape[1]==0:raise ValueError('no nontrivial hidden curvature')
        H=np.zeros((M,d,d))
        for k in range(M):
            v=N@rng.normal(size=N.shape[1])
            for a,(i,j) in zip(v,pairs):H[k,i,j]=H[k,j,i]=a
            H[k]*=.35/np.linalg.norm(H[k],2)
        # Common observed diagonal perturbation; hidden base curvature differs.
        Q=np.eye(d)+H+.08*np.diag(np.linspace(-1,1,d))
    else:raise ValueError(kind)
    return Family(Q,name=f'{kind}-{d}-{M}-{split}-{family_id}')

def embedded(d,t):
    k=d//2;J=np.ones((k,k));A=(1-.2/k)*J+.2*np.eye(k);B=(4-.2/k)*J+.2*np.eye(k)
    return Family(np.array([np.block([[A,c*J],[c*J,B]]) for c in (1-t,1.7)]),upper=.2,name=f'embedded-{d}-{t}')

def multidirection(d,M=8,split='development',family_id=0):
    k=d//2;rng=np.random.default_rng(seed('v3-block-v1',d,M,split,family_id))
    a=np.linspace(.6,1.5,k);b=np.linspace(.8,1.8,k)
    A=np.outer(a,a)*.35+np.diag(a*a*.65);B=np.outer(b,b)*.35+np.diag(b*b*.65)
    La=np.linalg.cholesky(A);Lb=np.linalg.cholesky(B)
    Q=[]
    for j in range(M):
        R=rng.normal(size=(k,k));R*=.85/np.linalg.norm(R,2)
        C=La@R@Lb.T;Q.append(np.block([[A,C],[C.T,B]]))
    return Family(np.array(Q),upper=.2,name=f'block-{d}-{M}-{split}-{family_id}')

def kl_matrix(fam):
    out=np.zeros((fam.M,fam.M))
    for i in range(fam.M):
        for j in range(fam.M):out[i,j]=.5*(np.trace(fam.inverse[j]@fam.Q[i])-fam.d+fam.logdet[j]-fam.logdet[i])
    return np.maximum(out,0)

def information_rows(fam,nats,cap=100000):
    K=kl_matrix(fam);positive=K[K>1e-12];minimum=float(positive.min()) if len(positive) else 0.
    m=min(cap,int(np.ceil(nats/minimum))) if nats and minimum else 0
    return m,minimum,m*minimum

def simulate(fam,truth,m,n_singleton,rng):
    """Evaluator-only: actual Gaussian masked returns and exact complete-mask ll."""
    Y=np.full((fam.d*n_singleton+m,fam.d),np.nan)
    for a in range(fam.d):Y[a*n_singleton:(a+1)*n_singleton,a]=rng.normal(size=n_singleton)*np.sqrt(fam.Q[truth,a,a])
    if m:Y[-m:]=rng.normal(size=(m,fam.d))@fam.chol[truth].T
    return Y

def draw_loglik(fam,truth,m,n_singleton,rng):
    """Exact sufficient-statistic likelihood of simulated returns, not a KL surrogate.

    Gaussian scatter is drawn as a Wishart statistic (Bartlett when m>=d).
    This has exactly the distribution of Y.T@Y; gates compare to raw-data paths.
    """
    d=fam.d
    scatter=np.zeros((d,d))
    if m>=d:
        T=np.tril(rng.normal(size=(d,d)),-1);T[np.diag_indices(d)]=np.sqrt(rng.chisquare(m-np.arange(d)))
        L=fam.chol[truth]@T;scatter=L@L.T
    elif m:
        Y=rng.normal(size=(m,d))@fam.chol[truth].T;scatter=Y.T@Y
    ll=-.5*(m*fam.logdet+np.einsum('mij,ji->m',fam.inverse,scatter)+m*d*np.log(2*np.pi))
    if n_singleton:
        sums=fam.Q[truth].diagonal()*rng.chisquare(n_singleton,size=d)
        diag=np.diagonal(fam.Q,axis1=1,axis2=2)
        ll-=.5*np.sum(n_singleton*np.log(2*np.pi*diag)+sums/diag,axis=1)
    return ll
