import numpy as np
from scipy.special import logsumexp
from core import MaskLikelihood
from research_v3.compat_models import geometry as old_geometry
from .mean_budget import Family
from .common import seed

def analytic(d,fraction):
    c=np.sqrt(d/(d-1));t=fraction/(16*d*c);v=c*(np.eye(d)-np.ones((d,d))/d)
    H=np.array([d/(d-2)*(x[:,None]+x[None,:]) for x in v]);H[:,np.arange(d),np.arange(d)]=0
    f=Family(np.eye(d)+t*H,upper=max(.2,2/d),name=f'analytic-{d}-{fraction}',metadata={'t':t,'fraction':fraction})
    return f,t*t/(2*(1-2*c*t))
def analytic_price(d,t,epsilon):
    c=np.sqrt(d/(d-1));L=t*t/(2*(1-2*c*t));out=[]
    for m in range(1,d):
        eta=(d-m)/(m*(d-1));a=1+2*c*t/(d-2)*(2-d/m);gamma=L-t*t*eta/(2*a);disc=t*t*eta-2*a*(L-epsilon)
        if epsilon>=L:r=0.;cost=0.
        elif disc>=0:r=2*(L-epsilon)/(t*np.sqrt(eta)+np.sqrt(disc));cost=r*r/2
        else:r=np.nan;cost=np.inf
        w=np.ones(d)/d
        if np.isfinite(r):w[:m]-=r*np.sqrt((d-m)/(d*m));w[m:]+=r*np.sqrt(m/(d*(d-m)))
        out.append(dict(m=m,gamma=gamma,cost=cost,beta=cost/L,w=w))
    return out

def make(kind,d=12,M=6,split='development',index=0):
    name=f'{kind}-{d}-{M}-{split}-{index}';s=seed('v4a-20261011-family',name)
    if kind!='block':
        old=old_geometry(kind,d,M,split=f'v4a-{s}',family_id=index)
        return Family(old.Q,.2,name,{'kind':kind,'split':split,'index':index,'seed':s})
    rng=np.random.default_rng(s);k=d//2;Q=[]
    for a in range(2):
        blocks=[];roots=[]
        for _ in range(2):
            U=np.linalg.qr(rng.normal(size=(k,k)))[0];lam=np.linspace(.7,2,k)
            blocks.append((U*lam)@U.T);roots.append((U*np.sqrt(lam))@U.T)
        A,B=blocks
        for b in range(M//2):
            low=rng.normal(size=(k,2))@rng.normal(size=(2,k));dense=rng.normal(size=(k,k))
            C=.5*low/np.linalg.norm(low,2)+.5*dense/np.linalg.norm(dense,2);C*=.6/np.linalg.norm(C,2)
            cross=roots[0]@C@roots[1];Q.append(np.block([[A,cross],[cross.T,B]]))
    return Family(np.array(Q),.2,name,{'kind':kind,'split':split,'index':index,'seed':s})

def masks(f,design):return [np.arange(f.d)] if design=='Joint' else [np.arange(f.d//2),np.arange(f.d//2,f.d)]
def equivalence(f,design):
    groups=[]
    for j in range(f.M):
        for g in groups:
            if all(np.array_equal(f.Q[j][np.ix_(S,S)],f.Q[g[0]][np.ix_(S,S)]) for S in masks(f,design)):g.append(j);break
        else:groups.append([j])
    return groups
def affinity(f,design):
    B=np.zeros((f.M,f.M))
    for S in masks(f,design):
        Q=f.Q[:,S][:,:,S];ld=np.linalg.slogdet(Q)[1]
        for j in range(f.M):
            for k in range(f.M):B[j,k]+=.5*np.linalg.slogdet((Q[j]+Q[k])/2)[1]-.25*(ld[j]+ld[k])
    return np.maximum(0,B)
def observations(f,truth,n,design,data_seed):
    rng=np.random.default_rng(data_seed);Y=np.full((2*n,f.d),np.nan)
    for time,S in enumerate(masks(f,design)):
        Q=f.Q[truth][np.ix_(S,S)];Y[time*n:(time+1)*n,S]=rng.normal(size=(n,len(S)))@np.linalg.cholesky(Q).T
    if n==0:return Y,np.zeros(f.M)
    L=MaskLikelihood(Y);return Y,np.array([L(q) for q in f.Q])
