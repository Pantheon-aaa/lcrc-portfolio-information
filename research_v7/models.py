import numpy as np
from research_v4a.families import make as legacy
from research_v4a.mean_budget import Family
from research_v6.certificates import Geometry
from .common import *
KINDS=('asymmetric','block')
def specs(split):return [(k,10 if i%2==0 else 12,i) for i in range(2 if split=='development' else 4) for k in KINDS]
def make(kind,d,index,split):
    raw=legacy(kind,d,6,'v7-20261017-'+split,index)
    return Family((raw.Q+raw.Q.transpose(0,2,1))/2,.2,f'{kind}-{d}-{split}-{index}',raw.metadata)
def cases(f):
    g=Geometry(f).gamma((1<<f.M)-1)['upper'];out=[]
    for j in range(f.M):out.append(dict(id=f'known{j}',panel='correct',kind='known',Q=f.Q,truth=f.Q[j],origin=j,strength=0))
    avg=f.Q.mean(0);out.append(dict(id='convex',panel='correct',kind='convex',Q=f.Q,truth=avg,origin=-1,strength=0))
    for j in range(f.M):out.append(dict(id=f'leave{j}',panel='leave_one',kind='leave_one',Q=np.delete(f.Q,j,axis=0),truth=f.Q[j],origin=j,strength=0))
    rng=np.random.default_rng(seed(f.name,'perturbation'));v=rng.normal(size=f.d);v-=v.mean();v/=np.linalg.norm(v)
    A=rng.normal(size=(f.d,f.d));H=A@A.T;H/=np.linalg.norm(H,2)
    for kind,D in [('blind',np.ones((f.d,f.d))/f.d),('rank1',np.outer(v,v)),('random_psd',H)]:
      for strength in (.25,1.,4.):
        Qstar=avg+strength*g/f.upper*D;Qstar=(Qstar+Qstar.T)/2
        out.append(dict(id=f'{kind}-{strength}',panel='shift',kind=kind,Q=f.Q,truth=Qstar,origin=-1,strength=strength))
    return out
def observations(Qstar,n,design,data_seed):
    rng=np.random.default_rng(data_seed);d=len(Qstar)
    if design=='Joint':return rng.normal(size=(n,d))@np.linalg.cholesky(Qstar).T
    Y=np.full((2*n,d),np.nan);k=d//2
    for offset,S in enumerate((np.arange(k),np.arange(k,d))):Y[offset*n:(offset+1)*n,S]=rng.normal(size=(n,len(S)))@np.linalg.cholesky(Qstar[np.ix_(S,S)]).T
    return Y
def prefix(X,n,design):
    if design=='Joint':return X[:n]
    half=len(X)//2;return np.concatenate([X[:n],X[half:half+n]])
