from dataclasses import dataclass
import numpy as np
from research_v4a.mean_budget import Family
from research_v4a.families import make as old_make,analytic
from research_v3.compat_models import e1
from .common import seed,KINDS
@dataclass(frozen=True)
class ObservationAction:
    name:str
    masks:tuple
    scalar_count:int
    independent_times:int
def actions(d):
    k=d//2
    return [ObservationAction('Singleton',tuple((i,) for i in range(d)),d,d),
      ObservationAction('Separated',(tuple(range(k)),tuple(range(k,d))),d,2),
      ObservationAction('Bridge',tuple((i,k+i) for i in range(k))+tuple((i,) for i in range(2*k,d)),d,k+d%2),
      ObservationAction('Joint',(tuple(range(d)),),d,1)]
def make(kind,d=12,M=6,split='development',index=0):
    # Reuse the exact public generator, with a new deterministic seed namespace.
    f=old_make(kind,d,M,f'v6-20261013-{split}',index)
    if not np.array_equal(f.Q,f.Q.transpose(0,2,1)):
        f=Family((f.Q+f.Q.transpose(0,2,1))/2,f.upper,f.name,dict(f.metadata))
    f.name=f'{kind}-{d}-{M}-{split}-{index}';f.metadata.update(split=split,index=index,kind=kind)
    return f
def specs(split):
    return [(k,10 if i%2==0 else 12,6,split,i) for i in range(2 if split=='development' else 4) for k in KINDS]
def equivalence(f,allowed):
    aa=actions(f.d);groups=[]
    for j in range(f.M):
        for g in groups:
            if all(np.array_equal(f.Q[j][np.ix_(s,s)],f.Q[g[0]][np.ix_(s,s)]) for a in allowed for s in aa[a].masks):g.append(j);break
        else:groups.append([j])
    return groups
def kl(f):
    out=np.zeros((4,f.M,f.M))
    for a,act in enumerate(actions(f.d)):
        for s in act.masks:
            qs=f.Q[:,s][:,:,s];inv=np.linalg.inv(qs);ld=np.linalg.slogdet(qs)[1]
            for i in range(f.M):
                for j in range(f.M):
                    if not np.array_equal(qs[i],qs[j]):out[a,i,j]+=.5*(np.trace(inv[j]@qs[i])-len(s)+ld[j]-ld[i])
    return np.maximum(out,0)
