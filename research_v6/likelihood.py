"""Gaussian masked path likelihood, with directed interval model constants.

Raw binary64 simulated observations are retained by seed. Algebraic rounding
enclosures use a conservative operation-count bound; threshold exponentials are
checked by mpmath interval arithmetic before any exclusion is accepted.
"""
import numpy as np
from mpmath import iv
from .models import actions
iv.dps=50
def constants(f):
    out=[]
    for a in actions(f.d):
        parts=[]
        for s in a.masks:
            matrices=[];errors=[];lds=[];lderrors=[]
            for q in f.Q[:,s][:,:,s]:
                A=iv.matrix([[iv.mpf(float(x)) for x in row] for row in q]);inv=A**-1
                det=iv.det(A);ld=iv.log(det)
                low=np.array([[float(inv[i,j].a) for j in range(len(s))] for i in range(len(s))]);hi=np.array([[float(inv[i,j].b) for j in range(len(s))] for i in range(len(s))])
                low=np.nextafter(low,-np.inf);hi=np.nextafter(hi,np.inf);mid=(low+hi)/2
                matrices.append(mid);errors.append(np.maximum(mid-low,hi-mid));lo=float(ld.a);up=float(ld.b);v=(lo+up)/2
                lds.append(v);lderrors.append(max(v-np.nextafter(lo,-np.inf),np.nextafter(up,np.inf)-v))
            parts.append((s,np.array(matrices),np.array(errors),np.array(lds),np.array(lderrors)))
        out.append(parts)
    return out
def increments(X,parts):
    n=len(X);M=len(parts[0][1]);ll=np.zeros((n,M));err=np.zeros((n,M));eps=np.finfo(float).eps
    for s,inv,ie,ld,le in parts:
        x=X[:,s];a=np.abs(x);v=np.einsum('ni,mij,nj->nm',x,inv,x,optimize=False)
        absolute=np.einsum('ni,mij,nj->nm',a,np.abs(inv),a,optimize=False)
        inverse_error=np.einsum('ni,mij,nj->nm',a,ie,a,optimize=False)
        ops=16*len(s)**2+128;gamma=ops*eps/(1-ops*eps)
        z=-.5*(v+ld);ll+=z
        err+=.5*(inverse_error+le)+gamma*(absolute+np.abs(ld)+np.abs(ll)+1)
    return ll,np.nextafter(err,np.inf)
class GaussianStream:
    """Evaluator-owned simulator. Policies receive only likelihood increments."""
    def __init__(self,f,truth,seed_value,model_constants):
        self.f=f;self.truth=truth;self.parts=model_constants
        self.rng=[np.random.default_rng(np.random.SeedSequence([seed_value,a])) for a in range(4)]
        self.chols=[]
        for act in actions(f.d):
            C=np.zeros((f.d,f.d))
            for s in act.masks:C[np.ix_(s,s)]=np.linalg.cholesky(f.Q[truth][np.ix_(s,s)])
            self.chols.append(C)
    def chunk(self,a,n):
        X=self.rng[a].standard_normal((n,self.f.d))@self.chols[a].T
        return increments(X,self.parts[a])
def verified_exclusions(ll,err,C,delta):
    low=[iv.mpf([float(x-e),float(x+e)]) for x,e in zip(ll,err)]
    excluded=[];M=len(ll)
    for j in range(M):
        if C>>j&1:
            total=sum((iv.exp(x-low[j]) for x in low),iv.mpf(0))/M
            if total.a>=iv.mpf(1)/iv.mpf(float(delta)):excluded.append(j)
    return excluded
