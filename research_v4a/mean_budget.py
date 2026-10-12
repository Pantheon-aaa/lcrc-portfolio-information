"""Centered regrets and independently evaluated primal/dual QP intervals."""
import numpy as np
from core import solve_qp,feasible
from scipy.special import logsumexp
from .common import digest

def qp(Q,upper):
    ans=solve_qp(Q,constraints={'upper':upper});w=ans['w'].copy();d=len(w)
    # Polish the active set; reject any polish that violates primal/dual signs.
    free=(w>1e-8)&(w<upper-1e-8);active=~free
    if free.any():
        K=np.block([[Q[np.ix_(free,free)],np.ones((free.sum(),1))],[np.ones((1,free.sum())),np.zeros((1,1))]])
        rhs=np.r_[-Q[np.ix_(free,active)]@w[active],1-w[active].sum()]
        z=np.linalg.solve(K,rhs);candidate=w.copy();candidate[free]=z[:-1]
        grad=Q@candidate+z[-1]
        if candidate.min()>=-1e-14 and candidate.max()<=upper+1e-14 and np.all(grad[w<1e-8]>=-1e-12) and np.all(grad[w>upper-1e-8]<=1e-12):w=feasible(candidate,np.full(d,upper))
    grad=Q@w;free=(w>1e-8)&(w<upper-1e-8)
    if free.any():y=-float(np.mean(grad[free]))
    else:
        lo=max(-grad[w<1e-8],default=-np.inf);hi=min(-grad[w>upper-1e-8],default=np.inf);y=(lo+hi)/2
    low=np.where(w<1e-8,np.maximum(grad+y,0),0);high=np.where(w>upper-1e-8,np.maximum(-grad-y,0),0)
    residual=grad+y-low+high
    gap=.5*residual@np.linalg.solve(Q,residual)+low@w+high@(upper-w)-y*(w.sum()-1)
    pad=8*np.finfo(float).eps*(abs(.5*w@Q@w)+np.linalg.norm(Q,2)*np.dot(w,w))
    gap=max(0,float(gap))+pad
    value=float(.5*w@Q@w)
    return dict(w=w,value_lb=value-gap,value_ub=value,gap=gap,primal_residual=max(abs(w.sum()-1),-w.min(),w.max()-upper,0.),dual_residual=float(np.max(abs(residual))),status='polished_qp')

class Family:
    def __init__(self,Q,upper=.2,name='family',metadata=None):
        self.Q=np.asarray(Q,float);self.M,self.d,_=self.Q.shape;self.upper=upper;self.name=name;self.metadata=metadata or {}
        if np.linalg.eigvalsh(self.Q).min()<=0 or upper*self.d<1:raise ValueError('invalid family')
        self.hash=digest(np.r_[self.Q.ravel(),upper]);self.oracle_diagnostics=[qp(q,upper) for q in self.Q]
        self.oracle=np.array([a['w'] for a in self.oracle_diagnostics]);self.errors=np.array([a['gap'] for a in self.oracle_diagnostics])
        self.VU=np.array([a['value_ub'] for a in self.oracle_diagnostics]);self.VL=self.VU-self.errors;self.V=(self.VU+self.VL)/2
        self.base=np.ones(self.d)/self.d;self.unit=max(float(self.regret_bounds(self.base)[1].max()),1e-12);self.scale=np.sqrt(self.unit)
        self.chol=np.linalg.cholesky(self.Q);self.inverse=np.linalg.inv(self.Q);self.logdet=np.linalg.slogdet(self.Q)[1]
    def costs(self,w):return .5*np.einsum('i,mij,j->m',w,self.Q,w)
    def regret_bounds(self,w):
        h=np.asarray(w)-self.oracle
        r=.5*np.einsum('mi,mij,mj->m',h,self.Q,h)+np.einsum('mi,mij,mj->m',h,self.Q,self.oracle)
        roundoff=16*np.finfo(float).eps*(np.linalg.norm(h,axis=1)+np.linalg.norm(h,axis=1)**2)*np.linalg.norm(self.Q,axis=(1,2))
        return np.maximum(0,r-roundoff),np.maximum(0,r+self.errors+roundoff)
    def mean(self,p):return qp(np.einsum('m,mij->ij',p,self.Q),self.upper)
    def posterior(self,ll,prior=None):
        b=np.ones(self.M)/self.M if prior is None else prior;logp=np.log(b)+ll;return np.exp(logp-logsumexp(logp))

class MeanBudget:
    def __init__(self,family,posterior):
        self.fam=family;self.p=np.asarray(posterior,float);self.p=self.p/self.p.sum();self.ma=family.mean(self.p);self.w=self.ma['w'];self.Q=np.einsum('m,mij->ij',self.p,family.Q)
        lo,hi=family.regret_bounds(self.w);self.Llo=max(0,float(self.p@lo)-self.ma['gap']);self.Lhi=float(self.p@hi)
        self.scale=max(self.Lhi,family.unit*1e-8,1e-16)
    def increment(self,w):
        h=np.asarray(w)-self.w;delta=float(.5*h@self.Q@h+h@self.Q@self.w)
        pad=16*np.finfo(float).eps*(np.linalg.norm(h)+np.linalg.norm(h)**2)*np.linalg.norm(self.Q)
        return max(0,delta-pad),max(0,delta+self.ma['gap']+pad)
    def bounds(self,beta):return (np.inf,np.inf) if np.isinf(beta) else (beta*self.Llo,beta*self.Lhi)
