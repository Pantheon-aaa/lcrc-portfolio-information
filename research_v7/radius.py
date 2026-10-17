"""Decision oscillation bounds and finite-look Gaussian calibration.

No unknown covariance or evaluator identity enters calibrate(). Numerical guards
are outward error allowances, never relaxed success/budget thresholds.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.stats import chi2

def outward(x,scale=0.):return float(np.nextafter(x+256*np.finfo(float).eps*abs(scale),np.inf))
def norm_upper(a):
    a=np.asarray(a);v=float(np.linalg.norm(a,2));return outward(v,max(a.shape)*float(np.linalg.norm(a,'fro') if a.ndim==2 else np.linalg.norm(a)))
def projection(d):return np.eye(d)-np.ones((d,d))/d
def envelope(D,cap):
    d=len(D);P=projection(d);w0=np.ones(d)/d;r=np.sqrt(max(0,cap-1/d))
    if not np.any(D):return dict(operator=0.,decision=0.,upper=0.)
    scale=float(np.linalg.norm(D,'fro'));err=512*d*np.finfo(float).eps*scale
    op=outward(cap*norm_upper(D),scale)
    dec=outward(2*r*(norm_upper(P@D@w0)+err)+r*r*(norm_upper(P@D@P)+err),scale)
    return dict(operator=op,decision=dec,upper=min(op,dec),qualification='outward_float_error_allowance')

@lru_cache(None)
def factors(n,d,delta,looks):
    # Both original and projected directions, including b cross moments.
    directions=2*d*d+2*d;alpha=delta/(looks*directions)
    lo=chi2.ppf(alpha/2,n);hi=chi2.ppf(1-alpha/2,n)
    return n/hi,n/lo,alpha
def variance_interval(z,n,d,delta,looks):
    low,high,_=factors(n,d,delta,looks);ss=float(z@z/n)
    err=256*n*np.finfo(float).eps*float(np.abs(z)@np.abs(z)/n)
    return max(0,np.nextafter((ss-err)*low,-np.inf)),np.nextafter((ss+err)*high,np.inf)
def matrix_intervals(X,delta=.05,looks=3):
    n,d=X.shape;L=np.empty((d,d));U=np.empty((d,d))
    diag=[]
    for i in range(d):
        good=np.isfinite(X[:,i]);nn=int(good.sum())
        if not nn:raise ValueError('no observed variance')
        l,u=variance_interval(X[good,i],nn,d,delta,looks);L[i,i]=l;U[i,i]=u;diag.append(u)
    for i in range(d):
      for j in range(i):
        good=np.isfinite(X[:,i])&np.isfinite(X[:,j]);nn=int(good.sum())
        if nn:
            a,b=variance_interval(X[good,i]+X[good,j],nn,d,delta,looks)
            c,e=variance_interval(X[good,i]-X[good,j],nn,d,delta,looks)
            l=np.nextafter((a-e)/4,-np.inf);u=np.nextafter((b-c)/4,np.inf)
        else:u=outward(np.sqrt(diag[i]*diag[j]),diag[i]+diag[j]);l=-u
        L[i,j]=L[j,i]=l;U[i,j]=U[j,i]=u
    return L,U

@dataclass
class CalibratedRadius:
    midpoint:np.ndarray
    halfwidth:np.ndarray
    A:np.ndarray
    EA:np.ndarray
    b:np.ndarray
    eb:np.ndarray
    n:int
    design:str
    delta:float
    looks:int
    def compare(self,Qbar,cap):
        d=len(Qbar);P=projection(d);w0=np.ones(d)/d;r=np.sqrt(max(0,cap-1/d))
        op=outward(cap*(norm_upper(self.midpoint-Qbar)+norm_upper(self.halfwidth)),np.linalg.norm(Qbar))
        dec=outward(2*r*(norm_upper(self.b-P@Qbar@w0)+norm_upper(self.eb))+r*r*(norm_upper(self.A-P@Qbar@P)+norm_upper(self.EA)),np.linalg.norm(Qbar))
        return dict(operator=op,decision=min(dec,op),projected=dec)
    def encloses(self,Q):
        # Evaluator-only diagnostic of the simultaneous confidence event.
        P=projection(len(Q));w0=np.ones(len(Q))/len(Q)
        return bool(np.all(np.abs(Q-self.midpoint)<=self.halfwidth) and np.all(np.abs(P@Q@P-self.A)<=self.EA) and np.all(np.abs(P@Q@w0-self.b)<=self.eb))

def calibrate(X,design,delta=.05,looks=3):
    d=X.shape[1];P=projection(d);w0=np.ones(d)/d
    L,U=matrix_intervals(X,delta,looks);mid=(L+U)/2;E=(U-L)/2
    rounding=1024*d*np.finfo(float).eps*np.maximum(1e-300,np.max(np.abs(U)))
    E+=rounding
    if design=='Joint':
        if not np.isfinite(X).all():raise ValueError('Joint calibration requires fully observed blocks')
        Y=X@P;y0=X@w0;LA,UA=matrix_intervals(Y,delta,looks);A=(LA+UA)/2;EA=(UA-LA)/2+rounding;b=[];eb=[]
        for i in range(d):
            a,z=variance_interval(Y[:,i]+y0,len(Y),d,delta,looks);c,e=variance_interval(Y[:,i]-y0,len(Y),d,delta,looks)
            low=(a-e)/4;high=(z-c)/4;b.append((low+high)/2);eb.append((high-low)/2+rounding)
        b=np.array(b);eb=np.array(eb)
    else:
        A=P@mid@P;EA=np.abs(P)@E@np.abs(P)+rounding;b=P@mid@w0;eb=np.abs(P)@E@w0+rounding
    return CalibratedRadius(mid,E,A,EA,b,eb,int(np.isfinite(X[:,0]).sum()),design,delta,looks)

def witnesses(Q,p):return np.concatenate([Q,np.mean(Q,axis=0)[None],np.einsum('m,mij->ij',p,Q)[None]])
def radius_bounds(cal,Q,p,cap):
    vals=[cal.compare(q,cap) for q in witnesses(Q,p)]
    return {k:dict(upper=min(x[k] for x in vals),witness=int(np.argmin([x[k] for x in vals]))) for k in ('decision','operator')}
def oracle_bounds(truth,Q,p,cap):
    vals=[envelope(truth-q,cap) for q in witnesses(Q,p)]
    return {k:dict(upper=min(x[k] for x in vals),witness=int(np.argmin([x[k] for x in vals]))) for k in ('upper','operator')}
