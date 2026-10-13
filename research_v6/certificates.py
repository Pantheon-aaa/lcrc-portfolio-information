"""Rational primal/dual certificates for the exact binary64 public matrices.

No conic solver's status is promoted to a proof. Fractions evaluate feasibility,
quadratic forms and weak duality without floating roundoff.
"""
from fractions import Fraction as F
import numpy as np
from research_v3.compatible_qcqp import solve
from .common import clean
def mat(Q):
    # x'Qx equals x'((Q+Q')/2)x, including any floating asymmetry in legacy inputs.
    return [[(F(float(Q[i,j]))+F(float(Q[j,i])))/2 for j in range(len(Q))] for i in range(len(Q))]
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def mv(Q,x):return [dot(row,x) for row in Q]
def quad(Q,x):return dot(x,mv(Q,x))/2
def linear(A,b):
    a=[r.copy()+[v] for r,v in zip(A,b)];n=len(b)
    for k in range(n):
        z=a[k][k]
        if not z:raise ArithmeticError('zero rational pivot')
        for i in range(k+1,n):
            z=a[i][k]/a[k][k]
            for j in range(k+1,n+1):a[i][j]-=z*a[k][j]
    x=[F(0)]*n
    for k in reversed(range(n)):x[k]=(a[k][n]-dot(a[k][k+1:n],x[k+1:]))/a[k][k]
    return x
def rational_weights(w,upper):
    den=2**44;cap=int(F(float(upper))*den);n=np.floor(np.clip(w,0,upper)*den).astype(np.int64)
    rem=den-int(n.sum())
    for i in np.argsort(-np.asarray(w)):
        step=min(rem,cap-int(n[i])) if rem>=0 else max(rem,-int(n[i]));n[i]+=step;rem-=step
        if not rem:break
    if rem:raise ArithmeticError('rational feasible reconstruction failed')
    return [F(int(v),den) for v in n]
def lower(Q,w,upper):
    # Arbitrary feasible dual multipliers give a global constrained-QP lower bound.
    g=mv(Q,w);tol=F(1,10**8);free=[i for i,x in enumerate(w) if x>tol and x<upper-tol]
    y=-sum((g[i] for i in free),F(0))/len(free) if free else -min(g)
    lo=[max(F(0),g[i]+y) if w[i]<tol else F(0) for i in range(len(w))]
    hi=[max(F(0),-g[i]-y) if w[i]>upper-tol else F(0) for i in range(len(w))]
    h=[y-a+b for a,b in zip(lo,hi)];x=linear(Q,h)
    return -dot(h,x)/2-y-upper*sum(hi)
def down(x):return float(np.nextafter(float(x),-np.inf))
def up(x):return float(np.nextafter(float(x),np.inf))
class Geometry:
    def __init__(self,f):
        self.f=f;self.Q=[mat(q) for q in f.Q];self.u=F(float(f.upper));self.cache={};self.VL=[];self.VU=[]
        self.common=all(all(sum(row)==sum(q[0]) for row in q) for q in self.Q) and F(1,f.d)<=self.u
        for Q,w in zip(self.Q,f.oracle):
            if self.common:
                rw=[F(1,f.d)]*f.d;v=quad(Q,rw);self.VL.append(v);self.VU.append(v);continue
            rw=rational_weights(w,f.upper);self.VL.append(lower(Q,rw,self.u));self.VU.append(quad(Q,rw))
    def regret(self,w):
        return [(quad(q,w)-vu,quad(q,w)-vl) for q,vl,vu in zip(self.Q,self.VL,self.VU)]
    def gamma(self,mask):
        if mask in self.cache:return self.cache[mask]
        ids=[j for j in range(self.f.M) if mask>>j&1]
        if not ids:raise ValueError('empty confidence set cannot certify')
        if self.common:
            ans=dict(lower=0.,upper=0.,weights=[1/self.f.d]*self.f.d,rational_weights=[str(F(1,self.f.d))]*self.f.d,regret_lower=[0.]*self.f.M,regret_upper=[0.]*self.f.M,status='exact_common_optimizer',primal_residual=0)
            self.cache[mask]=ans;return ans
        if len(ids)==1:w=self.f.oracle[ids[0]];q=np.ones(1);raw=None
        else:
            raw=solve(self.f,indices=ids,tight=True);w=raw.w;q=raw.diagnostics['dual_weights']
        rw=rational_weights(w,self.f.upper);r=self.regret(rw);U=max(r[j][1] for j in ids)
        qq=[F(float(max(0,x))) for x in q];z=sum(qq);qq=[x/z for x in qq]
        mix=[[sum((a*self.Q[j][k][l] for a,j in zip(qq,ids)),F(0)) for l in range(self.f.d)] for k in range(self.f.d)]
        L=max(F(0),lower(mix,rw,self.u)-dot(qq,[self.VU[j] for j in ids]))
        ans=dict(lower=down(L),upper=up(U),weights=[float(x) for x in rw],rational_weights=[str(x) for x in rw],regret_lower=[down(x[0]) for x in r],regret_upper=[up(x[1]) for x in r],status='exact_rational_weak_duality',solver_status=raw.status if raw else 'singleton',primal_residual=0)
        self.cache[mask]=ans;return ans
    def all(self):
        for mask in range(1,1<<self.f.M):self.gamma(mask)
        return self.cache
