"""Post-run reconstruction of stop bounds from public Q and stored rational witnesses.

This does not trust the saved numerical bound or the conic solver status. It is
an audit, not a new decision method, and never changes a trajectory.
"""
from fractions import Fraction as F
from .certificates import mat,quad,mv,linear,dot

def positive_definite(Q):
    A=[r.copy() for r in Q]
    for k in range(len(A)):
        if A[k][k]<=0:return False
        for i in range(k+1,len(A)):
            for j in range(k+1,len(A)):
                A[i][j]-=A[i][k]*A[k][j]/A[k][k]
    return True

def dual(Q,w,u):
    g=mv(Q,w);tol=F(1,10**8)
    free=[i for i,x in enumerate(w) if tol<x<u-tol]
    y=-sum((g[i] for i in free),F(0))/len(free) if free else -min(g)
    a=[max(F(0),g[i]+y) if w[i]<tol else F(0) for i in range(len(w))]
    b=[max(F(0),-g[i]-y) if w[i]>u-tol else F(0) for i in range(len(w))]
    h=[y-x+z for x,z in zip(a,b)]
    value=-dot(h,linear(Q,h))/2-y-u*sum(b)
    return value,dict(equality=str(y),lower_multipliers=list(map(str,a)),upper_multipliers=list(map(str,b)),value=str(value))

def verify(f,geometry):
    Q=list(map(mat,f.Q));u=F(float(f.upper));errors=[];oracles=[];vl=[];vu=[]
    for j,q in enumerate(Q):
        if not positive_definite(q):errors.append(f'nonpositive Q {j}')
        w=list(map(F,geometry[1<<j]['rational_weights']))
        if sum(w)!=1 or any(x<0 or x>u for x in w):errors.append(f'oracle feasibility {j}')
        low,d=dual(q,w,u);high=quad(q,w);vl.append(low);vu.append(high)
        oracles.append(dict(model=j,weights=list(map(str,w)),lower=str(low),upper=str(high),dual=d))
    for mask,g in geometry.items():
        w=list(map(F,g['rational_weights']))
        if sum(w)!=1 or any(x<0 or x>u for x in w):errors.append(f'witness feasibility {mask}')
        vals=[quad(q,w) for q in Q]
        for j,v in enumerate(vals):
            if F(g['regret_upper'][j])<v-vl[j]:errors.append(f'regret upper {mask}/{j}')
            if F(g['regret_lower'][j])>v-vu[j]:errors.append(f'regret lower {mask}/{j}')
        if F(g['upper'])<max(vals[j]-vl[j] for j in range(f.M) if mask>>j&1):errors.append(f'Gamma upper {mask}')
    return dict(family=f.name,input_hash=f.hash,passed=not errors,errors=errors,subsets=len(geometry),oracles=oracles,scope='exact SPD, feasibility, QP weak duality and all regret/stop upper bounds; no replacement of stored Gamma lower-bound optimization')
