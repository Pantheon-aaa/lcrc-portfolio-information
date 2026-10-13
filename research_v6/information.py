import numpy as np
from scipy.optimize import linprog
from .models import kl,equivalence
def allocation(K,theta,outside,cost,allowed):
    A=len(allowed)
    if not outside:return float('inf'),np.eye(4)[allowed[0]]
    # y denotes cost fractions. Pull fractions are proportional to y/cost.
    q=K[np.array(allowed),theta][:,outside].T/np.asarray(cost)[allowed]
    scale=max(float(q.max()),1e-300)
    if not q.any():return 0.,np.array([1/len(allowed) if a in allowed else 0 for a in range(4)])
    r=linprog(np.r_[np.zeros(A),-1],A_ub=np.c_[-q/scale,np.ones(len(outside))],b_ub=np.zeros(len(outside)),A_eq=[np.r_[np.ones(A),0]],b_eq=[1],bounds=[(0,None)]*(A+1),method='highs')
    if not r.success:raise RuntimeError(r.message)
    x=np.maximum(r.x[:A],0)/np.asarray(cost)[allowed];x/=x.sum();p=np.zeros(4);p[allowed]=x
    return max(0,r.x[-1]*scale),p
def tables(f,geodata,eps,cost,allowed):
    M=f.M;size=1<<M;K=kl(f)
    ok=np.zeros(size,bool);bad=np.zeros(size,bool);ok[0]=True
    for s,r in geodata.items():ok[int(s)]=r['upper']<=eps;bad[int(s)]=r['lower']>eps
    maximal=[s for s in range(1,size) if ok[s] and not any(ok[s|(1<<j)] for j in range(M) if not(s>>j&1))]
    if not maximal:raise RuntimeError('no certified answer')
    conflicts=np.zeros(size,np.int64)
    for s in range(1,size):
        if bad[s]:
            z=s
            for j in range(M):
                trial=z&~(1<<j)
                if trial and bad[trial]:z=trial
            conflicts[s]=z
    rate=np.full((M,len(maximal)),-1.);full=np.zeros((M,len(maximal),4));cf=np.zeros((size,M,len(maximal),4));memo={}
    for th in range(M):
        for g,G in enumerate(maximal):
            if not G>>th&1:continue
            rate[th,g],full[th,g]=allocation(K,th,[j for j in range(M) if not G>>j&1],cost,allowed)
            for C in range(1,size):
                H=conflicts[C];outside=tuple(j for j in range(M) if H>>j&1 and not G>>j&1)
                if not outside:cf[C,th,g]=full[th,g];continue
                key=(th,outside)
                if key not in memo:memo[key]=allocation(K,th,list(outside),cost,allowed)[1]
                cf[C,th,g]=memo[key]
    return dict(ok=ok,bad=bad,answers=np.array(maximal),conflicts=conflicts,rates=rate,full=full,conflict=cf,kl=K,classes=equivalence(f,allowed))
def randomized_platform(M,ok,group):
    sets=[s for s in range(1,1<<M) if ok[s]]
    cover=np.array([[float(s>>j&1) for s in sets] for j in group])
    r=linprog(np.r_[np.zeros(len(sets)),-1],A_ub=np.c_[-cover,np.ones(len(group))],b_ub=np.zeros(len(group)),A_eq=[np.r_[np.ones(len(sets)),0]],b_eq=[1],bounds=[(0,None)]*(len(sets)+1),method='highs')
    return dict(success=r.x[-1],failure=1-r.x[-1],sets=sets,probabilities=r.x[:-1])
