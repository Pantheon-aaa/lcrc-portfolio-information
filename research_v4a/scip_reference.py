"""Generic quadratic mixed-integer reference with legal model-specific big-M."""
from time import perf_counter
import numpy as np
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .budgeted_dcl import result

def solve_scip(f,p,epsilon,beta,time_limit=30.):
    from pyscipopt import Model,quicksum
    start=perf_counter();deadline=start+time_limit;mb=MeanBudget(f,p);geo=Geometry(f,epsilon);pc=PriceCache(mb,geo)
    if beta==0:
        from .budgeted_dcl import solve_budgeted_dcl
        return solve_budgeted_dcl(f,p,epsilon,beta)
    if mb.Llo<=0 and np.isfinite(beta):return result(mb,epsilon,beta,mb.w,0.,1.,np.inf,'budget_precision_limited',start,geo,pc)
    u=f.unit;s=np.sqrt(u);b=mb.w;m=Model();m.hideOutput();m.setIntParam('parallel/maxnthreads',1);m.setRealParam('limits/time',time_limit);m.setRealParam('numerics/feastol',1e-9)
    x=[m.addVar(lb=-b[i]/s,ub=(f.upper-b[i])/s,name=f'x{i}') for i in range(f.d)]
    z=[m.addVar(vtype='B',name=f'z{j}') for j in range(f.M)];m.addCons(quicksum(x)==0)
    def quad(Q):
        g=Q@b;g-=g.mean()
        return .5*quicksum(float(Q[i,k])*x[i]*x[k] for i in range(f.d) for k in range(f.d))+quicksum(float(g[i]/s)*x[i] for i in range(f.d))
    lo,hi=f.regret_bounds(b);big=[]
    for j in range(f.M):
        M=max(0,.5*np.linalg.eigvalsh(f.Q[j])[-1]*f.upper-f.VL[j]-epsilon);big.append(M)
        m.addCons(quad(f.Q[j])+float(lo[j]/u)<=epsilon/u+M/u*(1-z[j]))
    delta=quad(mb.Q);B=mb.bounds(beta)[1]
    if np.isfinite(B):m.addCons(delta<=B/u)
    objective=quicksum(float(p[j]*1e6)*z[j] for j in range(f.M));m.setObjective(objective,'maximize');m.setRealParam('limits/time',max(.001,deadline-perf_counter()));m.optimize()
    upper=min(1.,float(m.getDualbound())/1e6+1e-9);w=b.copy();mass=float(p[hi<=epsilon].sum());secondary=np.inf;status='scip_'+str(m.getStatus())
    if m.getNSols():
        sol=m.getBestSol();selected=tuple(j for j in range(f.M) if m.getSolVal(sol,z[j])>.5)
        try:
            r=pc.get(selected,deadline,tight=True)
            if r.epsilon_feasible and r.cost_upper<=mb.bounds(beta)[0]:w=r.witness_weights;mass=float(p[f.regret_bounds(w)[1]<=epsilon].sum())
        except TimeoutError:pass
    if upper-mass<=1e-8 and perf_counter()<deadline:
        m.freeTransform();m.addCons(objective>=mass*1e6);a=m.addVar(lb=0,name='relative_price');m.addCons(delta<=a)
        m.setObjective(a,'minimize');m.setRealParam('limits/time',max(.001,deadline-perf_counter()));m.optimize()
        lower=max(0,float(m.getDualbound())*u)
        if m.getNSols():
            sol=m.getBestSol();selected=tuple(j for j in range(f.M) if m.getSolVal(sol,z[j])>.5)
            try:
                r=pc.get(selected,deadline,tight=True)
                if r.epsilon_feasible and r.cost_upper<=mb.bounds(beta)[0] and float(p[f.regret_bounds(r.witness_weights)[1]<=epsilon].sum())>=mass-1e-12 and r.cost_upper<mb.increment(w)[1]:w=r.witness_weights
            except TimeoutError:pass
        secondary=max(0,mb.increment(w)[1]-lower)
        status='optimal' if secondary<=1e-6*max(mb.Llo,1e-30) else 'success_optimal_secondary_open'
    return result(mb,epsilon,beta,w,mass,upper,secondary,status,start,geo,pc,big_M=big)
