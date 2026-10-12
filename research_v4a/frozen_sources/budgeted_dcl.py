"""Budget-aware compatibility cuts with global lexicographic optimization."""
from time import perf_counter
from itertools import combinations
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache

def result(mb,epsilon,beta,w,lower,upper,secondary,status,start,geo,prices,**extra):
    lo,hi=mb.fam.regret_bounds(w);dl,du=mb.increment(w);bl,bu=mb.bounds(beta)
    return dict(weights=w,ma_weights=mb.w,success_mass_lower=float(mb.p[hi<=epsilon].sum()),success_mass_upper=float(mb.p[lo<=epsilon].sum()),
      optimal_mass_lower=float(lower),optimal_mass_upper=float(max(lower,upper)),primary_mass_gap=float(max(0,upper-lower)),
      secondary_price_gap=secondary,posterior_mean_regret_MA_lower=mb.Llo,posterior_mean_regret_MA_upper=mb.Lhi,
      posterior_mean_regret_method_lower=float(mb.p@lo),posterior_mean_regret_method_upper=float(mb.p@hi),
      mean_increment_lower=dl,mean_increment_upper=du,budget_lower=bl,budget_upper=bu,
      budget_certified=bool(du<=bl or beta==0),status=status,cold_seconds=perf_counter()-start,
      geometry_calls=geo.calls,price_calls=prices.calls,geometry_cuts=len(geo.cuts),price_cuts=len(prices.cache),
      primal_residual=max(abs(w.sum()-1),-w.min(),w.max()-mb.fam.upper,0.),**extra)

def master(prices,beta,deadline,secondary_mass=None):
    mb=prices.mb;p=mb.p;M=len(p);scale=mb.scale;rows=[];lb=[];ub=[]
    for C in prices.geometry.cuts:
        a=np.zeros(M+1);a[list(C)]=1;rows.append(a);lb.append(-np.inf);ub.append(len(C)-1)
    for C,r in prices.cache.items():
        if not C or not np.isfinite(r.cost_lower) or r.cost_lower<=0:continue
        a=np.zeros(M+1);a[list(C)]=r.cost_lower/scale;a[-1]=-1
        rows.append(a);lb.append(-np.inf);ub.append(r.cost_lower/scale*(len(C)-1))
    if secondary_mass is not None:
        a=np.r_[p,0.];rows.append(a);lb.append(secondary_mass);ub.append(np.inf)
    objective=np.r_[-p,0.] if secondary_mass is None else np.r_[np.zeros(M),1.]
    constraint=LinearConstraint(np.array(rows),lb,ub) if rows else None
    bound=mb.bounds(beta)[1]/scale
    remaining=deadline-perf_counter()
    if remaining<=0:raise TimeoutError('query deadline')
    ans=milp(objective,integrality=np.r_[np.ones(M),0],bounds=Bounds(np.zeros(M+1),np.r_[np.ones(M),bound]),constraints=constraint,
      options={'time_limit':remaining,'mip_rel_gap':0.})
    if ans.x is None:raise TimeoutError(f'master {ans.message}')
    return tuple(np.flatnonzero(ans.x[:M]>.5)),ans

def learn(prices,G,deadline):
    geo=prices.geometry;state,_=geo.classify(G,deadline)
    if state=='incompatible':
        C=list(G)
        for j in G:
            trial=[k for k in C if k!=j]
            if trial and geo.classify(trial,deadline)[0]=='incompatible':C=trial
        C=tuple(C)
        if C not in geo.cuts:geo.cuts.append(C)
        return None
    return prices.get(G,deadline)

def solve_budgeted_dcl(family,posterior,epsilon,beta,geometry_cache=None,posterior_price_cache=None,time_limit_seconds=30.):
    start=perf_counter();deadline=start+time_limit_seconds
    geo=geometry_cache or Geometry(family,epsilon)
    if geo.key!=(family.hash,float(epsilon)):raise ValueError('geometry cache signature mismatch')
    prices=posterior_price_cache or PriceCache(MeanBudget(family,posterior),geo);prices.validate(family,posterior,epsilon);mb=prices.mb;p=mb.p
    w=mb.w.copy();best=float(p[family.regret_bounds(w)[1]<=epsilon].sum());upper=1.;bestcost=0.;secondary=np.inf;status='timeout'
    if beta==0:return result(mb,epsilon,beta,w,best,float(p[family.regret_bounds(w)[0]<=epsilon].sum()),0.,'MA_endpoint',start,geo,prices)
    if mb.Llo<=0 and np.isfinite(beta):return result(mb,epsilon,beta,w,best,1.,np.inf,'budget_precision_limited',start,geo,prices)
    try:
        while perf_counter()<deadline:
            G,opt=master(prices,beta,deadline);upper=min(upper,float(-opt.mip_dual_bound)+1e-10)
            if upper-best<=1e-8:break
            old=prices.cache.get(G);r=learn(prices,G,deadline)
            if r is None:continue
            bl,bu=mb.bounds(beta)
            if r.cost_upper<=bl and r.epsilon_feasible:
                mass=float(p[family.regret_bounds(r.witness_weights)[1]<=epsilon].sum())
                if mass>best+1e-12 or (abs(mass-best)<=1e-12 and r.cost_upper<bestcost):best=mass;w=r.witness_weights;bestcost=r.cost_upper
                if upper-best<=1e-8:break
            elif r.cost_lower>bu:
                if old is not None and old.cost_lower>=r.cost_lower:
                    status='master_precision_limited';break
            else:
                r=prices.get(G,deadline,True)
                if r.cost_upper<=bl and r.epsilon_feasible:continue
                if r.cost_lower>bu:continue
                status='budget_precision_limited' if r.status!='geometry_indeterminate' else r.status;break
        if upper-best<=1e-8:
            status='success_optimal_secondary_open';secondary_lower=0.
            if best<=float(p[family.regret_bounds(mb.w)[1]<=epsilon].sum())+1e-12:
                w=mb.w;bestcost=0.;secondary=0.;status='optimal'
            else:
                while perf_counter()<deadline:
                    G,opt=master(prices,beta,deadline,secondary_mass=best)
                    secondary_lower=max(secondary_lower,float(opt.mip_dual_bound)*mb.scale)
                    secondary=max(0,bestcost-secondary_lower)
                    if secondary<=1e-6*max(mb.Llo,1e-30):status='optimal';break
                    if float(p[list(G)].sum())<best-1e-12:break
                    old=prices.cache.get(G);r=learn(prices,G,deadline)
                    if r is None:continue
                    if r.epsilon_feasible and r.cost_upper<=mb.bounds(beta)[0]:
                        mass=float(p[family.regret_bounds(r.witness_weights)[1]<=epsilon].sum())
                        if mass>=best-1e-12 and r.cost_upper<bestcost:w=r.witness_weights;bestcost=r.cost_upper
                    if old is not None:
                        r=prices.get(G,deadline,True)
                        if r.epsilon_feasible and r.cost_upper<=mb.bounds(beta)[0] and r.cost_upper<bestcost:
                            w=r.witness_weights;bestcost=r.cost_upper
                        secondary=max(0,bestcost-secondary_lower)
                        if secondary<=1e-6*max(mb.Llo,1e-30):status='optimal'
                        else:status='success_optimal_secondary_open'
                        break
    except TimeoutError:status='success_optimal_secondary_open' if upper-best<=1e-8 else 'timeout'
    return result(mb,epsilon,beta,w,best,upper,secondary,status,start,geo,prices)

def enumerate_frontier(family,posterior,epsilon,betas,time_limit=30.):
    start=perf_counter();deadline=start+time_limit;geo=Geometry(family,epsilon);prices=PriceCache(MeanBudget(family,posterior),geo);mb=prices.mb
    for size in range(family.M+1):
        for G in combinations(range(family.M),size):
            if perf_counter()>deadline:raise TimeoutError('enumeration deadline')
            prices.get(G,deadline,tight=True)
    results=[]
    for beta in betas:
        bl,bu=mb.bounds(beta);feasible=[(float(mb.p[list(G)].sum()),r.cost_upper,G,r) for G,r in prices.cache.items() if r.cost_upper<=bl and r.epsilon_feasible]
        feasible.append((float(mb.p[family.regret_bounds(mb.w)[1]<=epsilon].sum()),0.,(),prices.get(())))
        feasible.sort(key=lambda x:(-x[0],x[1],x[2]));mass,cost,G,r=feasible[0]
        upper=max([mass]+[float(mb.p[list(G)].sum()) for G,r in prices.cache.items() if r.cost_lower<=bu])
        lowercost=min([r.cost_lower for G,r in prices.cache.items() if float(mb.p[list(G)].sum())>=mass-1e-12]+[cost])
        results.append(result(mb,epsilon,beta,r.witness_weights,mass,upper,max(0,cost-lowercost),'enumerated',start,geo,prices,selected=G))
    return results,prices
