"""Past-only decision adapter. No evaluator import and no future fields in input."""
from dataclasses import dataclass
import time
import numpy as np
from research_v4a.mean_budget import Family,MeanBudget
from research_v4a.api import solve_budgeted_dcl
from research_v4a.price_oracle import Geometry,PriceCache
from research_v4a.baselines import budget_greedy,budget_surrogate
from research_v4a.budgeted_dcl import result
from .common import array_hash,digest

@dataclass(frozen=True)
class DecisionInput:
    Q:np.ndarray
    p:np.ndarray
    log_p:np.ndarray
    risk_unit:float
    epsilon:float
    upper:float
    shared_cost_id:str

class Solver:
    def __init__(self,inp):
        self.inp=inp;self.start=time.process_time();self.f=Family(inp.Q/inp.risk_unit,inp.upper);self.mb=MeanBudget(self.f,inp.p)
        self.eps=inp.epsilon/inp.risk_unit;self.geo=Geometry(self.f,self.eps);self.pc=PriceCache(self.mb,self.geo)
        self.input_hash=digest(array_hash(inp.Q),array_hash(inp.p),inp.epsilon,inp.upper,inp.risk_unit)
        self.prepare_cpu=time.process_time()-self.start
    def metrics(self,w,beta):
        lo,hi=self.f.regret_bounds(w);dl,du=self.mb.increment(w);bl,bu=self.mb.bounds(beta);u=self.inp.risk_unit
        return dict(success_mass_lower=float(self.mb.p[hi<=self.eps].sum()),success_mass_upper=float(self.mb.p[lo<=self.eps].sum()),
          mean_increment_lower=dl*u,mean_increment_upper=du*u,budget_lower=bl*u,budget_upper=bu*u,
          mean_regret_lower=float(self.mb.p@lo)*u,mean_regret_upper=float(self.mb.p@hi)*u,
          budget_certified=bool(beta==0 or du<=bl),primal_residual=float(max(abs(w.sum()-1),-w.min(),w.max()-self.f.upper,0.)))
    def solve(self,method,beta,limit=30):
        start=time.process_time();wall=time.perf_counter();raw=None;error=None
        try:
            if method=='A1B':raw=solve_budgeted_dcl(self.f,self.inp.p,self.eps,beta,self.geo,self.pc,limit,self.inp.log_p)
            elif method=='MA':raw=dict(weights=self.mb.w.copy(),status='MA')
            elif method=='EW':raw=dict(weights=np.ones(self.f.d)/self.f.d,status='equal_weight')
            elif method=='Greedy':raw=budget_greedy(self.f,self.inp.p,self.eps,beta,self.geo)
            elif method=='Greedy1swap':raw=self.swap(beta,limit)
            elif method in ('Hinge','CVaR05','CVaR025'):
                if method=='Hinge' and np.all(self.f.regret_bounds(self.mb.w)[1]<=self.eps):
                    raw=dict(weights=self.mb.w.copy(),status='hinge_zero_MA_secondary',surrogate_objective_lower=0.,surrogate_objective_upper=0.,secondary_price_gap=self.mb.ma['gap'])
                else:
                    raw=budget_surrogate(self.f,self.inp.p,self.eps,beta,'hinge' if method=='Hinge' else 'tail',.25 if method=='CVaR025' else .5)
                    if method=='Hinge' and np.all(self.f.regret_bounds(raw['weights'])[1]<=self.eps) and time.perf_counter()<wall+limit:
                        try:
                            z=self.pc.get(range(self.f.M),wall+limit,tight=True)
                            if z.epsilon_feasible and z.cost_upper<=self.mb.bounds(beta)[0]:
                                raw['weights']=z.witness_weights;raw['secondary_price_gap']=z.cost_upper-z.cost_lower
                        except TimeoutError:raw['secondary_price_gap']=np.inf
            else:raise ValueError('unknown method '+method)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        raw_w=None if raw is None else np.asarray(raw['weights']);reason=None
        if raw_w is None or not np.isfinite(raw_w).all():reason='no_valid_target'
        else:
            m=self.metrics(raw_w,beta)
            if m['primal_residual']>1e-9:reason='primal_constraint_violation'
            elif method not in ('MA','EW') and np.isfinite(beta) and m['mean_increment_lower']>m['budget_upper']:reason='proven_budget_violation'
        w=self.mb.w.copy() if reason else raw_w.copy()
        if not np.isfinite(w).all() or self.metrics(w,0)['primal_residual']>1e-9:w=np.ones(self.f.d)/self.f.d;reason='MA_invalid_equal_weight'
        final=self.metrics(w,np.inf if method=='EW' else beta);guarded=w if final['budget_certified'] or not np.isfinite(beta) else self.mb.w.copy()
        return dict(method=method,beta=beta,input_hash=self.input_hash,shared_cost_id=self.inp.shared_cost_id,
          raw_target=raw_w,evaluated_target=w,guarded_target=guarded,raw_result=raw,raw_status=error if raw is None else raw['status'],
          fallback_reason=reason,**final,budget_applies=method!='EW',cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall)

    def swap(self,beta,limit):
        start=time.perf_counter();deadline=start+limit;pc=PriceCache(self.mb,self.geo);G=[];r=pc.get(())
        for j in sorted(range(self.f.M),key=lambda j:(-self.mb.p[j],j)):
            if time.perf_counter()>=deadline:break
            try:z=pc.get(G+[j],deadline)
            except TimeoutError:break
            if z.epsilon_feasible and z.cost_upper<=self.mb.bounds(beta)[0]:G.append(j);r=z
        G=tuple(sorted(G));mass=float(self.mb.p[list(G)].sum());changed=True
        while changed and time.perf_counter()<deadline:
            changed=False
            for removed in [None,*G]:
                base=[j for j in G if j!=removed]
                for added in [None,*[j for j in range(self.f.M) if j not in G]]:
                    trial=tuple(sorted(base+([] if added is None else [added])))
                    if trial==G:continue
                    m=float(self.mb.p[list(trial)].sum())
                    if m<mass-1e-12:continue
                    try:z=pc.get(trial,deadline)
                    except TimeoutError:break
                    if z.epsilon_feasible and z.cost_upper<=self.mb.bounds(beta)[0] and (m>mass+1e-12 or (abs(m-mass)<=1e-12 and z.cost_upper<r.cost_upper-1e-12)):
                        G=trial;r=z;mass=m;changed=True;break
                if changed or time.perf_counter()>=deadline:break
        actual=float(self.mb.p[self.f.regret_bounds(r.witness_weights)[1]<=self.eps].sum())
        return result(self.mb,self.eps,beta,r.witness_weights,actual,1.,None,'greedy_1swap' if time.perf_counter()<deadline else 'greedy_1swap_timeout',start,self.geo,pc,selected=G)
