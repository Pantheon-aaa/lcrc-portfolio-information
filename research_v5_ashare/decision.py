"""Past-only decision adapter. No evaluator import and no future fields in input."""
from dataclasses import dataclass
import time
import numpy as np
from research_v4a.mean_budget import Family,MeanBudget
from research_v4a.api import solve_budgeted_dcl
from research_v4a.price_oracle import Geometry,PriceCache
from research_v4a.baselines import budget_greedy,budget_surrogate
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
            elif method in ('Hinge','CVaR05','CVaR025'):
                raw=budget_surrogate(self.f,self.inp.p,self.eps,beta,'hinge' if method=='Hinge' else 'tail',.25 if method=='CVaR025' else .5)
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
        final=self.metrics(w,beta);guarded=w if final['budget_certified'] or not np.isfinite(beta) else self.mb.w.copy()
        return dict(method=method,beta=beta,input_hash=self.input_hash,shared_cost_id=self.inp.shared_cost_id,
          raw_target=raw_w,evaluated_target=w,guarded_target=guarded,raw_result=raw,raw_status=error if raw is None else raw['status'],
          fallback_reason=reason,**final,cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall)
