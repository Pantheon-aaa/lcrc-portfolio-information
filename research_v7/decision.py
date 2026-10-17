from dataclasses import dataclass
import time
import numpy as np
from research_v4a.mean_budget import Family,MeanBudget
from research_v4a.baselines import budget_greedy
from research_v4a.api import solve_budgeted_dcl
from research_v4a.price_oracle import Geometry,PriceCache

@dataclass(frozen=True)
class DecisionInput:
    Q:np.ndarray
    p:np.ndarray
    log_p:np.ndarray
    upper:float
    epsilon:float

def solve(inp,betas=(.1,.25,np.inf),exact=True,family=None):
    f=family or Family(inp.Q,inp.upper);mb=MeanBudget(f,inp.p);geo=Geometry(f,inp.epsilon);pc=PriceCache(mb,geo);out=[]
    sat=bool(np.all(f.regret_bounds(mb.w)[1]<=inp.epsilon))
    for method,beta in [('MA',0.)]+[(m,b) for b in betas for m in (('Greedy','A1-B') if exact else ('Greedy',))]:
        tick=time.process_time()
        try:
            if method=='MA' or sat:raw=dict(weights=mb.w.copy(),status='MA_saturated' if sat else 'MA_endpoint')
            elif method=='Greedy':raw=budget_greedy(f,inp.p,inp.epsilon,beta,geo)
            else:raw=solve_budgeted_dcl(f,inp.p,inp.epsilon,beta,geo,pc,30,inp.log_p)
            w=np.array(raw['weights']);lo,hi=f.regret_bounds(w);dl,du=mb.increment(w);bl,bu=mb.bounds(beta)
            residual=float(max(abs(w.sum()-1),-w.min(),w.max()-f.upper,0))
            reason='constraint_violation' if residual>1e-9 else 'proven_budget_violation' if dl>bu else None
            if reason:w=mb.w.copy();lo,hi=f.regret_bounds(w);dl,du=mb.increment(w)
            out.append(dict(method=method,beta=beta,weights=w,raw=raw,fallback=reason,mean_regret_lower=inp.p@lo,mean_regret_upper=inp.p@hi,success_lower=inp.p[hi<=inp.epsilon].sum(),success_upper=inp.p[lo<=inp.epsilon].sum(),mean_increment_lower=dl,mean_increment_upper=du,budget_lower=bl,budget_upper=bu,budget_certified=bool(du<=bl),budget_MA_identity=bool(np.array_equal(w,mb.w)),max_regret_upper=max(hi),primal_residual=residual,cpu=time.process_time()-tick))
        except Exception as e:out.append(dict(method=method,beta=beta,error=repr(e),cpu=time.process_time()-tick))
    return out
