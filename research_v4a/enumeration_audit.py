"""Independent CVXPY modeling of every small S3 subset; not a new method."""
from itertools import combinations
import time
import numpy as np
from .common import OUT,seed,rows,write,run
from .families import make,observations
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .baselines import cvx_price
from .price_precision import MPPrices

def execute(budget):
    records=[];errors=[]
    for d in (10,20):
        for kind in ('asymmetric','hidden_curvature','block'):
            f=make(kind,d,6,'computation',0);_,ll=observations(f,0,32,'Joint',seed('S3-observations',f.name));p=f.posterior(ll);g=Geometry(f,0).gamma(range(6));gamma=(g.lower+g.upper)/2
            for ratio in (.5,.75):
                mb=MeanBudget(f,p);eps=ratio*gamma;pc=PriceCache(mb,Geometry(f,eps));checks=[];start=time.process_time();high_precision=None
                for size in range(1,7):
                    for G in combinations(range(6),size):
                        budget.check();z=pc.get(G,tight=True)
                        try:ref=cvx_price(f,p,G,eps)
                        except Exception as ex:checks.append(dict(models=G,lower=z.cost_lower,upper=z.cost_upper,status='independent_solver_failure',error=repr(ex)));continue
                        precision=None
                        if np.isfinite(ref['value']):
                            outside=max(z.cost_lower-ref['value'],ref['value']-z.cost_upper,0);normalized=outside/max(mb.Lhi,1e-30)
                            if normalized>1e-6 and np.isfinite(z.cost_upper):
                                if high_precision is None:high_precision=MPPrices(f,p)
                                precision=high_precision.check(G,eps,z,pc.geometry.gamma(G).w)
                                if precision['certificate_consistent'] is False:errors.append(dict(family=f.name,ratio=ratio,models=G,error='high precision certificate contradiction',normalized_error=normalized,precision=precision))
                            elif normalized>1e-6:errors.append(dict(family=f.name,ratio=ratio,models=G,error='independent finite optimum vs incompatible proof',normalized_error=normalized))
                        else:
                            normalized=None
                            if np.isfinite(z.cost_upper):errors.append(dict(family=f.name,ratio=ratio,models=G,error='finite certificate vs independent infeasible'))
                        checks.append(dict(models=G,lower=z.cost_lower,upper=z.cost_upper,cvx_price=ref['value'],cvx_status=ref['status'],normalized_error=normalized,high_precision=precision))
                records.append(dict(family=f.name,ratio=ratio,epsilon=eps,posterior=p,checks=checks,independent_audit_cpu_seconds=time.process_time()-start))
                print(f'Independent S3 enumeration audit {f.name} ratio={ratio}',flush=True)
    write(OUT/'S3_independent_enumeration.json',dict(passed=not errors,errors=errors,cases=records,scope='All 63 nonempty subsets for all twelve M=6 S3 cases. Native enumeration timing and CVXPY audit timing remain separate.'))
    if errors:raise RuntimeError('independent S3 enumeration mismatch')

if __name__=='__main__':
    used=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl') if r['stage'].startswith('S3_'))
    run('S3_independent_enumeration',execute,allocation=max(1,3600-used))
