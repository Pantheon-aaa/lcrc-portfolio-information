"""Gamma(all) ratio-one diagnostics, outside all performance comparisons."""
import numpy as np
from .common import OUT,append,rows,run
from .families import make
from .price_oracle import Geometry,PriceCache
from .mean_budget import MeanBudget
from .budgeted_dcl import solve_budgeted_dcl
from .experiments import specs,BETAS

def execute(budget):
    path=OUT/'S2_boundary_diagnostics.jsonl';done={r['key'] for r in rows(path)}
    for split in ('development','confirmation'):
        for spec in specs(split):
            f=make(*spec);p=np.ones(6)/6;g=Geometry(f,0).gamma(range(6),tight=True)
            for location,eps in [('lower',g.lower),('midpoint',(g.lower+g.upper)/2),('upper',g.upper)]:
                geo=Geometry(f,eps);pc=PriceCache(MeanBudget(f,p),geo)
                for beta in BETAS:
                    key=f'{f.name}-{location}-{beta}'
                    if key in done:continue
                    budget.check();a=solve_budgeted_dcl(f,p,eps,beta,geo,pc);state,z=geo.classify(range(6))
                    assert a['success_mass_lower']<=a['optimal_mass_upper']+1e-8
                    if state=='compatible':assert z.upper<=eps
                    if state=='incompatible':assert z.lower>eps
                    append(path,dict(key=key,family=f.name,split=split,kind=f.metadata['kind'],location=location,epsilon=eps,gamma_interval=[g.lower,g.upper],beta=beta,posterior=p,full_set_classification=state,performance_comparison=False,**a))
            print(f'Gamma boundary diagnostic {f.name}',flush=True)

if __name__=='__main__':run('S2_numerical_boundaries',execute)
