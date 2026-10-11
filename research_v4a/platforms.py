"""Finite-class platform by exhaustive subset pricing plus independent CVXPY checks."""
import json
from itertools import combinations
import numpy as np
from .common import OUT,ROOT,rows,append,write,run
from .families import make,equivalence,affinity
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .baselines import cvx_price
from .experiments import specs,BETAS

def execute(budget):
    cfg=json.loads((ROOT/'config_frozen.json').read_text());path=OUT/'S2_platforms.jsonl';done={r['key'] for r in rows(path)}
    for spec in specs('confirmation'):
        f=make(*spec);ga=Geometry(f,0).gamma(range(6));gamma=(ga.lower+ga.upper)/2;uf=float(np.median(f.costs(f.base)))
        epsilons=[('geometry',x,x*gamma) for x in cfg['geometry_ratios']]+[('common',x,x*uf) for x in cfg['common_epsilon_scales']]
        for design in ('Separated','Joint'):
            classes=equivalence(f,design)
            for scale,ratio,eps in epsilons:
                key=f'{f.name}-{design}-{scale}-{ratio}'
                if all(key+'-'+str(b) in done for b in BETAS):continue
                budget.check();values=[]
                for C in classes:
                    p=np.zeros(6);p[C]=1/len(C);mb=MeanBudget(f,p)
                    if len(C)==1:
                        values.append(dict(models=C,mass=len(C)/6,common_optimizer=True,values=[dict(beta=b,lower=1.,upper=1.,budget_margin=None) for b in BETAS]));continue
                    geo=Geometry(f,eps);pc=PriceCache(mb,geo);cross=[]
                    for size in range(7):
                        for G in combinations(range(6),size):
                            budget.check();r=pc.get(G,tight=True)
                            if G and r.epsilon_feasible:
                                ref=cvx_price(f,p,G,eps);error=max(r.cost_lower-ref['value'],ref['value']-r.cost_upper,0)
                                cross.append(dict(models=G,value=ref['value'],interval_error=error))
                                if error>2e-6*max(mb.Lhi,1e-20):raise RuntimeError('platform independent price mismatch')
                    vals=[]
                    for beta in BETAS:
                        lo,hi=f.regret_bounds(mb.w);base=float(p[hi<=eps].sum());bl,bu=mb.bounds(beta)
                        lower=max([base]+[float(p[list(G)].sum()) for G,r in pc.cache.items() if r.epsilon_feasible and r.cost_upper<=bl])
                        upper=max([lower]+[float(p[list(G)].sum()) for G,r in pc.cache.items() if np.isfinite(r.cost_lower) and r.cost_lower<=bu])
                        intervals=[(r.cost_lower,r.cost_upper) for r in pc.cache.values() if np.isfinite(r.cost_upper)]
                        margin=min([max(l-bu,bl-u,0) for l,u in intervals],default=0) if np.isfinite(beta) else None
                        vals.append(dict(beta=beta,lower=lower,upper=upper,budget_margin=margin))
                    values.append(dict(models=C,mass=len(C)/6,common_optimizer=False,Lstar=[mb.Llo,mb.Lhi],values=vals,cvx_checks=cross))
                for bi,beta in enumerate(BETAS):
                    append(path,dict(key=key+'-'+str(beta),family=f.name,kind=f.metadata['kind'],d=f.d,M=6,mask_design=design,epsilon_scale=scale,epsilon_ratio=ratio,epsilon=eps,beta=beta,
                      platform_lower=sum(x['mass']*x['values'][bi]['lower'] for x in values),platform_upper=sum(x['mass']*x['values'][bi]['upper'] for x in values),classes=values))
        print(f'Independent platforms {f.name} complete',flush=True)

if __name__=='__main__':run('platforms',execute)
