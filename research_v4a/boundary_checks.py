"""S1 full small enumeration and every reachable price jump, including open boundaries."""
import itertools,time
import numpy as np
import mpmath as mp
from .common import OUT,write,run
from .families import analytic,analytic_price
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .baselines import cvx_price

def execute(budget):
    jump_records=[];enumeration=[];errors=[];timings=[];mp.mp.dps=80
    for d in (3,6,10,20):
        for fraction in (.25,.75):
            f,L0=analytic(d,fraction);p=np.ones(d)/d;mb=MeanBudget(f,p);t=f.metadata['t'];c=mp.sqrt(mp.mpf(d)/(d-1));tm=mp.mpf(t);Lm=tm*tm/(2*(1-2*c*tm))
            for ratio in (.5,.75,.875,.95,1.05):
                budget.check();cpu_start=time.process_time();wall_start=time.perf_counter();eps=L0*ratio;pc=PriceCache(mb,Geometry(f,eps));formula=analytic_price(d,t,eps)
                prices={m:pc.get(range(m),tight=True) for m in range(d+1)}
                timings.append(dict(d=d,fraction=fraction,ratio=ratio,cardinality_prices=d+1,cpu_seconds=time.process_time()-cpu_start,wall_seconds=time.perf_counter()-wall_start,geometry_calls=pc.geometry.calls,price_calls=pc.calls))
                for a in formula:
                    if not np.isfinite(a['cost']) or a['cost']<=0:continue
                    m=a['m'];r=prices[m];eta=mp.mpf(d-m)/(m*(d-1));am=1+2*c*tm/(d-2)*(2-mp.mpf(d)/m);disc=tm*tm*eta-2*am*(Lm-mp.mpf(eps))
                    if disc<0:continue
                    radius=2*(Lm-mp.mpf(eps))/(tm*mp.sqrt(eta)+mp.sqrt(disc));exact=float(radius*radius/2);beta=a['beta']
                    # The analytic Q is rounded when assembled. Use its explicitly measured
                    # floating-matrix discrepancy when comparing exact-real and float inputs.
                    normalized_error=max(r.cost_lower-exact,exact-r.cost_upper,0)/L0
                    if normalized_error>1e-6:errors.append(dict(d=d,m=m,ratio=ratio,error='high precision price',value=normalized_error))
                    for side,mult in [('below',1-1e-5),('on',1),('above',1+1e-5)]:
                        b=beta*mult;bl,bu=mb.bounds(b)
                        lower=max([0.]+[k/d for k,z in prices.items() if z.epsilon_feasible and z.cost_upper<=bl]);upper=max([lower]+[k/d for k,z in prices.items() if np.isfinite(z.cost_lower) and z.cost_lower<=bu])
                        attainable=[x['m']/d for x in formula if x['cost']<=b*L0];expected=max([0.]+attainable)
                        # An exact-real jump may shift by <= numerical price uncertainty in
                        # the stored float covariance. On-jump points are reported open.
                        compatible=lower-1e-8<=expected<=upper+1e-8
                        if side!='on' and not compatible:errors.append(dict(d=d,m=m,ratio=ratio,side=side,error='jump interval',bounds=[lower,upper],expected=expected))
                        jump_records.append(dict(d=d,fraction=fraction,ratio=ratio,m=m,side=side,beta=b,price_interval=[r.cost_lower,r.cost_upper],high_precision_exact_input_cost=str(radius*radius/2),normalized_price_error=normalized_error,mass_lower=lower,mass_upper=upper,analytic_mass=expected,analytic_float_input_compatible=compatible,status='open_exact_input_boundary' if side=='on' and not compatible else 'certified_interval',deterministic_worst_model_success=1. if lower==1 else 0.,uniform_random_group_worst_success=lower))
                if d<=6:
                    refs=[];start=time.perf_counter()
                    for size in range(d+1):
                        for G in itertools.combinations(range(d),size):
                            budget.check();z=pc.get(G,tight=True)
                            if G and z.epsilon_feasible:
                                ref=cvx_price(f,p,G,eps);err=max(z.cost_lower-ref['value'],ref['value']-z.cost_upper,0)/L0
                                if err>1e-6:errors.append(dict(d=d,models=G,error='independent full enumeration',value=err))
                                refs.append(dict(models=G,price_lower=z.cost_lower,price_upper=z.cost_upper,cvx_cost=ref['value'],w=z.witness_weights))
                    for beta in (0,.01,.025,.05,.1,.25,np.inf):
                        bl,bu=mb.bounds(beta);base=float(p[f.regret_bounds(mb.w)[1]<=eps].sum())
                        lower=max([base]+[len(r['models'])/d for r in refs if r['price_upper']<=bl]);upper=max([lower]+[len(G)/d for G,z in pc.cache.items() if np.isfinite(z.cost_lower) and z.cost_lower<=bu]);minprice=min([0.] if lower==base else [r['price_upper'] for r in refs if len(r['models'])/d>=lower and r['price_upper']<=bl])
                        enumeration.append(dict(d=d,fraction=fraction,ratio=ratio,beta=beta,subsets=2**d,lower=lower,upper=upper,secondary_upper=minprice,shared_enumeration_wall_seconds=time.perf_counter()-start,prices=refs))
    write(OUT/'S1_boundaries.json',dict(passed=not errors,errors=errors,jumps=jump_records,small_full_enumeration=enumeration,symmetry_precomputation_timings=timings))
    print(f'S1 boundary checks: {len(jump_records)} points, {len(enumeration)} exhaustive queries; errors={len(errors)}',flush=True)
    if errors:raise RuntimeError('S1 boundary contradiction')

if __name__=='__main__':run('S1_boundaries',execute)
