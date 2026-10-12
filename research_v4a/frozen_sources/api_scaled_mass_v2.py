"""Public API with a structural zero case kept separate from numerical underflow.

The frozen experiments call the general solver directly: none of their families
has identical objective matrices. This wrapper adds the degenerate API contract,
without changing a single experimental decision or a frozen algorithm file.
"""
import numpy as np
from time import perf_counter
from .budgeted_dcl import solve_budgeted_dcl as _solve,result
from .price_oracle import price_oracle,Geometry,PriceCache
from .mean_budget import MeanBudget

def shared_optimizer_certificate(family):
    # Identical objective functions are a sufficient algebraic certificate.
    # All other cases stay unclassified: rounded zero posterior weights are irrelevant.
    if all(np.array_equal(q,family.Q[0]) for q in family.Q):
        return dict(certified=True,reason='All public objective matrices are bitwise identical; every model has the same strictly convex program.',Lstar_exact=0.)
    return dict(certified=False,reason='No implemented structural certificate; numerical near-zero or underflow is insufficient.')

def solve_budgeted_dcl(family,posterior,epsilon,beta,geometry_cache=None,posterior_price_cache=None,time_limit_seconds=30.,log_posterior=None):
    cert=shared_optimizer_certificate(family)
    if not cert['certified'] or beta==0 or np.isinf(beta):
        out=_solve(family,posterior,epsilon,beta,geometry_cache,posterior_price_cache,time_limit_seconds)
        if not cert['certified'] and np.any(np.asarray(posterior)==0):
            from .posterior_precision import qualify
            out['beta']=beta;out=qualify(out,family,posterior,log_posterior);out.pop('beta',None)
        return out
    start=perf_counter();geo=geometry_cache or Geometry(family,epsilon)
    if geo.key!=(family.hash,float(epsilon)):raise ValueError('geometry cache signature mismatch')
    pc=posterior_price_cache or PriceCache(MeanBudget(family,posterior),geo);pc.validate(family,posterior,epsilon);mb=pc.mb
    lo,hi=family.regret_bounds(mb.w);lower=float(mb.p[hi<=epsilon].sum());upper=float(mb.p[lo<=epsilon].sum())
    out=result(mb,epsilon,beta,mb.w,lower,upper,mb.ma['gap'],'structural_zero_mean_numerical_open',start,geo,pc,structural_certificate=cert)
    out['posterior_mean_regret_MA_lower']=0.;out['posterior_mean_regret_MA_upper']=0.;out['budget_lower']=0.;out['budget_upper']=0.
    # The exact optimizer is known to be feasible; its stored float approximation
    # still has a nonzero primal/dual gap. Do not certify zero-budget float holdings.
    out['budget_certified']=out['mean_increment_upper']<=0
    out['optimizer_representation_l2_error_bound']=float(np.sqrt(2*mb.ma['gap']/np.linalg.eigvalsh(mb.Q)[0]))
    return out
