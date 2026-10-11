"""Strict feasibility qualifications; never modify a holding or its success event."""
import numpy as np


def qualify(out,beta=None):
    if 'weights' not in out:return out
    beta=float(out.get('beta',beta))
    if not np.isfinite(beta):return out
    strict=float(out['mean_increment_upper'])<=float(out['budget_lower'])
    if strict:return out
    out=dict(out)
    out['nominal_budget_certified']=out.get('nominal_budget_certified',out['budget_certified'])
    out['budget_certified']=False
    out['nominal_optimal_mass_lower']=out.get('nominal_optimal_mass_lower',out['optimal_mass_lower'])
    out['optimal_mass_lower']=0.
    if beta==0:
        out['ma_endpoint_by_definition']=True
        out['nominal_optimal_mass_upper']=out.get('nominal_optimal_mass_upper',out['optimal_mass_upper'])
        out['optimal_mass_upper']=1.
        out['status']='MA_endpoint_numerical_open'
    elif out['status'] not in ('posterior_precision_open','budget_precision_limited'):
        out['nominal_status']=out.get('nominal_status',out['status']);out['status']='budget_precision_limited'
    out['primary_mass_gap']=max(0.,out['optimal_mass_upper']-out['optimal_mass_lower'])
    out['nominal_secondary_price_gap']=out.get('nominal_secondary_price_gap',out['secondary_price_gap'])
    out['secondary_price_gap']=np.inf
    out['strict_budget_qualification']='No certified floating feasible incumbent for this budget; nominal bounds retained separately. The MA endpoint policy is unchanged.'
    return out
