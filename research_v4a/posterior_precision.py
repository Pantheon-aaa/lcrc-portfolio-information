"""Zero floating posterior weights require an explicit precision qualification."""
import numpy as np
import json
import mpmath as mp
from .common import OUT,ROOT,rows,write,run

def posterior_error(p,log_posterior):
    with mp.workdps(80):
        ll=[mp.mpf(float(x)) for x in log_posterior];mx=max(ll);v=[mp.exp(x-mx) for x in ll];den=sum(v);v=[x/den for x in v]
        q=np.asarray(p,float);q=q/q.sum();tv=sum(abs(x-mp.mpf(float(y))) for x,y in zip(v,q))+mp.mpf('1e-70')
        # Upward float conversion; the 80-digit arithmetic allowance is explicit.
        upper=np.nextafter(float(tv),np.inf)
        return dict(TV_upper=float(upper),TV_upper_decimal=str(tv),zero_coordinates=np.flatnonzero(q==0).tolist(),reference='80-digit normalization of log posterior weights; conservative L1 bound also covers normalization; includes 1e-70 arithmetic allowance')

def qualify(out,f,p,log_posterior=None):
    if not np.any(np.asarray(p)==0):return out
    Rmax=float(max(.5*np.linalg.eigvalsh(q)[-1]*f.upper-v for q,v in zip(f.Q,f.VL)))
    precision=posterior_error(p,log_posterior) if log_posterior is not None else dict(TV_upper=1.,reference='Input contains zeros without log-posterior evidence; source precision unknown.')
    return qualify_bound(out,Rmax,precision)

def qualify_bound(out,Rmax,precision):
    out=dict(out)
    delta=precision['TV_upper'];oldLlo=out['posterior_mean_regret_MA_lower'];oldLhi=out['posterior_mean_regret_MA_upper'];beta=out.get('beta')
    for prefix in ('success_mass','optimal_mass'):
        out['nominal_'+prefix+'_lower']=out[prefix+'_lower'];out['nominal_'+prefix+'_upper']=out[prefix+'_upper']
        out[prefix+'_lower']=max(0.,float(np.nextafter(out[prefix+'_lower']-delta,-np.inf)))
        out[prefix+'_upper']=min(1.,float(np.nextafter(out[prefix+'_upper']+delta,np.inf)))
    out['posterior_mean_regret_MA_lower']=max(0.,oldLlo-Rmax*delta);out['posterior_mean_regret_MA_upper']=oldLhi+Rmax*delta
    out['posterior_mean_regret_method_lower']=max(0.,out['posterior_mean_regret_method_lower']-Rmax*delta);out['posterior_mean_regret_method_upper']+=Rmax*delta
    out['mean_increment_lower']=max(0.,out['mean_increment_lower']-2*Rmax*delta);out['mean_increment_upper']+=2*Rmax*delta
    out['nominal_secondary_price_gap']=out['secondary_price_gap'];out['secondary_price_gap']=np.inf
    out['nominal_status']=out['status'];out['status']='posterior_precision_open';out['posterior_precision']=precision
    # A tiny posterior perturbation can change budget-feasible subsets at a jump.
    # The no-budget objective is TV-Lipschitz; finite-budget optimum needs a margin.
    if beta is not None and np.isfinite(float(beta)):
        out['budget_lower']=float(beta)*out['posterior_mean_regret_MA_lower'];out['budget_upper']=float(beta)*out['posterior_mean_regret_MA_upper']
        out['budget_certified']=out['mean_increment_upper']<=out['budget_lower']
        out['optimal_mass_upper']=1.
        if not out['budget_certified']:out['optimal_mass_lower']=0.
    out['primary_mass_gap']=max(0.,out['optimal_mass_upper']-out['optimal_mass_lower'])
    return out

def execute(budget):
    records=[];risk_bounds={}
    for split in ('development','confirmation'):
        for r in rows(OUT/f'S2_{split}_evidence.jsonl'):
            if np.any(np.array(r['posterior'])==0):
                budget.check()
                if r['family'] not in risk_bounds:
                    family=json.loads((OUT/'families'/f'{r["family"]}.json').read_text());Q=np.array(family['Q']);VL=np.array([x['value_lb'] for x in family['oracle_diagnostics']]);risk_bounds[r['family']]=float(np.max(.5*np.linalg.eigvalsh(Q)[:,-1]*family['upper']-VL))
                records.append(dict(dataset=r['dataset'],split=split,family=r['family'],n_blocks=r['n_blocks'],mask_design=r['mask_design'],Rmax=risk_bounds[r['family']],**posterior_error(r['posterior'],r['log_observation_density'])))
    write(OUT/'posterior_precision.json',dict(datasets=records,scope='Nominal frozen decision records are preserved. Zeros from finite Gaussian likelihoods are underflow, not structural zeros; their global secondary certificates remain open. Actual evaluated holdings and success observations are unchanged.'))
    print(f'Posterior underflow qualifications: {len(records)} datasets',flush=True)

if __name__=='__main__':run('posterior_precision',execute)
