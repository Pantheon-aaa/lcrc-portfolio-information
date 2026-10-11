"""Tighten conservative reference bounds; preserve every original record/holding."""
import json,hashlib
import numpy as np
from .common import ROOT,OUT,rows,write,run
from .families import analytic
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache

def execute(budget):
    corrections=[];bycase={}
    for r in rows(OUT/'S1.jsonl'):
        if r['method']!='A1-B-symmetry' or r['true_model']!=0:continue
        d=r['d'];fraction=float(r['family'].rsplit('-',1)[1]);key=(d,fraction,r['epsilon'])
        if key not in bycase:
            budget.check();f,L=analytic(d,fraction);mb=MeanBudget(f,np.ones(d)/d);pc=PriceCache(mb,Geometry(f,r['epsilon']));prices=[pc.get(range(m),tight=True) for m in range(d+1)];bycase[key]=(mb,prices)
        mb,prices=bycase[key];beta=float(r['beta']);bu=mb.bounds(beta)[1];lower=max(r['optimal_mass_lower'],r['success_mass_lower']);upper=max([lower]+[m/d for m,z in enumerate(prices) if np.isfinite(z.cost_lower) and z.cost_lower<=bu]);exact=r['analytic_mass']
        if not lower-1e-8<=exact<=upper+1e-8:raise RuntimeError('refined S1 interval excludes independent formula')
        corrections.append(dict(query=r['query'],old_lower=r['optimal_mass_lower'],old_upper=r['optimal_mass_upper'],lower=lower,upper=upper,primary_mass_gap=max(0,upper-lower),holdings_changed=False))
    write(OUT/'S1_refined_certificates.json',corrections)
    audit=json.loads((OUT/'S3_independent_enumeration.json').read_text());casechecks={f'{x["family"]}-{x["ratio"]}':x for x in audit['cases']};refined=[]
    for r in rows(OUT/'S3_computation.jsonl'):
        if r['method']!='Enumeration' or 'weights' not in r:continue
        case=casechecks[r['case']];p=np.array(case['posterior']);bu=float(r['budget_upper']);lower=r['optimal_mass_lower'];upper=max([lower]+[float(p[x['models']].sum()) for x in case['checks'] if 'lower' in x and np.isfinite(float(x['lower'])) and float(x['lower'])<=bu])
        refined.append(dict(case=r['case'],beta=r['beta'],old_lower=lower,old_upper=r['optimal_mass_upper'],lower=lower,upper=upper,primary_mass_gap=max(0,upper-lower),holdings_changed=False))
    write(OUT/'S3_refined_certificates.json',refined)
    old=ROOT/'frozen_sources'/'budgeted_dcl.py';new=ROOT/'budgeted_dcl.py'
    write(OUT/'reference_correction_provenance.json',dict(original_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),delivery_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),scope='This reference refinement excludes proven infinite/infeasible prices under beta=inf and promotes actual S1 holding mass to its lower bound. It changes no holding. Separately, scaled-mass-v2 repairs MILP numerical certificates and qualifies global secondary precision; all S2 A1-B/legacy A1 decisions and the full S3 benchmark are recomputed on the same fixed data. See config_frozen.json numerical_revision and the validated manifests. Initial streams are retained.',original_sources='research_v4a/frozen_sources/budgeted_dcl.py'))
    print(f'Refined {len(corrections)} S1 and {len(refined)} enumeration certificates',flush=True)

if __name__=='__main__':run('reference_certificate_refinement',execute)
