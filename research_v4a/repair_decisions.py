"""Re-evaluate both MILP methods on every frozen dataset, preserve other baselines.

This is a numerical-correctness repair, never a selection of favorable datasets.
Original logs remain unchanged. The output is a complete replacement stream in
the original data/threshold/method order, with a separate completion manifest.
"""
import json,itertools,time
import numpy as np
from .common import ROOT,OUT,rows,append,write,run
from .families import make
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .budgeted_dcl import solve_budgeted_dcl
from .baselines import original_a1
from .posterior_precision import posterior_error,qualify_bound
from .experiments import BETAS,record

KEYS=['dataset','experiment','split','family','family_seed','data_seed','family_hash','kind','d','M','mask_design','n_blocks','observed_scalar_count','rep','prior_name','risk_unit','epsilon','epsilon_scale','epsilon_ratio']

def pending_rows(path,eligible):
    # The controlled writer places dataset first. Decode only that JSON string
    # for completed groups; do not repeatedly parse every portfolio and diagnostic.
    # Archive-only replay retains the generic reader and identical record order.
    if not path.exists():
        for r in rows(path):
            if r['dataset'] in eligible:yield r
        return
    decoder=json.JSONDecoder();prefix='{"dataset":'
    with path.open('r',encoding='utf-8') as source:
        for line in source:
            if line.startswith(prefix):
                dataset,_=decoder.raw_decode(line,len(prefix))
                if dataset not in eligible:continue
            r=json.loads(line)
            if r['dataset'] in eligible:yield r

def execute(budget,split):
    cfg=json.loads((ROOT/'config_frozen.json').read_text());source_done={r['dataset'] for r in rows(OUT/f'S2_{split}_completed.jsonl')};completed=OUT/f'S2_{split}_validated_completed.jsonl';done={r['dataset'] for r in rows(completed)};path=OUT/f'S2_{split}_validated.jsonl'
    evidence={r['dataset']:r for r in rows(OUT/f'S2_{split}_evidence.jsonl')};families={};geometry={};expected=444 if split=='development' else 4344
    for dataset,stream in itertools.groupby(pending_rows(OUT/f'S2_{split}.jsonl',source_done-done),key=lambda r:r['dataset']):
        original=list(stream)
        if dataset not in source_done or dataset in done:continue
        budget.check();meta=original[0];name=meta['family']
        if name not in families:
            f=make(meta['kind'],meta['d'],6,split,int(name.rsplit('-',1)[1]));assert f.hash==meta['family_hash'];families[name]=f
        f=families[name];e=evidence[dataset];p=np.array(e['posterior']);posterior_guard=None
        if np.any(p==0):
            posterior_guard=posterior_error(p,e['log_observation_density']);posterior_guard['Rmax']=float(np.max(.5*np.linalg.eigvalsh(f.Q)[:,-1]*f.upper-f.VL))
        replacements={}
        for scale,ratio,eps in dict.fromkeys((r['epsilon_scale'],r['epsilon_ratio'],r['epsilon']) for r in original):
            geo=geometry.setdefault((name,eps),Geometry(f,eps));pc=PriceCache(MeanBudget(f,p),geo)
            for method,beta in [('A1-v3',np.inf)]+[('A1-B',b) for b in BETAS]:
                budget.check()
                try:
                    a=original_a1(f,p,eps) if method=='A1-v3' else solve_budgeted_dcl(f,p,eps,beta,geo,pc)
                    if posterior_guard is not None:
                        a['beta']=beta;a=qualify_bound(a,posterior_guard['Rmax'],posterior_guard);a.pop('beta',None)
                    a['numerical_revision']=cfg['numerical_revision']['id'];replacements[(scale,ratio,method,str(float(beta)))]=a
                except Exception as ex:replacements[(scale,ratio,method,str(float(beta)))]=dict(status='failed',failure_reason=repr(ex),numerical_revision=cfg['numerical_revision']['id'])
            append(OUT/f'S2_{split}_validated_certificates.jsonl',dict(dataset=dataset,epsilon=eps,epsilon_scale=scale,posterior=p,numerical_revision=cfg['numerical_revision']['id'],geometry_cuts=geo.cuts,
                prices=[dict(models=G,lower=r.cost_lower,upper=r.cost_upper,status=r.status,witness_weights=r.witness_weights,epsilon_feasible=r.epsilon_feasible,primal_residual=r.primal_residual,dual_residual=r.dual_residual,diagnostics=r.diagnostics) for G,r in pc.cache.items()]))
        for r in original:
            key=(r['epsilon_scale'],r['epsilon_ratio'],r['method'],str(float(r['beta'])))
            if key not in replacements:append(path,r);continue
            a=replacements[key];base={k:r[k] for k in KEYS}
            if 'weights' in a:record(path,base,r['method'],r['beta'],f,p,r['epsilon'],a,[r['true_model']])
            else:append(path,{**base,'method':r['method'],'beta':r['beta'],'true_model':r['true_model'],**a})
        append(completed,dict(dataset=dataset));done.add(dataset)
        if len(done)%24==0:print(f'Validated {split}: {len(done)}/{expected}; CPU {time.process_time():.1f}s',flush=True)
    append(OUT/f'S2_{split}_validated_geometry.jsonl',dict(completed_datasets=len(done),families=[dict(family=name,epsilon=eps,cuts=geo.cuts,certificates=[dict(models=G,lower=r.lower,upper=r.upper,weights=r.w,status=r.status) for G,r in geo.cache.items()]) for (name,eps),geo in geometry.items()]))
    write(OUT/f'S2_{split}_validated_manifest.json',dict(complete=len(done)==expected,datasets=len(done),expected_datasets=expected,numerical_revision=cfg['numerical_revision']['id'],unchanged_observations_and_design=True,recomputed_methods=['A1-B','A1-v3']))

if __name__=='__main__':
    import sys
    split=sys.argv[1];run('repair_'+split,lambda b:execute(b,split))
