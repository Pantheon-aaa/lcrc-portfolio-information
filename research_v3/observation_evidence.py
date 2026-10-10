"""Regenerate and persist the frozen A-family likelihood evidence; no refitting."""
import json
import numpy as np
from .common import OUT,CFG,rows,append,write,seed,timed_stage
from .compat_models import e1,geometry,draw_loglik

def run(budget):
    cfg=json.loads((CFG/'protocol.json').read_text());path=OUT/'observation_evidence.jsonl'
    done={r['dataset'] for r in rows(path)};families={};counts={};max_error=0.
    for filename in ('E1.jsonl','E2_development.jsonl','E2_confirmation.jsonl'):
        count=0
        for r in rows(OUT/filename):
            if r['method']!='MA' or r['epsilon_ratio'] not in ('fixed',.25):continue
            count+=1
            if r['dataset'] in done:continue
            budget.check();name=r['family']
            if name not in families:
                families[name]=e1(r['t']) if r['experiment']=='E1' else geometry(r['kind'],r['d'],r['M'],r['split'])
            f=families[name];rng_seed=seed(cfg['seed_namespace'],r['dataset'])
            ll=draw_loglik(f,r['truth'],r['joint_rows'],r['singleton_per_asset'],np.random.default_rng(rng_seed))
            p=f.posterior(ll);hi=f.regret_bounds(np.array(r['weights']))[1]
            error=abs(float(p@hi)-r['posterior_mean_upper']);max_error=max(max_error,error)
            assert error<1e-9,'regenerated evidence disagrees with saved objective'
            append(path,dict(dataset=r['dataset'],family=name,experiment=r['experiment'],split=r['split'],
              truth_for_evaluator_only=r['truth'],joint_rows=r['joint_rows'],singleton_per_asset=r['singleton_per_asset'],
              rng_seed=rng_seed,prior=np.ones(f.M)/f.M,log_likelihood=ll,posterior=p,
              saved_MA_objective_reconstruction_error=error))
            done.add(r['dataset'])
        counts[filename]=count
    write(OUT/'observation_evidence_manifest.json',{'passed':True,'datasets':len(done),'counts':counts,
      'max_MA_objective_error_this_run':max_error,
      'provenance':'Exact regeneration from the original frozen seeds and unchanged likelihood/generator modules. No experimental decision was recomputed or replaced.'})
    print(f'Frozen observation evidence verified: {len(done)} datasets; max objective error {max_error:.3g}',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
