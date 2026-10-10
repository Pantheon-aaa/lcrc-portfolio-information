"""Read-only output audit plus separately reproducible numerical reference checks."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,CFG,rows,write,timed_stage,seed
from .compat_models import geometry,Family,simulate,draw_loglik,kl_matrix
from .compatible_qcqp import Compatibility,solve,cvx_reference
from .dcl_master import DCL,enumerate_exact

def run(budget):
    config=json.loads((CFG/'protocol.json').read_text(encoding='utf-8'));checks={};counts={};errors=[]
    expected={'E1.jsonl':3*4*2*config['e1_repetitions']*9,
      'E2_development.jsonl':16*4*2*config['e2_repetitions']*4*8,
      'E2_confirmation.jsonl':16*4*2*config['confirmation_repetitions']*4*8,
      'E3.jsonl':2*3*7*2*config['e3_repetitions']*8,
      'E4_development.jsonl':2*4*config['e4_repetitions']*13,
      'E4_confirmation.jsonl':2*4*config['confirmation_repetitions']*13,
      'E4_development_safe_reference.jsonl':2*4*config['e4_repetitions'],
      'E4_confirmation_safe_reference.jsonl':2*4*config['confirmation_repetitions']}
    for filename,target in expected.items():
        budget.check();seen=set();n=0;failures=0;max_resid=0;gaps=[];uncertain=0;open_gap=0
        for r in rows(OUT/filename):
            n+=1;key=(r['dataset'],r['method'],str(r.get('epsilon_ratio','')))
            if key in seen:errors.append(f'duplicate {filename} {key}')
            seen.add(key)
            if 'error' in r:failures+=1;continue
            if filename.startswith('E4_') and r['epsilon_ratio']!=1.:
                errors.append(f'E4 epsilon label {key}')
            if 'weights' in r:
                w=np.array(r['weights']);max_resid=max(max_resid,abs(w.sum()-1),float(-w.min()))
                if max_resid>1e-8:errors.append(f'infeasible {filename} {key}')
            if 'posterior_mass_lower' in r:
                if r['posterior_mass_lower']>r['posterior_mass_upper']+1e-9:errors.append(f'mass ordering {key}')
            if r.get('optimization_gap') is not None:
                gap=r['optimization_gap'];gaps.append(gap)
                if gap< -1e-8:errors.append(f'negative gap {key}')
                if gap>1e-6:open_gap+=1
            uncertain+=bool(r.get('success_indeterminate',False))
        counts[filename]={'rows':n,'expected':target,'complete':n==target,'failures':failures,'success_indeterminate':uncertain,
          'max_feasibility_residual':max_resid,'max_recorded_gap':max(gaps) if gaps else None,'gaps_above_1e-6':open_gap}
    # Out-of-run independent exhaustive reference, beyond the three-model gate.
    refs=[]
    for kind in ('asymmetric','hidden_curvature'):
        f=geometry(kind,6,6,'verification');c=Compatibility(f);g=c.gamma(range(6));gamma=.5*(g.lower+g.upper)
        for ratio in (.25,.5,.75,1.):
            epsilon=ratio*gamma
            for case in range(3):
                p=np.random.default_rng(seed('reference-p',kind,case)).dirichlet(np.ones(6))
                a=DCL(c,epsilon).solve(p);ref,upper=enumerate_exact(c,p,epsilon)
                if ratio<1:
                    valid=abs(a['mass_lower']-ref[0])<=1e-8 and upper-ref[0]<=1e-8
                else:
                    # At Gamma(all), unresolved sets are not certified conflicts.
                    valid=max(a['mass_lower'],ref[0])<=min(a['mass_upper'],upper)+1e-8
                if not valid:errors.append(f'enumeration mismatch {kind}-{ratio}-{case}')
                refs.append({'kind':kind,'ratio':ratio,'case':case,'status':a['status'],
                  'A1_lower':a['mass_lower'],'A1_upper':a['mass_upper'],'enumeration_lower':ref[0],
                  'enumeration_upper':upper,'intervals_consistent':valid})
        for ids,r in c.cache.items():
            reference,_=cvx_reference(f,ids)
            if not r.lower-1e-9<=reference<=r.upper+1e-9:
                errors.append(f'independent QCQP mismatch {kind}-{ids}')
    checks['six_model_enumeration']=refs
    observation=json.loads((OUT/'observation_evidence_manifest.json').read_text())
    checks['regenerated_observation_evidence']=observation
    if not observation['passed'] or observation['datasets']!=3*4*2*config['e1_repetitions']+16*4*2*(config['e2_repetitions']+config['confirmation_repetitions']):
        errors.append('observation evidence incomplete')
    # Check likelihood sufficient-statistic algebra on an actual mask dataset.
    f=geometry('asymmetric',6,6,'verification');Y=simulate(f,2,25,32,np.random.default_rng(seed('raw-ll-audit')))
    raw=f.likelihood(Y);scatter=Y[-25:].T@Y[-25:]
    fast=-.5*(25*f.logdet+np.einsum('mij,ji->m',f.inverse,scatter)+25*6*np.log(2*np.pi))
    for j in range(6):
        y=Y[j*32:(j+1)*32,j];diag=f.Q[:,j,j]
        fast-=.5*(32*np.log(2*np.pi*diag)+np.sum(y*y)/diag)
    llerr=float(np.max(abs(raw-fast)));checks['raw_scatter_likelihood_error']=llerr
    if llerr>1e-8:errors.append('raw scatter likelihood mismatch')
    ledger=list(rows(OUT/'runtime_ledger.jsonl'));cpu=sum(r['cpu_seconds'] for r in ledger)+time.process_time()-budget.cpu
    checks['cpu_seconds_before_finalization']=cpu
    if cpu>21600:errors.append('CPU budget exceeded')
    checks['stage_cpu']={stage:sum(r['cpu_seconds'] for r in ledger if r['stage']==stage) for stage in set(r['stage'] for r in ledger)}
    checks['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    frozen=json.loads((ROOT/'environment.json').read_text())['algorithm_sha256'];provenance={}
    for filename,expected_hash in frozen.items():
        raw=(ROOT/filename).read_bytes()
        # Reconstruct the sole post-run runner edit: correcting E4's metadata label.
        if filename=='experiments.py':raw=raw.replace(b'epsilon_ratio=1.)',b'epsilon_ratio=.5)')
        provenance[filename]=hashlib.sha256(raw).hexdigest()==expected_hash
        if not provenance[filename]:errors.append(f'frozen algorithm changed: {filename}')
    checks['frozen_algorithm_consistency_excluding_documented_metadata_fix']=provenance
    passed=not errors and all(r['complete'] for r in counts.values())
    result={'passed':passed,'counts':counts,'checks':checks,'errors':errors}
    write(OUT/'delivery_verification.json',result)
    print(json.dumps({'passed':passed,'counts':counts,'errors':errors},ensure_ascii=False,indent=2),flush=True)
    if errors:raise RuntimeError('delivery integrity errors')

if __name__=='__main__':timed_stage('confirmation',3600,run)
