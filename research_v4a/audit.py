"""Independent record audit, reproduction spot checks, and explicit missing grid."""
import json,hashlib,collections,itertools,ast
import numpy as np
from .common import ROOT,OUT,rows,write,append,run,decision_path
from .experiments import specs,BETAS
from .families import make,observations,masks
from .posterior_precision import qualify_bound
from .certificate_qualification import qualify as strict_budget

def audit(budget):
    cfg=json.loads((ROOT/'config_frozen.json').read_text());errors=[];flags=collections.Counter();counts={};unbudgeted_pairs={};failure_records=OUT/'record_failures.jsonl'
    precision={r['dataset']:r for r in json.loads((OUT/'posterior_precision.json').read_text())['datasets']} if (OUT/'posterior_precision.json').exists() else {}
    failure_records.write_text('',encoding='utf-8')
    for name,expected in cfg.get('validated_algorithm_sha256',cfg['frozen_algorithm_sha256']).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            original=ROOT/'frozen_sources'/name
            if name=='budgeted_dcl.py' and original.exists() and hashlib.sha256(original.read_bytes()).hexdigest()==expected:
                old=original.read_text(encoding='utf-8');new=(ROOT/name).read_text(encoding='utf-8')
                allowed=old.replace('for G,r in prices.cache.items() if r.cost_lower<=bu])','for G,r in prices.cache.items() if np.isfinite(r.cost_lower) and r.cost_lower<=bu])')
                if allowed==new:flags['reference_only_infinite_bound_correction_no_decision_change']+=1
                else:errors.append(dict(error='unapproved frozen source difference',file=name))
            else:errors.append(dict(error='frozen algorithm changed',file=name))
    for stage in ('S1','S2_development','S2_confirmation','S3_computation'):
        n=0;status=collections.Counter();maxres=0.;maxgap=0.;uncertain=0;uncertified=0
        grid={};grid_keys=set();grid_ds=None;grid_rows=0;grid_expected=0;malformed=[];method_rows=collections.Counter()
        def finish_group():
            if grid_ds is None:return
            if grid_ds in grid or grid_rows!=grid_expected or len(grid_keys)!=grid_expected:malformed.append(grid_ds)
            grid[grid_ds]=grid_rows
        source=decision_path(stage.removeprefix('S2_')) if stage.startswith('S2_') else OUT/f'{stage}.jsonl'
        for r in rows(source):
            n+=1
            if n%10000==0:budget.check()
            if stage.startswith('S2_'):
                if r['dataset']!=grid_ds:
                    finish_group();grid_ds=r['dataset'];grid_rows=0;grid_keys=set();grid_expected=1344 if r['n_blocks']==0 else 224
                grid_rows+=1;grid_keys.add((r['epsilon_scale'],r['epsilon_ratio'],r['method'],str(float(r['beta'])),r['true_model']));method_rows[r['method']]+=1
            if r.get('dataset') in precision and 'weights' in r and 'posterior_precision' not in r:
                guard=precision[r['dataset']];r=qualify_bound(r,guard['Rmax'],guard)
            r=strict_budget(r)
            status[r['status']]+=1
            if stage=='S2_confirmation' and r.get('method') in ('A1-v3','A1-B') and r.get('beta')=='inf' and 'weights' in r:
                key=(r['dataset'],r['epsilon_scale'],r['epsilon_ratio']);unbudgeted_pairs.setdefault(key,{})[r['method']]={k:r[k] for k in ('optimal_mass_lower','optimal_mass_upper','success_mass_lower','mean_increment_lower','mean_increment_upper','status','posterior_mean_regret_MA_upper')}
            if 'weights' not in r:
                append(failure_records,dict(stage=stage,**r));continue
            w=r['weights'];cap=max(.2,2/r['d']) if stage=='S1' else .2
            res=max(r['primal_residual'],abs(sum(w)-1),-min(w),max(w)-cap,0.);maxres=max(maxres,res)
            if res>1e-9:errors.append(dict(stage=stage,key=r.get('dataset',r.get('case')),error='primal residual',value=res))
            if r['success_mass_lower']>r['success_mass_upper']+1e-12:errors.append(dict(stage=stage,error='success interval reversed'))
            if r['optimal_mass_lower']>r['optimal_mass_upper']+1e-12:errors.append(dict(stage=stage,error='objective interval reversed'))
            if r['posterior_mean_regret_method_upper']<r['posterior_mean_regret_MA_lower']-1e-12:
                errors.append(dict(stage=stage,error='returned holding contradicts MA mean optimality'))
            if r['status']=='optimal' and r['primary_mass_gap']>1e-8:
                if r.get('method')=='A1-v3':flags['legacy_A1_status_does_not_close_new_interval']+=1
                else:errors.append(dict(stage=stage,error='false primary optimal label'))
            if r.get('budget_certified') and r['beta']!='inf' and r['mean_increment_upper']>r['budget_lower']:errors.append(dict(stage=stage,error='false budget certificate'))
            if not r.get('budget_certified'):uncertified+=1
            if r['status']=='optimal' and not r.get('budget_certified'):flags['primary_closed_budget_numerically_open']+=1
            if r.get('true_success_lower')!=r.get('true_success_upper'):uncertain+=1
            maxgap=max(maxgap,r['primary_mass_gap'])
            if r['status'] in ('timeout','failed','open_mass_gap','master_precision_limited','budget_precision_limited','success_optimal_secondary_open','posterior_precision_open','geometry_indeterminate','MA_endpoint_numerical_open') or r['status'].startswith('scip_'):
                append(failure_records,dict(stage=stage,key=r.get('dataset',r.get('case',r.get('query'))),method=r.get('method'),beta=r['beta'],epsilon=r.get('epsilon'),true_model=r.get('true_model'),status=r['status'],primary_gap=r['primary_mass_gap'],secondary_gap=r['secondary_price_gap'],budget_certified=r['budget_certified']))
        finish_group()
        if malformed:errors.append(dict(stage=stage,error='malformed decision grid',datasets=malformed))
        counts[stage]=dict(rows=n,status=dict(status),max_residual=maxres,max_primary_gap=maxgap,uncertain_success_rows=uncertain,uncertified_budget_rows=uncertified,dataset_grid=dict(counts=grid,method_rows=dict(method_rows),malformed=malformed))
    same_primary=0
    for key,pair in unbudgeted_pairs.items():
        if len(pair)<2:continue
        a=pair['A1-B'];old=pair['A1-v3'];same_primary+=1
        if max(a['optimal_mass_lower'],old['optimal_mass_lower'])>min(a['optimal_mass_upper'],old['optimal_mass_upper'])+1e-8:errors.append(dict(error='old/new unconstrained objectives disagree',key=key))
        if a['status']=='optimal' and old['success_mass_lower']>=a['optimal_mass_lower']-1e-12 and a['mean_increment_lower']>old['mean_increment_upper']+1e-6*a['posterior_mean_regret_MA_upper']:
            errors.append(dict(error='global secondary is dominated by old feasible holding',key=key))
    flags['old_new_unbudgeted_pairs_checked']=same_primary
    # Independently check interval compatibility across every same-target S3 solver.
    comparisons=collections.defaultdict(list)
    for r in rows(OUT/'S3_computation.jsonl'):
        r=strict_budget(r)
        if 'weights' in r:comparisons[(r['case'],str(r['beta']))].append(r)
    for key,group in comparisons.items():
        for a,b in itertools.combinations(group,2):
            if max(a['optimal_mass_lower'],b['optimal_mass_lower'])>min(a['optimal_mass_upper'],b['optimal_mass_upper'])+1e-8:
                errors.append(dict(error='same-target S3 intervals disagree',key=key,methods=[a['method'],b['method']]))
        for a,b in itertools.permutations(group,2):
            tol=1e-6*a['posterior_mean_regret_MA_upper']
            if a.get('primary_exact_at_float_posterior') and float(a['secondary_price_gap'])<=tol and b.get('budget_certified') and b['success_mass_lower']>=a['optimal_mass_lower']-1e-12 and a['mean_increment_lower']>b['mean_increment_upper']+2*tol:
                errors.append(dict(error='S3 claimed global secondary dominated by feasible comparator',key=key,methods=[a['method'],b['method']]))
    flags['S3_same_target_budget_groups_checked']=len(comparisons)
    # Complete expected grid is enumerated independently of the result file.
    missing=[]
    for split in ('development','confirmation'):
        suffix='_validated_completed' if 'numerical_revision' in cfg else '_completed'
        done={r['dataset'] for r in rows(OUT/f'S2_{split}{suffix}.jsonl')};nr=cfg[f'{split}_repetitions'];fs=[make(*s) for s in specs(split)]
        jobs=[(0,0,0,mask,f) for mask in ('Separated','Joint') for f in fs]
        jobs += [(rep,j,n,mask,f) for rep in range(nr) for j in range(6) for n in cfg['n_blocks'] if n for mask in ('Separated','Joint') for f in fs]
        for rep,j,n,mask,f in jobs:
            ds=f'S2-{split}-{f.name}-{mask}-{n}-{j}-{rep}'
            if ds not in done:missing.append(dict(stage=split,dataset=ds))
    done={(r['case'],r['method'],str(float(r['beta']))) for r in rows(OUT/'S3_computation.jsonl')}
    for d,M,kind,ratio,beta,method in itertools.product((10,20),(6,12,24),('asymmetric','hidden_curvature','block'),(.5,.75),BETAS,('A1-B-cold','A1-B-cached','SCIP','Enumeration')):
        if method=='Enumeration' and M>10:continue
        case=f'{kind}-{d}-{M}-computation-0-{ratio}'
        if (case,method,str(float(beta))) not in done:missing.append(dict(stage='S3',case=case,method=method,beta=beta))
    pressure=list(rows(OUT/'S3_perturbation.jsonl'));degenerate={r['index'] for r in pressure if r['location']=='degenerate'};done_pressure={(r['index'],r['location'],r['delta']) for r in pressure}
    for i,location,delta in itertools.product(range(20),('far','below','above'),(0,1e-4,1e-3,1e-2)):
        if i not in degenerate and (i,location,delta) not in done_pressure:missing.append(dict(stage='S3_perturbation',index=i,location=location,delta=delta))
    # Re-generate saved raw likelihood evidence and directly evaluate masked Gaussian densities.
    likelihood_checks=[];tested=set()
    for r in rows(OUT/'S2_confirmation_evidence.jsonl'):
        k=(r['kind'],r['mask_design'])
        if not r['n_blocks'] or k in tested:continue
        tested.add(k);index=int(r['family'].rsplit('-',1)[1]);f=make(r['kind'],r['d'],6,'confirmation',index)
        Y,ll=observations(f,r['true_model_for_evaluator'],r['n_blocks'],r['mask_design'],r['data_seed']);direct=np.zeros(6)
        for S in masks(f,r['mask_design']):
            data=Y[:,S];data=data[np.isfinite(data).all(1)]
            for j in range(6):
                Q=f.Q[j][np.ix_(S,S)];direct[j]+=-.5*(len(data)*(len(S)*np.log(2*np.pi)+np.linalg.slogdet(Q)[1])+np.sum(data*np.linalg.solve(Q,data.T).T))
        err=max(np.max(abs(direct-ll)),np.max(abs(ll-np.array(r['log_observation_density']))));likelihood_checks.append(dict(dataset=r['dataset'],maximum_absolute_error=err));
        if err>1e-8:errors.append(dict(error='likelihood mismatch',dataset=r['dataset']))
        if len(tested)==6:break
    write(OUT/'missing_cells.json',missing)
    cpu=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl'))
    write(OUT/'audit.json',dict(passed=not errors,errors=errors,flags=dict(flags),counts=counts,missing_count=len(missing),likelihood_checks=likelihood_checks,cpu_seconds_before_this_audit=cpu,expected_confirmation_datasets=4344,expected_S3_queries=600))
    print(json.dumps(dict(passed=not errors,errors=len(errors),missing=len(missing),cpu_hours=cpu/3600)),flush=True)
    if errors:raise RuntimeError('record audit failed; see audit.json')

if __name__=='__main__':run('audit',audit)
