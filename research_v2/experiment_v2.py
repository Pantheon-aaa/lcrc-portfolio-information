"""Frozen, serial, resumable experiments. Run from the repository root with -m."""
import argparse, json, hashlib, platform, sys, traceback
from pathlib import Path
from time import perf_counter, time, process_time
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import psutil
from core import solve_qp,em_cov,lcrc_affine
from .scenario_inference import posterior,posterior_from_loglik,component_logmasses,mixture_logprior,fit_eb
from .new_generators import family,scenarios,truth_points,data_instance,seed_for,factor_path,estimated_scenarios,importance_scenarios
from .registry import all_specs,core_specs,blend_specs
from .risk_envelopes import RiskSpec
from .risk_center import solve_center

ROOT=Path(__file__).parent; OUT=ROOT/'results'; CFG=ROOT/'configs'
OUT.mkdir(exist_ok=True); CFG.mkdir(exist_ok=True)


def clean(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,default=clean,ensure_ascii=False),encoding='utf-8')


def source_hash():
    h=hashlib.sha256()
    for p in sorted(ROOT.glob('*.py')): h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def manifest(stage,reps=10):
    rows=[]
    if stage in ('pilot','development'):
        count=10 if stage=='pilot' else 24
        for i in range(count):
            kind='G1' if i%2==0 else 'G2'; family_id=i%3
            rows.append(dict(kind=kind,family_id=family_id,point_id=i%8,rep=i,n=64 if i%2 else 256,m=(0,4,16,64)[i%4]))
    else:
        kinds=['G1','G2'] if stage=='confirm' else [stage]
        for kind in kinds:
            for f in range(3):
                for point in range(8):
                    for n,m in ([(256,16)] if kind=='G1' else [(n,m) for n in (64,256) for m in (0,4,16,64)]):
                        for rep in range(reps): rows.append(dict(kind=kind,family_id=f,point_id=point,rep=rep,n=n,m=m))
    partition='confirm' if stage=='confirm' else ('development' if stage in ('pilot','development') else 'screen')
    for r in rows:
        r['family_seed']=seed_for('family',partition,r['kind'],r['family_id'])
        r['point_seed']=seed_for('points',partition,r['kind'],r['family_id'])
        r['data_seed']=seed_for('data',stage,*r.values())
        r['instance_id']=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()[:16]
    return rows


def init():
    assert json.loads((OUT/'gate_v2.json').read_text())['passed']
    cfg={'created_unix':time(),'source_hash':source_hash(),'budget_seconds':86400,
         'stage_budgets_hours':[2,4,8,4,6],'risk_unit_policy':'median independent development equal-weight future loss',
         'seed_policy':'SHA256 stable keys; all predeclared, no outcome filtering',
         'scenarios':128,'blas_threads':1,'screen_reps':10,'confirm_reps':30,
         'development_specs':[asdict(s) for s in all_specs()],
         'method_aliases':{'A-ATK':'A-Tail-U with equal base masses','B-ATK':'B-Tail-U with equal conditional base masses',
            'REF-MA':'A-Mean','REF-MAX':'A-Max'},
         'g3_variants':['complete','uneven','correlation','volatility','t6','informative'],
         'g3_dimensions':[20,50],'g3_paths_per_cell':3,'g3_windows_per_path':12,
         'confirmation_policy':'fixed candidates before confirmation; only development future losses tune parameters',
         'public_push':False,'real_data':False,'environment':{'python':sys.version,'platform':platform.platform()}}
    path=CFG/'protocol.json'
    if path.exists(): raise RuntimeError('protocol already frozen; do not overwrite')
    write_json(path,cfg)
    for stage,reps in [('pilot',1),('development',1),('G1',10),('G2',10),('confirm',30)]:
        write_json(CFG/f'{stage}_manifest.json',manifest(stage,reps))
    print('Protocol and manifests frozen before performance runs.')


def append_row(path,row):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,default=clean,ensure_ascii=False,allow_nan=True)+'\n'); f.flush()


def load_rows(stage):
    p=OUT/f'{stage}.jsonl'
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []


def baseline_weights(name,fam,Y,post,sc,unit,eb):
    if name=='REF-EB':
        p=posterior_from_loglik(sc,post.loglik,mixture_logprior(sc,eb))
        return solve_center(sc,p,RiskSpec(name,'mean',rho=1),unit)
    if name.startswith('OraclePrior-'):
        j=int(name.split('-')[1]); p=posterior_from_loglik(sc,post.loglik,component_logmasses(sc)[j])
        return solve_center(sc,p,RiskSpec(name,'mean',rho=1),unit)
    return None


def single_method(sc,post,spec,unit,eb=None):
    if spec.name=='REF-EB':
        post=posterior_from_loglik(sc,post.loglik,mixture_logprior(sc,eb))
    if spec.name=='B-Plugin':
        # Discrete posterior mode outer node, retaining its entire conditional law.
        ix=sc.groups==np.argmax(post.log_alpha)
        from .scenario_inference import ScenarioSet
        subset=ScenarioSet(sc.Q[ix],sc.nodes[ix],np.zeros(ix.sum(),int),sc.log_reference[ix],
            mu=sc.mu[ix],upper=sc.upper,oracle_lower=sc.oracle_lower[ix],oracle_upper=sc.oracle_upper[ix]).prepare()
        subpost=posterior_from_loglik(subset,post.loglik[ix])
        return solve_center(subset,subpost,RiskSpec('B-Plugin',rho=spec.rho),unit)
    return solve_center(sc,post,spec,unit)


def setup_development():
    # Independent development realizations only; no true covariance reaches selector.
    values=[]; evidence=[]; data=[]
    for i in range(24):
        kind='G1' if i%2==0 else 'G2'; fam=family(kind,seed_for('dev_unit_family',i))
        rng=np.random.default_rng(seed_for('devunit',i)); theta=rng.uniform(-1,1,3)
        Y=data_instance(fam,theta,128,(0,4,16,64)[i%4],seed_for('devunitdata',i))
        future=rng.normal(size=(128,20))@np.linalg.cholesky(fam.Q(theta)).T
        values.append(float(.5*np.mean((future@np.ones(20)/20)**2)))
        sc=(scenarios(fam,128,seed_for('devnodes',i)) if kind=='G1' else
            importance_scenarios(fam,Y,256,seed_for('devnodes',i)))
        post=posterior(sc,Y)
        evidence.append(logsumexp(component_logmasses(sc)+post.loglik,axis=1))
    eb,diag=fit_eb(evidence)
    write_json(OUT/'development_setup.json',{'risk_unit':float(np.median(values)),'risk_unit_samples':values,
        'eb_weights':eb,'eb_fit':diag,'used_truth_labels':False,'note':'future realizations for scale; observed likelihood for EB'})
    print('development unit',np.median(values),'EB',eb,flush=True)


def run(stage,limit=None,shard=None):
    assert json.loads((OUT/'gate_v2.json').read_text())['passed']
    if stage in ('G2','confirm'):
        review=CFG/'integration_review.json'
        if review.exists() and json.loads(review.read_text())['status']!='passed':
            raise RuntimeError('G2 integration accuracy review pending; stop dependent performance stage')
    cfg=json.loads((CFG/'protocol.json').read_text()); setup=json.loads((OUT/'development_setup.json').read_text())
    unit=setup['risk_unit']; eb=np.array(setup['eb_weights'])
    if stage=='pilot':
        lookup={s.name:s for s in all_specs()}
        specs=[lookup[n] for n in ['A-Mean','A-Tail-P:0.5','B-Tail-P:0.5','B-Max','C-Soft:0.1','H-RankThenP']]
    elif stage=='development': specs=all_specs()
    elif stage=='confirm':
        sel=json.loads((CFG/'selection.json').read_text()); specs=[RiskSpec(**s) for s in sel['confirmation_specs']]
    else:
        sel=json.loads((CFG/'selection.json').read_text()); specs=core_specs()+[RiskSpec(**s) for s in sel['extra_screen_specs']]
        specs+=[RiskSpec('REF-EB','mean',rho=1),RiskSpec('B-Plugin',rho=.5)]
    specs=list({s.name:s for s in specs}.values())
    rows=json.loads((CFG/f'{stage}_manifest.json').read_text())
    if limit: rows=rows[:limit]
    output_stage=stage if shard is None else stage+'-shard-'+shard.replace('/','of')
    old=load_rows(stage)
    if shard is not None:
        shard_id,shard_count=map(int,shard.split('/'))
        assert 0<=shard_id<shard_count
        rows=[r for i,r in enumerate(rows) if i%shard_count==shard_id]
        old+=load_rows(output_stage)
    done={(r['instance_id'],r['method']) for r in old}; cache={}
    sha=source_hash(); begin=perf_counter(); cpu=process_time(); successes=0
    already_spent=sum(x['wall_seconds'] for x in load_rows('runtime_ledger'))
    for idx,r in enumerate(rows):
        if (CFG/'budget_stop.json').exists(): break
        if all((r['instance_id'],s.name) in done for s in specs): continue
        key=(r['kind'],r['family_seed'])
        if key not in cache:
            fam=family(r['kind'],r['family_seed']); sc=scenarios(fam,cfg['scenarios'],seed_for('quadrature',*key))
            cache[key]=(fam,sc)
        fam,sc=cache[key]; theta=truth_points(r['point_seed'])[r['point_id']]
        Y=data_instance(fam,theta,r['n'],r['m'],r['data_seed'])
        scenario_build_start=perf_counter()
        if r['kind']=='G2':
            sc=importance_scenarios(fam,Y,256,seed_for('quadrature',*key))
        scenario_build_seconds=perf_counter()-scenario_build_start
        start=perf_counter(); post=posterior(sc,Y); inference=perf_counter()-start
        trueQ=fam.Q(theta); oracle=solve_qp(trueQ,constraints={'upper':.2})
        rng=np.random.default_rng(seed_for('future',stage,r['instance_id']))
        future=rng.normal(size=(128,20))@np.linalg.cholesky(trueQ).T
        for spec in specs:
            if (r['instance_id'],spec.name) in done: continue
            base=r|{'stage':stage,'method':spec.name,'spec':asdict(spec),'source_hash':sha,'theta':theta,
                'unit':unit,'inference_seconds':inference,'scenario_oracle_seconds':sc.oracle_seconds,
                'posterior':post.diagnostics,'alpha':np.exp(post.log_alpha),'mask_scalar_count':int(np.isfinite(Y).sum()),
                'scenario_count':len(sc.Q),'scenario_seed':seed_for('quadrature',*key)}
            base['scenario_build_seconds']=scenario_build_seconds
            base['proposal_diagnostics']=getattr(sc,'proposal_diagnostics',None)
            try:
                ans=single_method(sc,post,spec,unit,eb)
                truecost=.5*ans.weights@trueQ@ans.weights
                base.update(weights=ans.weights,regret=float(truecost-(oracle['value_lb']+oracle['value_ub'])/2),
                    regret_scaled=float((truecost-(oracle['value_lb']+oracle['value_ub'])/2)/unit),
                    validation_loss=float(.5*np.mean((future@ans.weights)**2)),
                    oracle_lower=oracle['value_lb'],oracle_upper=oracle['value_ub'],
                    objective=ans.objective,lower=ans.lower,upper=ans.upper,gap=ans.gap,status=ans.status,
                    seconds=ans.seconds,residual=ans.residual,qp_calls=ans.qp_calls,diagnostics=ans.diagnostics,
                    adversary=ans.adversary,support=int(np.sum(ans.weights>1e-6)),failure=False)
                successes+=1
            except Exception as exc:
                base.update(failure=True,status='exception',error=repr(exc),traceback=traceback.format_exc())
            append_row(OUT/f'{output_stage}.jsonl',base)
        if idx%5==0 or idx==len(rows)-1:
            print(stage,idx+1,'/',len(rows),'elapsed',round(perf_counter()-begin,1),flush=True)
        # Resource budget never inspects performance; do not start another dataset.
        if already_spent+perf_counter()-begin>cfg['budget_seconds']:
            print('budget reached; remaining manifest entries explicitly pending',flush=True); break
    append_row(OUT/'runtime_ledger.jsonl',{'stage':output_stage,'wall_seconds':perf_counter()-begin,'process_cpu_seconds':process_time()-cpu,
        'peak_working_set':psutil.Process().memory_info().peak_wset if hasattr(psutil.Process().memory_info(),'peak_wset') else psutil.Process().memory_info().rss,
        'new_successful_rows':successes,'source_hash':sha,'unix':time()})


def candidate_stable(row,unit):
    if row.get('seconds',float('inf'))>60:return False
    if row.get('status')=='converged':return True
    if row.get('status')!='heuristic':return False
    starts=row.get('diagnostics',{}).get('starts',[])
    values=np.array([s['value'] for s in starts])
    return len(starts)==4 and np.isfinite(values).all() and np.ptp(values)<=1e-4*unit


def select():
    rows=load_rows('development'); df=pd.DataFrame(rows)
    assert len(df)==24*len(all_specs()),(len(df),24*len(all_specs()))
    good=df[~df.failure].copy(); unit=json.loads((OUT/'development_setup.json').read_text())['risk_unit']
    baseline=good[good.method=='A-Mean'].set_index('instance_id').validation_loss
    good['excess']=[(v-baseline.loc[i])/unit for v,i in zip(good.validation_loss,good.instance_id)]
    stats=good.groupby('method').agg(mean_excess=('excess','mean'),count=('excess','size'),seconds=('seconds','median'))
    worst=good.groupby(['method','kind','family_id']).validation_loss.mean().groupby('method').max()
    stats['worst_group_excess']=(worst-worst['A-Mean'])/unit
    stability=good.groupby('method').status.apply(lambda x:bool((x=='converged').all()))
    stats['all_converged']=stability
    good['numerically_stable']=[candidate_stable(r,unit) for r in good.to_dict('records')]
    stats['all_stable']=good.groupby('method').numerically_stable.all()
    valid=stats[(stats['count']==24)&(stats.seconds<60)&stats.all_stable]
    eligible=valid.drop(index=['A-Mean','B-Mean'],errors='ignore')
    mean_names=list(eligible[eligible.mean_excess<0].sort_values('mean_excess').head(2).index)
    tail_names=list(eligible[(eligible.mean_excess<=.005)&(eligible.worst_group_excess<0)].sort_values('worst_group_excess').head(2).index)
    # Mechanism candidates are fixed hypotheses, not selected on evaluator regret.
    mechanism=['B-Tail-P:0.25','B-Tail-P:0.5']
    lookup={s.name:s for s in all_specs()}
    extra=[]
    for prefix in ('C-LP-Max:','C-LP-Tail:','C-Soft:','C-Soft-Cond:','C-LR:','C-Budget:','H-'):
        sub=valid[valid.index.str.startswith(prefix)]
        if len(sub): extra.append(lookup[sub.mean_excess.idxmin()])
    # Blends require a non-dominated observed mean-versus-group tradeoff in development.
    trigger=any((valid.mean_excess>0)&(valid.worst_group_excess<0))
    if trigger: extra+=blend_specs()
    confirm_names=list(dict.fromkeys(['A-Mean','REF-EB']+mean_names+tail_names+mechanism+['B-Max','A-Tail-P:0.5']))
    lookup['REF-EB']=RiskSpec('REF-EB','mean',rho=1)
    selection={'created_unix':time(),'source_hash':source_hash(),'selection_uses':'independent realized future loss only; no true covariance/regret',
        'mean_candidates':mean_names,'tradeoff_candidates':tail_names,'mechanism_hypotheses':mechanism,
        'confirmation_specs':[asdict(lookup[n]) for n in confirm_names],
        'extra_screen_specs':[asdict(s) for s in extra],
        'blend_trigger':bool(trigger),'development_table':stats.reset_index().to_dict('records')}
    path=CFG/'selection.json'
    if path.exists(): raise RuntimeError('selection already frozen')
    write_json(path,selection); stats.to_csv(OUT/'development_selection.csv')
    print(json.dumps({k:selection[k] for k in ('mean_candidates','tradeoff_candidates','mechanism_hypotheses','blend_trigger')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['init','setup','pilot','development','select','G1','G2','confirm'])
    parser.add_argument('--limit',type=int); parser.add_argument('--shard'); a=parser.parse_args()
    if a.stage=='init': init()
    elif a.stage=='setup': setup_development()
    elif a.stage=='select': select()
    else: run(a.stage,a.limit,a.shard)
