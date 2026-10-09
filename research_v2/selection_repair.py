"""Transparent exploratory repair of the mean/worst-group selection criterion.

Original confirmation stays frozen and is never relabeled as fresh confirmation.
"""
import json,time,multiprocessing as mp
from dataclasses import asdict
import numpy as np
import pandas as pd
from core import solve_qp
from .experiment_v2 import OUT,CFG,ROOT,load_rows,write_json,append_row,source_hash
from .registry import all_specs,blend_specs
from .new_generators import family,truth_points,data_instance,scenarios,importance_scenarios,seed_for
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec
from .risk_center import solve_center

def work(task):
    r,source_stage,specdicts=task;tic=time.perf_counter();cpu=time.process_time()
    setup=json.loads((OUT/'development_setup.json').read_text());unit=setup['risk_unit']
    f=family(r['kind'],r['family_seed']);theta=truth_points(r['point_seed'])[r['point_id']]
    Y=data_instance(f,theta,r['n'],r['m'],r['data_seed']);seed=seed_for('quadrature',r['kind'],r['family_seed'])
    sc=importance_scenarios(f,Y,256,seed) if r['kind']=='G2' else scenarios(f,128,seed)
    p=posterior(sc,Y);Q=f.Q(theta);oracle=solve_qp(Q,constraints={'upper':.2})
    rng=np.random.default_rng(seed_for('future',source_stage,r['instance_id']));future=rng.normal(size=(128,20))@np.linalg.cholesky(Q).T
    rows=[]
    for sd in specdicts:
        spec=RiskSpec(**sd)
        try:
            ans=solve_center(sc,p,spec,unit);w=ans.weights
            rows.append(r|{'method':spec.name,'spec':sd,'source_stage':source_stage,'unit':unit,
                'weights':w.tolist(),'regret_scaled':float((.5*w@Q@w-(oracle['value_lb']+oracle['value_ub'])/2)/unit),
                'validation_loss':float(.5*np.mean((future@w)**2)),'status':ans.status,'gap':ans.gap,
                'lower':ans.lower,'upper':ans.upper,'objective':ans.objective,'seconds':ans.seconds,
                'adversary':ans.adversary.tolist(),'posterior':p.diagnostics,'alpha':np.exp(p.log_alpha).tolist(),
                'diagnostics':ans.diagnostics,'failure':False})
        except Exception as exc:rows.append(r|{'method':spec.name,'source_stage':source_stage,'failure':True,'error':repr(exc)})
    return rows,time.perf_counter()-tic,time.process_time()-cpu

def panel(stage,tasks):
    path=OUT/(stage+'.jsonl');old=load_rows(stage);done={(r['instance_id'],r['method']) for r in old}
    tasks=[(r,s,[a for a in methods if (r['instance_id'],a['name']) not in done]) for r,s,methods in tasks]
    tasks=[t for t in tasks if t[2]];wall=cpu=0.;sha=source_hash()
    with mp.get_context('spawn').Pool(8) as pool:
        for i,(rows,w,c) in enumerate(pool.imap_unordered(work,tasks,chunksize=4)):
            wall+=w;cpu+=c
            for row in rows:append_row(path,row|{'stage':stage,'source_hash':sha,'evidence_status':'exploratory selection repair; original confirmation already inspected'})
            if i%100==0:print(stage,i+1,'/',len(tasks),flush=True)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':stage,'wall_seconds':wall,'process_cpu_seconds':cpu,'unix':time.time(),
        'timing_scope':'sum of task work durations across eight workers'})

def run():
    development=json.loads((CFG/'development_manifest.json').read_text())
    panel('blend_development',[(r,'development',[asdict(s) for s in blend_specs()]) for r in development])
    df=pd.DataFrame(load_rows('development')+load_rows('blend_development'));good=df[~df.failure]
    unit=good.unit.iloc[0];means=good.groupby('method').validation_loss.mean()
    worst=good.groupby(['method','kind','family_id']).validation_loss.mean().groupby('method').max()
    table=pd.DataFrame({'mean_excess':(means-means['A-Mean'])/unit,'worst_group_excess':(worst-worst['A-Mean'])/unit,
        'count':good.groupby('method').size(),'all_converged':good.groupby('method').status.apply(lambda x:(x=='converged').all())})
    valid=table[(table['count']==24)&table.all_converged].drop(index=['A-Mean','B-Mean'])
    tail=list(valid[(valid.mean_excess<=.005)&(valid.worst_group_excess<0)].sort_values('worst_group_excess').head(2).index)
    lookup={s.name:s for s in all_specs()+blend_specs()}
    config={'criterion':'max group mean realized validation loss minus max group MA loss; groups=(kind,family_id)',
        'mean_increase_budget':.005,'tradeoff_candidates':tail,'blend_trigger':bool(((valid.mean_excess>0)&(valid.worst_group_excess<0)).any()),
        'original_confirmation_preserved':True,'supplement_status':'exploratory; not a new untouched confirmation',
        'data_manifests_and_seeds':'unchanged original G1/G2/confirm','selection_uses':'development future realized loss only',
        'created_unix':time.time(),'table':table.reset_index().to_dict('records')}
    write_json(CFG/'selection_repair.json',config);print('repaired tradeoff candidates',tail,flush=True)
    screen_specs=[asdict(lookup[n]) for n in list(dict.fromkeys(tail+['A-Blend','B-Blend']))]
    tasks=[]
    for stage in ('G1','G2'):
        tasks +=[(r,stage,screen_specs) for r in json.loads((CFG/(stage+'_manifest.json')).read_text())]
    panel('repair_screen',tasks)
    original={s['name'] for s in json.loads((CFG/'selection.json').read_text())['confirmation_specs']}
    added=[asdict(lookup[n]) for n in tail if n not in original]
    panel('repair_confirmation',[(r,'confirm',added) for r in json.loads((CFG/'confirm_manifest.json').read_text())])

if __name__=='__main__':run()
