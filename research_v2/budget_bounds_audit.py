"""Re-evaluate evidence-budget support upper bounds; retain original holdings/results."""
import json,time,multiprocessing as mp
import numpy as np
from .experiment_v2 import OUT,ROOT,load_rows,write_json,append_row,source_hash
from .new_generators import family,scenarios,importance_scenarios,truth_points,data_instance
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec,Envelope
from .risk_center import bounds,project_envelope

def audit(item):
    stage,row=item;tic=time.perf_counter();cpu=time.process_time()
    fam=family(row['kind'],row['family_seed']);theta=truth_points(row['point_seed'])[row['point_id']]
    Y=data_instance(fam,theta,row['n'],row['m'],row['data_seed'])
    sc=(importance_scenarios(fam,Y,256,row['scenario_seed']) if row['kind']=='G2' else scenarios(fam,128,row['scenario_seed']))
    post=posterior(sc,Y);spec=RiskSpec(**row['spec']);env=Envelope(sc,post,spec,row['unit'])
    q=project_envelope(np.array(row['adversary']),env);w=np.array(row['weights'])
    L,U,_=bounds(sc,env,w,q);gap=U-L
    revision=dict(lower=L,upper=U,gap=gap,adversary=q.tolist(),
        status='converged' if -1e-9*row['unit']<=gap<=1e-6*row['unit'] else 'gap_open',
        bounds_audit_seconds=time.perf_counter()-tic,bounds_audit_cpu_seconds=time.process_time()-cpu,
        bounds_audit='LP dual support upper; original holding and evaluator losses retained',
        original_lower=row['lower'],original_upper=row['upper'],original_gap=row['gap'])
    return stage,row['instance_id'],row['method'],revision

def run():
    tasks=[];loaded={}
    for stage in ('development','G1','G2'):
        loaded[stage]=load_rows(stage)
        tasks.extend((stage,r) for r in loaded[stage] if r['method'].startswith('C-Budget') and not r['failure'])
    archive=OUT/'pre_budget_support_bound';archive.mkdir(exist_ok=True)
    for stage,rows in loaded.items():
        p=archive/(stage+'.jsonl')
        if p.exists():raise RuntimeError('budget audit already archived')
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['method'].startswith('C-Budget')),encoding='utf8')
    results={}
    with mp.get_context('spawn').Pool(8) as pool:
        for i,(stage,instance,method,revision) in enumerate(pool.imap_unordered(audit,tasks,chunksize=4)):
            results[stage,instance,method]=revision
            if i%200==0:print('budget bounds',i+1,'/',len(tasks),flush=True)
    for stage,rows in loaded.items():
        for r in rows:
            key=stage,r['instance_id'],r['method']
            if key in results:r.update(results[key]);r['bounds_review_source_hash']=source_hash()
        temp=OUT/(stage+'.bounds.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        temp.replace(OUT/(stage+'.jsonl'))
    write_json(OUT/'budget_support_audit.json',{'rows':len(results),'holdings_changed':False,
        'max_gap':max(r['gap'] for r in results.values()),'min_gap':min(r['gap'] for r in results.values()),
        'open_gaps':sum(r['status']=='gap_open' for r in results.values()),
        'reason':'LP primal maximizer is not an upper support certificate; use repaired LP dual objective'})
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'budget_support_audit','wall_seconds':sum(r['bounds_audit_seconds'] for r in results.values()),
        'process_cpu_seconds':sum(r['bounds_audit_cpu_seconds'] for r in results.values()),'unix':time.time(),
        'timing_scope':'sum of individual audit work durations across eight workers'})

if __name__=='__main__':run()
