"""Frozen, resumable real-data stages with a measured process CPU budget."""
import argparse
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from .common import PRIVATE, ROOT, dump, read, ledger, seed
from .data import Data
from .model import fit_scenarios, decisions, predictive_log_score, POWERS, SHRINK, PARAMS, regularize, stable_solve
from .book import isolated_risk
from ..scenario_inference import ScenarioSet,posterior_from_loglik
from ..new_generators import em_initialized
from ..risk_envelopes import RiskSpec
from ..risk_center import solve_center
from core import solve_qp

DATA=None
def data():
    global DATA
    if DATA is None:DATA=Data()
    return DATA

def initialize():
    rows=read(PRIVATE/'manifest_all.json'); D=data()
    selected=[r for r in rows if r['d']==20 and r['panel']==0 and r['period']=='development' and r['ids']]
    scale=[];units=[]
    for r in selected:
        Y=D.past(r['t'],r['ids']);scale.append(float(np.nanmedian(np.nanvar(Y,axis=0))))
        centered=Y-np.nanmean(Y,axis=0)
        # All selected constituents have enough history; EW row averages are only scale calibration.
        units.append(float(.5*np.nanmean(np.nanmean(centered,axis=1)**2)))
    cfg=dict(version=1,data_scale=float(np.median(scale)),risk_unit_raw=float(np.median(units)),
             development_windows=len(selected),cpu_limit_seconds=43200,workers=4,
             lookback=252,fit=189,score=63,horizon=21,size=256,upper=.2,
             hyperparameters=dict(powers=POWERS,shrink=SHRINK,lambda_grid=PARAMS,tau_grid=PARAMS),
             started_unix=time.time(),date_splits=['2013-2016','2017-2019','2020-2025','2026 extension'])
    cfg['risk_unit']=cfg['risk_unit_raw']/cfg['data_scale']
    dump(ROOT/'configs/base.json',cfg); print(cfg,flush=True)

def scene(row,size=256,scramble=0):
    D=data();cfg=read(ROOT/'configs/base.json'); Y=D.past(row['t'],row['ids'])/np.sqrt(cfg['data_scale'])
    path=PRIVATE/'scenes'/f"{row['key']}_{size}_{scramble}.npz"; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        a=np.load(path)
        sc=ScenarioSet(a['Q'],a['nodes'],a['groups'],a['lr'],upper=.2,oracle_lower=a['ol'],oracle_upper=a['ou'],
                       source='nested bootstrap reference; Gaussian/generalized score').prepare()
        return sc,a['ll'],a['nominal'],a['mean'],Y,dict(cached=True,scenario_count=size,oracle_seconds=0.)
    sc,ll,nominal,mean,diag=fit_scenarios(Y,size,seed(row['seed'],scramble))
    np.savez(path,Q=sc.Q,nodes=sc.nodes,groups=sc.groups,lr=sc.log_reference,ol=sc.oracle_lower,ou=sc.oracle_upper,
             ll=ll,nominal=nominal,mean=mean)
    dump(path.with_suffix('.json'),diag)
    return sc,ll,nominal,mean,Y,diag

def work(task):
    start=time.perf_counter();cpu=time.process_time();row=task['row'];stage=task['stage'];size=task.get('size',256);scramble=task.get('scramble',0)
    output=PRIVATE/'records'/stage/f"{row['key']}_{size}_{scramble}.json"
    if output.exists():return dict(cached=True,key=row['key'])
    record=dict(key=row['key'],date=row['date'],period=row['period'],cohort=row['cohort'],panel=row['panel'],d=row['d'],
                t=row['t'],ids=row['ids'],size=size,scramble=scramble,stage=stage)
    try:
        cfg=read(ROOT/'configs/base.json');unit=cfg['risk_unit']
        sc,ll,nominal,mean,Y,diag=scene(row,size,scramble)
        record['scene']=diag
        if stage=='scores':
            # Future data enters only this evaluator, never fit_scenarios or a decision API.
            D=data();future=np.array(D.a['return_total'][row['t']+1:row['t']+22,row['ids']],dtype=float)/np.sqrt(cfg['data_scale'])
            good=D.a['observed'][row['t']+1:row['t']+22,row['ids']] & ~D.a['total_adjustment_suspect'][row['t']+1:row['t']+22,row['ids']]
            future[~good]=np.nan
            record['scores']={str(p):predictive_log_score(sc,ll,mean,future,p) for p in POWERS}
        else:
            choice=read(ROOT/'configs/selected.json') if (ROOT/'configs/selected.json').exists() else dict(power=1.,shrink=.25,lam=.01,tau=.1)
            answers,postdiag=decisions(sc,ll,nominal,Y,unit,**choice,audit=stage=='audit' or row['d']==50)
            record['posterior']=postdiag
            if stage=='tune':
                post=posterior_from_loglik(sc,ll,power=choice['power'])
                for kind in ('lambda','tau'):
                    for v in PARAMS:
                        spec=RiskSpec(f'{kind}={v}','max',penalty=v) if kind=='lambda' else RiskSpec(f'{kind}={v}','soft',temperature=v)
                        a=stable_solve(sc,post,spec,unit)
                        answers.append(dict(method=spec.name,weights=a.weights.tolist(),gap=a.gap,status=a.status,seconds=a.seconds))
                Q=regularize(em_initialized(Y-np.nanmean(Y,axis=0),maxiter=40))
                for v in SHRINK:
                    a=solve_qp((1-v)*Q+v*np.diag(np.diag(Q)),constraints={'upper':.2})
                    answers.append(dict(method=f'shrink={v}',weights=a['w'].tolist(),status='baseline',gap=a['value_ub']-a['value_lb']))
            if stage in ('test','audit'):
                # Same fitted scene and original score power; only diagnostic core methods.
                if choice['power']!=1.:
                    p1=posterior_from_loglik(sc,ll,power=1.)
                    for spec in [RiskSpec('MA-power1','mean'),RiskSpec('B-Tail-0.5-power1',conditional=True)]:
                        a=solve_center(sc,p1,spec,unit)
                        answers.append(dict(method=spec.name,weights=a.weights.tolist(),status=a.status,gap=a.gap,seconds=a.seconds,
                                             objective=a.objective,power1_ess=p1.diagnostics['ess']))
            for a in answers:
                if a.get('weights') is not None:
                    a['evaluation']=isolated_risk(data(),row['t'],row['ids'],a['weights'])
            record['methods']=answers
        record['status']='ok'
    except Exception:
        record['status']='exception';record['traceback']=traceback.format_exc()
    import psutil
    memory=psutil.Process().memory_info()
    record['worker_rss_bytes']=memory.rss
    record['worker_peak_rss_bytes']=getattr(memory,'peak_wset',memory.rss)
    record['seconds']=time.perf_counter()-start;record['cpu_seconds']=time.process_time()-cpu
    dump(output,record)
    return dict(key=row['key'],status=record['status'],seconds=record['seconds'],cpu_seconds=record['cpu_seconds'])

def tasks(stage):
    allrows=read(PRIVATE/'manifest_all.json');timing=set(read(PRIVATE/'timing_keys.json'))
    n=read(ROOT/'configs/scale.json')['panels'] if (ROOT/'configs/scale.json').exists() else 1
    rows=[r for r in allrows if r['ids'] is not None and r['d']==20 and r['panel']<n]
    if stage=='timing':rows=[r for r in rows if r['key'] in timing]
    elif stage in ('scores','tune'):rows=[r for r in rows if r['period']=='validation' and r['evaluation_end']<='2019-12-31']
    elif stage=='test':rows=[r for r in rows if r['period'] in ('test','extension')]
    elif stage=='audit':
        # First eligible window of each specified year: selected independently of performance.
        selected=[]
        for y in (2020,2022,2024,2026):
          for c in ('long','limited'):
            selected.append(next(r for r in rows if r['panel']==0 and r['date'].startswith(str(y)) and r['cohort']==c))
        return [dict(stage=stage,row=r,size=size,scramble=s) for r in selected for size in (128,256,512) for s in range(3)]
    elif stage=='fifty':
        rows=[r for r in allrows if r['ids'] is not None and r['d']==50 and r['period'] in ('test','extension')]
    return [dict(stage=stage,row=r) for r in rows]

def execute(stage,limit):
    assert (ROOT/'output/gates.json').exists(),'Run gates before performance experiments'
    import psutil
    todo=tasks(stage);dump(PRIVATE/f'tasks_{stage}.json',todo)
    todo=[t for t in todo if not (PRIVATE/'records'/stage/f"{t['row']['key']}_{t.get('size',256)}_{t.get('scramble',0)}.json").exists()]
    start=time.perf_counter();cpu=time.process_time();done=[];stop=False;pending={};index=0
    historical=sum(read(p)['cpu_seconds'] for p in (PRIVATE/'ledger').glob('run_*.json'))
    allowed=min(limit*3600,max(0,43200-historical));workers=1 if stage=='timing' else 4
    print(stage,'tasks',len(todo),'workers',workers,'CPU allowed',allowed,flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
      while pending or (index<len(todo) and not stop):
        used=time.process_time()-cpu
        for child in psutil.Process().children():
            try:ct=child.cpu_times();used+=ct.user+ct.system
            except psutil.Error:pass
        if used>=allowed:stop=True
        while not stop and index<len(todo) and len(pending)<workers:
            task=todo[index];pending[pool.submit(work,task)]=task;index+=1
        if not pending:break
        ready,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
        for future in ready:
            task=pending.pop(future)
            try:r=future.result()
            except Exception:r=dict(key=task['row']['key'],status='worker_exception',error=traceback.format_exc())
            done.append(r)
            if len(done)%10==0 or r.get('status') not in ('ok',None) or stage=='timing' or len(done)==len(todo):
                print(stage,len(done),'/',len(todo),r,flush=True)
      # Includes imports and idle-worker CPU, not merely sums of solver timers.
      used=time.process_time()-cpu
      for child in psutil.Process().children():
          try:ct=child.cpu_times();used+=ct.user+ct.system
          except psutil.Error:pass
    name='run_'+stage+'_'+str(int(start))
    ledger(name,time.perf_counter()-start,used,dict(completed=len(done),planned=len(todo),budget_stopped=stop,workers=workers))
    dump(PRIVATE/(name+'.json'),dict(tasks=done,not_dispatched=todo[index:]))

def freeze_scale():
    timing=[read(p) for p in (PRIVATE/'records/timing').glob('*.json')]
    assert len(timing)==10 and all(r['status']=='ok' for r in timing)
    cost=max(float(np.percentile([r['cpu_seconds'] for r in timing],90)),.01)
    rows=read(PRIVATE/'manifest_all.json');counts={p:sum(r['d']==20 and r['panel']==0 and r['period']==p for r in rows) for p in ('validation','test','extension')}
    # Two validation passes, a larger tuning grid, and safety factor; no performance input.
    val=cost*counts['validation']*2.5; test=cost*(counts['test']+counts['extension'])*1.3
    panels=3 if val*3<=7200 and test*3<=21600 else 1
    cfg=dict(panels=panels,timing_cpu_p90=cost,estimated_validation_cpu=val*panels,estimated_test_cpu=test*panels,
             selection_basis='ten-window CPU timing only; no risk/rank input',counts_one_panel=counts)
    dump(ROOT/'configs/scale.json',cfg);print(cfg,flush=True)

def choose_scores():
    eligible={t['row']['key'] for t in tasks('scores')}
    records=[read(p) for p in (PRIVATE/'records/scores').glob('*.json')]
    records=[r for r in records if r['key'] in eligible]
    expected=len(tasks('scores'));assert len(records)==expected and all(r['status']=='ok' for r in records)
    means={str(p):float(np.mean([r['scores'][str(p)] for r in records])) for p in POWERS}
    power=float(max(means,key=means.get));dump(ROOT/'configs/selected.json',dict(power=power,shrink=.25,lam=.01,tau=.1))
    dump(ROOT/'output/power_selection.json',dict(scores=means,criterion='validation whole-block Gaussian log predictive density per observed scalar',windows=len(records),selected=power))
    print('power',power,means,flush=True)

def choose_tuning():
    eligible={t['row']['key'] for t in tasks('tune')}
    records=[read(p) for p in (PRIVATE/'records/tune').glob('*.json')]
    records=[r for r in records if r['key'] in eligible]
    assert len(records)==len(tasks('tune')) and all(r['status']=='ok' for r in records)
    cfg=read(ROOT/'configs/selected.json');detail={}
    for kind,grid in [('lambda',PARAMS),('tau',PARAMS),('shrink',SHRINK)]:
        values={}
        for v in grid:
            a=[next(a for a in r['methods'] if a['method']==f'{kind}={v}') for r in records]
            assert all(x.get('weights') is not None for x in a)
            values[str(v)]=float(np.mean([x['evaluation']['second_moment'] for x in a]))
        selected=float(min(values,key=values.get));cfg['lam' if kind=='lambda' else kind]=selected
        detail[kind]=dict(values=values,selected=selected)
    dump(ROOT/'configs/selected.json',cfg);dump(ROOT/'output/tuning.json',dict(criterion='validation fresh-account future 21 opening-to-opening second moment',choices=detail,windows=len(records)))
    import hashlib
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    dump(ROOT/'configs/test_freeze.json',dict(frozen_unix=time.time(),parameters=cfg,
        validation_keys_sha256=hashlib.sha256('\n'.join(sorted(eligible)).encode()).hexdigest(),
        source_hashes=sources,test_windows=len(tasks('test')),audit_tasks=len(tasks('audit')),fifty_tasks=len(tasks('fifty')),
        selection_rule='validation only; no test performance loaded'))
    print('frozen',cfg,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--hours',type=float,default=1.)
    args=parser.parse_args()
    if args.stage=='initialize':initialize()
    elif args.stage=='freeze':freeze_scale()
    elif args.stage=='choose-power':choose_scores()
    elif args.stage=='choose-tuning':choose_tuning()
    else:execute(args.stage,args.hours)
