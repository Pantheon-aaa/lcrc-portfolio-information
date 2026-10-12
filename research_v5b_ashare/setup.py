"""Pre-test timing, independent acceptance checks and immutable experiment manifest."""
import time
import numpy as np
from scipy.special import logsumexp
from .common import *
from .data import Data
from .pipeline import panels,inp
from .risk import frontend,log_density,posterior
from .decision import Solver

def benchmark():
    from .runner import solve_job
    data=Data();cfg=dict(tau=read(OUT/'tau_calibration.json')['tau'],**read(OUT/'validation_selection.json'),eta_F2=.3)
    selected=[]
    for domain in ['csi300','csi500']:
        for year in ['2015','2016']:selected.extend([p for p in panels('development') if p['seed']==11 and p['domain']==domain and p['date'].startswith(year) and p['status']=='ready'][:5])
    jobs=[dict(id='timing_'+p['id'],stage='R2',panel=p,N=20,M=6) for p in selected]
    jobs += [dict(id='timing50_'+p['id'],stage='R2',panel=p,N=50,M=12) for p in selected[:4]]
    jobs += [dict(id='timing100_'+selected[0]['id'],stage='R4-N',panel=selected[0],N=100,M=12)]
    jobs += [dict(id='timingF2_'+selected[0]['id'],stage='F2',panel=selected[0],N=50,M=12,front='F2')]
    quarter=[p for p in panels('development') if p['seed']==11 and p['quarterly'] and p['status']=='ready' and p['date'].startswith(('2015','2016'))][:8]
    for p in quarter:
        if p['status']!='ready':continue
        t=p['t'];ids=data.members[p['date'],p['domain']];valid=data.a['valid'];cnt=valid[t-566:t+1].sum(0);recent=valid[t-62:t+1].sum(0)
        amt=data.a['amount'][t-19:t+1];ac=np.isfinite(amt).sum(0);adv=np.nansum(amt,axis=0)/np.maximum(ac,1)
        eligible=[j for j in ids if cnt[j]>=540 and recent[j]>=60 and ac[j]>=15 and adv[j]>0 and data.a['trading_status'][t,j]==1]
        q=dict(p,ranking=sorted(eligible,key=lambda j:digest(11,data.symbols[j])));jobs.append(dict(id='timingall_'+p['id'],stage='R4-index',panel=q,N=len(eligible),M=6))
    records=[]
    for job in jobs:
        if job['id'].startswith('timingall') and (OUT/'full_index_budget_bound.json').exists() and not (PRIVATE/'benchmark'/f'{job["id"]}.json').exists():
            records.append(dict(id=job['id'],N=job['N'],M=job['M'],front='F1',cpu_seconds=0.,status='not_run_optional_budget_excluded',time_is_lower_bound=True));continue
        r=solve_job(data,job,cfg,folder='benchmark',timing=True);records.append(dict(id=job['id'],N=job['N'],M=job['M'],front=job.get('front','F1'),cpu_seconds=r['cpu_seconds'],status=r['status']))
        print('BENCHMARK',job['id'],r['status'],round(r['cpu_seconds'],2),flush=True)
    write(OUT/'benchmark.json',records)

def validate_f2():
    data=Data();rows=[]
    for p in panels('validation'):
        if p['seed']!=11 or not p['quarterly'] or p['status']!='ready' or phase(data.dates[p['t']+1])!='validation':continue
        path=PRIVATE/'validation_f2'/f'{p["id"]}.json'
        if path.exists():rows.append(read(path));continue
        f=frontend(data,p,mode='F2');x=data.returns(p['t']+1,f['columns'],1)[0];ll=np.array([log_density(q,x) for q in f['Q']])
        nll=[float(-logsumexp(posterior(f['scores'],eta)[1]+ll)/50) for eta in [0,.1,.3,1]]
        row=dict(id=p['id'],date=p['date'],domain=p['domain'],nll=nll);write(path,row);rows.append(row);print('F2 VALIDATE',p['id'],flush=True)
    mean=np.mean([r['nll'] for r in rows],axis=0);write(OUT/'f2_temperature.json',dict(eta=[0,.1,.3,1][int(np.argmin(mean))],nll=mean,rows=len(rows)))

def freeze():
    from .information import create_panels
    data=Data();create_panels(data);bench=read(OUT/'benchmark.json');jobs=[]
    def add(stage,p,N,M,**kw):jobs.append(dict(id=f'{stage}_{p["id"]}_N{N}_M{M}'+(f'_o{kw["offset"]}' if 'offset' in kw else ''),stage=stage,panel=p,N=N,M=M,**kw))
    base=[p for p in panels() if p['phase'] in ['test','extension'] or p['date']=='2019-12-31']
    for p in base:
        add('R2',p,50,12)
        if p['seed']==11:
            for N in [20,100]:add('R4-N',p,N,12)
            if p['quarterly']:
                for M in [6,24]:add('R4-M',p,50,M)
    for p in read(PRIVATE/'natural_panels.json'):
        if p['phase'] in ['test','extension']:add('R3b',p,50,12)
    # Conservative p90 extrapolation includes full measured frontend and method grids.
    main=np.quantile([r['cpu_seconds'] for r in bench if r['N']==50 and r['front']=='F1'],.9)
    core_est=1.25*(main*len(jobs)+720*2+90*2+len(read(PRIVATE/'information_panels.json'))*15*main)
    prior=PRIOR_CPU+sum(r['cpu_seconds'] for r in rows(PRIVATE/'runtime_ledger.jsonl'))
    remaining=69120-prior;f2cost=max(r['cpu_seconds'] for r in bench if r['front']=='F2')
    f2panels=[p for p in base if p['seed']==11 and p['quarterly']]
    f2_est=1.25*f2cost*(len(f2panels)+24*2+24)
    enable_f2=core_est+f2_est<remaining
    if enable_f2:
        for p in f2panels:add('F2',p,50,12,front='F2')
        dev=[p for p in panels('development') if p['seed']==11 and p['date'].startswith(('2015','2016')) and p['status']=='ready'][:4]
        for p in dev:
            for N in [20,50]:
                for offset in [0,10000,20000]:add('F2-library',p,N,12,front='F2',offset=offset)
        for p in panels():
            if p['seed']==11 and p['date'][5:7]=='06' and p['phase'] in ['validation','test']:
                for offset in [0,10000,20000]:add('F2-library',p,50,12,front='F2',offset=offset)
    allbench=[r['cpu_seconds'] for r in bench if r['id'].startswith('timingall')];allcost=np.quantile(allbench,.9)
    monthly=1.25*allcost*3*len([p for p in base if p['seed']==11]);quarter=monthly/3
    slack=remaining-core_est-(f2_est if enable_f2 else 0);frequency='monthly' if monthly<slack else 'quarterly' if quarter<slack else 'not_scheduled_budget'
    if (OUT/'full_index_budget_bound.json').exists():frequency='not_scheduled_budget'
    if frequency!='not_scheduled_budget':
        for p in base:
            if p['seed']!=11 or (frequency=='quarterly' and not p['quarterly']):continue
            t=p['t'];ids=data.members.get((p['date'],p['domain']),[]);v=data.a['valid'];n=v[t-566:t+1].sum(0);recent=v[t-62:t+1].sum(0)
            amt=data.a['amount'][t-19:t+1];cnt=np.isfinite(amt).sum(0);adv=np.nansum(amt,axis=0)/np.maximum(cnt,1)
            eligible=[j for j in ids if n[j]>=540 and recent[j]>=60 and cnt[j]>=15 and adv[j]>0 and data.a['trading_status'][t,j]==1]
            q=dict(p,ranking=sorted(eligible,key=lambda j:digest(11,data.symbols[j])),status=p['status'] if p['status']!='ready' else 'ready' if len(eligible)>=5 else 'insufficient_universe')
            for M in [6,12]:add('R4-index',q,len(eligible),M)
    cfg=dict(status='frozen',base_commit=BASE,tau=read(OUT/'tau_calibration.json')['tau'],eta=read(OUT/'validation_selection.json')['eta'],expert_index=read(OUT/'validation_selection.json')['expert_index'],
       eta_R3=read(OUT/'information_temperature.json')['eta'],eta_F2=read(OUT/'f2_temperature.json')['eta'] if enable_f2 else None,
       enable_F2=enable_f2,full_index_frequency=frequency,cost_estimates=dict(core=core_est,F2=f2_est,all_monthly=monthly,remaining_optimization=remaining,p90_main=main,safety_factor=1.25),
       label_stale_main=63,label_sensitivity=[20,126,0],source_hashes=source_hashes(),manifest_hash=digest(json.dumps(clean(jobs),sort_keys=True)),frozen_unix=time.time(),test_results_read=False,
       amendment_sha256=sha(ROOT/'PROTOCOL_AMENDMENT_ZH.md'),parent_protocol_sha256=sha(ROOT.parent/'research_v5_ashare/INPUT_PROTOCOL_ZH.md'),
       source_sha256=read(OUT/'valuation_audit.json')['source_sha256'],membership_hashes=read(ROOT.parent/'research_v5_ashare/output/data_audit.json')['membership_hashes'])
    write(PRIVATE/'manifest.json',jobs);write(ROOT/'protocol_frozen.json',cfg)
    write(OUT/'scheduled_manifest.json',[{k:v for k,v in j.items() if k!='panel'}|{k:j['panel'][k] for k in ['date','domain','seed','phase','status']} for j in jobs])
    print('FROZEN',len(jobs),'F2',enable_f2,'all-index',frequency,'core CPU forecast',round(core_est),flush=True)
