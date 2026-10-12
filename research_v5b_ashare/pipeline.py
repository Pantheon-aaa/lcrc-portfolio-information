"""Resumable stage orchestration; development results cannot unlock test grids implicitly."""
import time,traceback
import numpy as np
from scipy.special import logsumexp
from .common import *
from .data import Data
from .risk import frontend,posterior,log_density
from .decision import DecisionInput,Solver
from .evaluator import label,assess

def panels(which=None):
    a=read(PRIVATE/'panels.json')
    return a if which is None else [r for r in a if r['phase']==which]

def gate0():
    from .checks import run
    run()

def prepare():
    from .valuation import build
    build(Data())

def benchmark():
    from .setup import benchmark as run
    run()

def validate_information():
    from .information import validate_information as run
    run()

def validate_f2():
    from .setup import validate_f2 as run
    run()

def freeze():
    from .setup import freeze as run
    run()

def r1():
    from .bridge import r1 as run
    run()

def semisynthetic():
    from .bridge import semisynthetic as run
    run()

def batch(stage):
    from .runner import batch as run
    run(stage)

def information():
    from .information import run
    run()

def evaluate():
    from .proxy_book import run
    run()

def label_audit():
    data=Data();details=[];counts={};groups={};total=0
    for panel in panels('development'):
        if panel['status']!='ready' or panel['t']+64>=len(data.dates) or phase(data.dates[panel['t']+64])!='development':continue
        cols=panel['ranking'][:50];rr=np.arange(panel['t']+2,panel['t']+65);valid=np.array(data.a['valid'][np.ix_(rr,cols)])
        for i,j in zip(*np.where(~valid)):
            day=rr[i];col=cols[j];values={k:float(data.a[k][day,col]) for k in ['return_total','observed','total_adjustment_suspect','trading_status','close','open','return_ex_dividend']}
            reason='missing_reported_return' if not np.isfinite(values['return_total']) else 'adjustment_suspect' if values['total_adjustment_suspect']==1 else 'not_observed'
            counts[reason]=counts.get(reason,0)+1;total+=1
            details.append(dict(panel_id=panel['id'],date=data.dates[day],symbol=data.symbols[col],reason=reason,**values))
        key=f'{panel["domain"]}_{panel["seed"]}';groups.setdefault(key,dict(instances=0,complete=0,raw_reported_complete=0,missing=0,flagged=0));g=groups[key]
        g['instances']+=1;g['complete']+=int(valid.all());g['raw_reported_complete']+=int(np.isfinite(data.a['return_total'][np.ix_(rr,cols)]).all())
        g['missing']+=int((~np.isfinite(data.a['return_total'][np.ix_(rr,cols)])).sum());g['flagged']+=int((data.a['total_adjustment_suspect'][np.ix_(rr,cols)]==1).sum())
    write(PRIVATE/'label_missing_details.json',details)
    statuscounts={}
    for row in details:
        key=str(row['trading_status']);statuscounts[key]=statuscounts.get(key,0)+1
    write(OUT/'label_data_audit.json',dict(groups=groups,invalid_entries_with_panel_overlap=total,invalid_reasons=counts,historical_status_on_invalid_entries=statuscounts,
      conclusion='Unknown returns cannot be zero-filled or identified as suspensions from missingness alone. No source mutation or future survivor filtering is applied.',
      detailed_private_file='label_missing_details.json'))
    print('LABEL AUDIT',groups,counts,flush=True)

def inp(front,tau,eta):
    p,lp=posterior(front['scores'],eta)
    return DecisionInput(front['Q'],p,lp,front['u'],tau*front['u'],min(.2,5/front['Q'].shape[1]),front['shared_cost_id'])

def timing():
    data=Data();selected=[]
    for domain in ['csi300','csi500']:
        for year in ['2015','2016']:
            selected.extend([r for r in panels('development') if r['domain']==domain and r['seed']==11 and r['date'].startswith(year) and r['status']=='ready'][:5])
    jobs=[dict(panel=r,N=20,M=6) for r in selected]+[dict(panel=r,N=50,M=12) for r in selected[:4]]
    write(PRIVATE/'timing_manifest.json',jobs)
    for k,job in enumerate(jobs):
        path=PRIVATE/'timing'/f'{k:02d}.json'
        if path.exists():continue
        start=cpu();panel=job['panel']
        try:
            f=frontend(data,panel,job['N'],strengths=(.1,.6) if job['M']==6 else (.1,.3,.6,.9))
            s=Solver(inp(f,.1,0));results=[s.solve('A1B',b) for b in [0,.1,.25,np.inf]]
            out=dict(**job,status='complete',shared_cost_id=f['shared_cost_id'],frontend_cpu=f['metadata']['cpu_seconds'],solver_prepare_cpu=s.prepare_cpu,results=results,cpu_seconds=cpu()-start)
        except Exception:out=dict(**job,status='failed',error=traceback.format_exc(),cpu_seconds=cpu()-start)
        write(path,out);print('TIMING',k+1,len(jobs),panel['id'],job['N'],round(out['cpu_seconds'],2),out['status'],flush=True)
    write(OUT/'timing_summary.json',[{k:v for k,v in read(p).items() if k not in ('panel','results')} for p in sorted((PRIVATE/'timing').glob('*.json'))])

def calibrate():
    data=Data();jobs=panels('development');results=[]
    for k,panel in enumerate(jobs):
        path=PRIVATE/'calibration'/f'{panel["id"]}.json'
        if path.exists():results.append(read(path));continue
        row=dict(id=panel['id'],date=panel['date'],domain=panel['domain'],seed=panel['seed'])
        try:
            if panel['status']!='ready':row['status']=panel['status']
            else:
                # Uniform MA requires current Q only. Prequential scores cannot alter tau calibration.
                f=frontend(data,panel,score=False);s=Solver(inp(f,0,0));lab=label(data,panel,f['columns'])
                row.update(status=lab['status'],shared_cost_id=f['shared_cost_id'],label_end=lab.get('label_end'))
                if lab['status']=='complete':
                    a=assess(lab,s.mb.w,0);row.update(ratio_lower=a['regret_lower']/f['u'],ratio_upper=a['regret_upper']/f['u'],ma=s.mb.w,oracle=lab['oracle'])
        except Exception:row.update(status='failed',error=traceback.format_exc())
        write(path,row);results.append(row)
        if (k+1)%12==0:print('CALIBRATION',k+1,len(jobs),flush=True)
    usable=[r for r in results if r['status']=='complete']
    if not usable:raise ValueError('No complete development labels; freezing prohibited')
    # Equal index/panel weights, then equal dates within each cell. No test data here.
    groupcounts={(d,s):sum(r['domain']==d and r['seed']==s for r in usable) for d in ['csi300','csi500'] for s in [11,29,47]}
    if min(groupcounts.values())==0:
        write(OUT/'tau_calibration.json',dict(status='blocked_insufficient_complete_development_labels',tau=None,rows=len(results),complete=len(usable),
          counts=[dict(domain=k[0],seed=k[1],n=v) for k,v in groupcounts.items()],
          statuses={s:sum(r['status']==s for r in results) for s in sorted(set(r['status'] for r in results))},
          calibration_rows_hash=digest(json.dumps(clean(results),sort_keys=True)),
          rule='Protocol section 7.2/7.3: all six equally weighted index/panel groups must have complete development labels. No test labels may substitute.'))
        print('FREEZE BLOCKED: four of six calibration groups have no complete development labels',flush=True)
        return
    values=np.array([(r['ratio_lower']+r['ratio_upper'])/2 for r in usable]);weights=np.array([1/(6*groupcounts[r['domain'],r['seed']]) for r in usable]);order=np.argsort(values)
    cdf=np.cumsum(weights[order]);tau=[float(values[order[min(np.searchsorted(cdf,q),len(order)-1)]]) for q in [.3,.5,.7]]
    write(OUT/'tau_calibration.json',dict(tau=tau,quantile_rule='left inverse weighted empirical CDF; equal index/panel, equal available complete development dates within group',
      rows=len(results),complete=len(usable),unresolved=len(results)-len(usable),calibration_rows_hash=digest(json.dumps(clean(results),sort_keys=True)),
      max_oracle_ratio_width=max(r['ratio_upper']-r['ratio_lower'] for r in usable),counts=[dict(domain=k[0],seed=k[1],n=v) for k,v in groupcounts.items()],
      statuses={s:sum(r['status']==s for r in results) for s in sorted(set(r['status'] for r in results))}))
    print('TAU',tau,'complete',len(usable),'of',len(results),flush=True)

def validate():
    data=Data();jobs=panels('validation');results=[];etas=[0,.1,.3,1.]
    for k,panel in enumerate(jobs):
        path=PRIVATE/'validation'/f'{panel["id"]}.json'
        if path.exists():results.append(read(path));continue
        row=dict(id=panel['id'],date=panel['date'],domain=panel['domain'],seed=panel['seed'])
        try:
            if panel['status']!='ready':row['status']=panel['status']
            elif phase(data.dates[panel['t']+1])!='validation':row['status']='cross_phase_next_day'
            else:
                f=frontend(data,panel);x=data.returns(panel['t']+1,f['columns'],1)[0]
                if not np.isfinite(x).any():row['status']='empty_next_day'
                else:
                    ll=np.array([log_density(q,x) for q in f['Q']]);nll=[]
                    for eta in etas:
                        p,lp=posterior(f['scores'],eta);nll.append(float(-logsumexp(lp+ll)/50))
                    lab=label(data,panel,f['columns']);loss=[float(.5*w@lab['Q']@w) for w in Solver(inp(f,0,0)).f.oracle] if lab['status']=='complete' else None
                    row.update(status='complete',shared_cost_id=f['shared_cost_id'],nll=nll,next_day_log_likelihood=ll,expert_future_loss=loss,expert_label_status=lab['status'])
        except Exception:row.update(status='failed',error=traceback.format_exc())
        write(path,row);results.append(row)
        print('VALIDATION',k+1,len(jobs),panel['id'],row['status'],flush=True)
    good=[r for r in results if r['status']=='complete']
    if not good:raise ValueError('No validation scores')
    def balanced(field):
        monthly=[]
        for day in sorted(set(r['date'] for r in good)):
            domains=[]
            for domain in ['csi300','csi500']:
                rr=[r[field] for r in good if r['date']==day and r['domain']==domain and r[field] is not None]
                if len(rr)==3:domains.append(np.mean(rr,axis=0))
            if len(domains)==2:monthly.append(np.mean(domains,axis=0))
        return np.mean(monthly,axis=0),len(monthly)
    scores,count=balanced('nll');expert_scores,expert_count=balanced('expert_future_loss')
    eta=etas[int(np.argmin(scores))];expert=int(np.argmin(expert_scores))
    write(OUT/'validation_selection.json',dict(eta=eta,eta_grid=etas,nll=scores,complete_months=count,expert_index=expert,expert_scores=expert_scores,expert_complete_months=expert_count,
      rows=len(results),complete=len(good),row_hash=digest(json.dumps(clean(results),sort_keys=True))))
    print('ETA',eta,'EM-GMV expert',expert,flush=True)

def r1_geometry():
    """Uniform-weight geometric diagnostic only if empirical tau cannot be frozen."""
    from itertools import combinations
    from research_v4a.baselines import cvx_price
    from research_v4a.price_oracle import Geometry,PriceCache
    data=Data();jobs=[p for p in panels('development') if p['seed']==11 and p['quarterly'] and p['date'][:4] in ['2015','2016']][:16]
    write(PRIVATE/'r1_geometry_manifest.json',jobs)
    for panel in jobs:
        if panel['status']!='ready':
            write(PRIVATE/'r1_geometry'/f'{panel["id"]}_unavailable.json',dict(id=panel['id'],status=panel['status']));continue
        f=frontend(data,panel,N=20,strengths=(.1,.6),score=False);s0=Solver(inp(f,0,0));g=Geometry(s0.f,0).gamma(range(6));gamma=(g.lower+g.upper)/2
        for ratio in [.5,.75,1.05]:
            path=PRIVATE/'r1_geometry'/f'{panel["id"]}_{ratio}.json'
            if path.exists():continue
            start=cpu();s=Solver(inp(f,ratio*gamma,0));entries=[]
            for size in range(7):
                for G in combinations(range(6),size):
                    try:
                        price=s.pc.get(G,time.perf_counter()+30,tight=True)
                        cvx=cvx_price(s.f,s.mb.p,G,s.eps)
                        wc=cvx['weights'];res=None if wc is None else max(abs(wc.sum()-1),-wc.min(),wc.max()-s.f.upper,0.)
                        eps_res=None if wc is None or not G else float(np.max(s.f.regret_bounds(wc)[1][list(G)]-s.eps))
                        entries.append(dict(subset=G,mass=len(G)/6,price_lower=price.cost_lower,price_upper=price.cost_upper,epsilon_feasible=price.epsilon_feasible,
                          status=price.status,weights=price.witness_weights,cvx_status=cvx['status'],cvx_price=cvx['value'],cvx_weights=wc,cvx_primal_residual=res,cvx_epsilon_residual=eps_res))
                    except Exception:entries.append(dict(subset=G,status='failed',error=traceback.format_exc()))
            solutions=[s.solve(method,b) for b in [0,.1,.25,.5,np.inf] for method in ['A1B','Greedy']]
            out=dict(id=panel['id'],date=panel['date'],domain=panel['domain'],status='complete',role='uniform_weight_geometry_only_not_frozen_R1_primary',N=20,M=6,
              epsilon_gamma_ratio=ratio,epsilon_over_u=ratio*gamma,gamma_lower_over_u=g.lower,gamma_upper_over_u=g.upper,
              entries=entries,solutions=solutions,shared_cost_id=f['shared_cost_id'],cpu_seconds=cpu()-start)
            write(path,out);print('R1 GEOMETRY',panel['id'],ratio,'CPU',round(out['cpu_seconds'],2),flush=True)

def n100_attempt():
    data=Data();jobs=read(PRIVATE/'timing_manifest.json')[:4]
    for k,job in enumerate(jobs):
        panel=job['panel'];path=PRIVATE/'n100_attempt'/f'{k}.json'
        if path.exists():continue
        start=cpu()
        try:
            f=frontend(data,panel,N=100,score=False);s=Solver(inp(f,.1,0));r=s.solve('A1B',.1,120)
            out=dict(status='complete',id=panel['id'],N=100,M=12,role='development_timing_at_fixed_epsilon_over_u_0.1_not_primary_tau',
              result=r,shared_cost_id=f['shared_cost_id'],frontend_cpu=f['metadata']['cpu_seconds'],prepare_cpu=s.prepare_cpu,cpu_seconds=cpu()-start)
        except Exception:out=dict(status='failed',id=panel['id'],error=traceback.format_exc(),cpu_seconds=cpu()-start)
        write(path,out);print('N100',k+1,out['status'],round(out['cpu_seconds'],2),flush=True)

def report():
    from .reporting import report as run
    run()

def archive():
    from .reporting import archive as run
    run()

def delivery_check():
    from .reporting import delivery_check as run
    run()
