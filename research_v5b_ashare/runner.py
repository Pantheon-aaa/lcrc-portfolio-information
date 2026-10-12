"""Atomic batch runner: shared inputs, method-neutral raw records, fixed calendar."""
import time,traceback
import numpy as np
from core import feasible
from research_v4a.mean_budget import qp
from .common import *
from .risk import frontend
from .pipeline import inp
from .decision import Solver
from .evaluator import label,assess,crossfit_label,assess_crossfit

def methods(k):
    out=[('MA',0.),('EW',0.),('InverseVol',0.),('EMGMV',0.),('LW',0.)]
    out += [('A1B',b) for b in [.1,.25,.5,np.inf]]
    out += [(m,b) for b in [.1,.25] for m in ['Greedy','Hinge','CVaR05','CVaR025']]
    if k==1:out += [('Greedy1swap',b) for b in [.1,.25]]
    return out

def practical(data,panel,front,s,method,expert):
    upper=s.f.upper;N=s.f.d;history=data.returns(panel['t'],front['columns'],252)
    if method=='InverseVol':
        v=1/np.sqrt(np.nanmean(history**2,axis=0));w=feasible(v/v.sum(),np.full(N,upper))
    elif method=='EMGMV':
        if s.f.M==12:w=s.f.oracle[expert]
        else:
            from .risk import em_second,shrink
            L=[126,252,504][expert//4];alpha=[.1,.3,.6,.9][expert%4];S,_=em_second(data.returns(panel['t'],front['columns'],L));w=qp(shrink(S,alpha)/front['u'],upper)['w']
    elif method=='LW':
        if not np.isfinite(history).all():return None
        from sklearn.covariance import LedoitWolf
        q=LedoitWolf(assume_centered=True).fit(history).covariance_;w=qp(q/front['u'],upper)['w']
    else:raise ValueError(method)
    return dict(method=method,beta=0.,input_hash=s.input_hash,shared_cost_id=front['shared_cost_id'],raw_target=w,evaluated_target=w,
        guarded_target=w,raw_status='practical_baseline',fallback_reason=None,**s.metrics(w,np.inf),budget_applies=False,cpu_seconds=0.)

def solve_job(data,job,cfg,folder='decisions',timing=False):
    path=PRIVATE/folder/f'{job["id"]}.json'
    if path.exists():return read(path)
    start=cpu();p=job['panel'];out=dict(id=job['id'],stage=job['stage'],panel_id=p['id'],date=p['date'],phase=p['phase'],domain=p['domain'],seed=p['seed'],
       N=job['N'],M=job['M'],status=p['status'],cells=[])
    if p['status']=='ready' and len(p['ranking'])>=job['N']:
        try:
            lengths=(126,252,504) if job['M']!=24 else (63,126,189,252,378,504)
            strengths=(.1,.6) if job['M']==6 else (.1,.3,.6,.9)
            front=frontend(data,p,job['N'],lengths,strengths,mode=job.get('front','F1'),offset=job.get('offset',0))
            out.update(status='complete',columns=front['columns'],shared_cost_id=front['shared_cost_id'],frontend_cpu=front['metadata']['cpu_seconds'],u=front['u'])
            crossfit=None if timing else crossfit_label(data,p,front['columns'])
            if job['stage']=='R2':tau_indices=[0,1,2]
            else:tau_indices=[1]
            for k in tau_indices:
                s=Solver(inp(front,cfg['tau'][k],cfg['eta_F2'] if job.get('front')=='F2' else cfg['eta']));evaluations={}
                if not timing:
                    for stale in [20,63,126,0]:evaluations[f'H63_stale{stale}']=label(data,p,front['columns'],63,stale)
                    evaluations['H126_stale63']=label(data,p,front['columns'],126,63)
                selected=methods(k)
                if job.get('front')=='F2':selected=[a for a in selected if a[0] not in ['InverseVol','EMGMV','LW']]
                for method,beta in selected:
                    tick=cpu()
                    r=practical(data,p,front,s,method,cfg['expert_index']) if method in ['InverseVol','EMGMV','LW'] else s.solve(method,beta,120 if job['N']>100 else 30)
                    if r is None:
                        out['cells'].append(dict(tau_index=k,method=method,beta=beta,status='unavailable_incomplete_joint_history'));continue
                    scores={name:assess(lab,r['evaluated_target'],cfg['tau'][k]*front['u'])|{'quality':lab.get('quality')} for name,lab in evaluations.items()}
                    if crossfit is not None:scores['H126_crossfit63']=assess_crossfit(crossfit,r['evaluated_target'])
                    out['cells'].append(dict(tau_index=k,method=method,beta=beta,status='complete',decision=r,evaluation=scores,measured_cpu=cpu()-tick))
                    if method=='A1B' and beta in [.1,.25]:
                        guarded=dict(r);guarded['evaluated_target']=r['guarded_target'];guarded['method']='A1B-guarded';guarded.update(s.metrics(r['guarded_target'],beta));guarded['guard_applied']=not np.array_equal(r['guarded_target'],r['evaluated_target'])
                        gs={name:assess(lab,r['guarded_target'],cfg['tau'][k]*front['u'])|{'quality':lab.get('quality')} for name,lab in evaluations.items()}
                        if crossfit is not None:gs['H126_crossfit63']=assess_crossfit(crossfit,r['guarded_target'])
                        out['cells'].append(dict(tau_index=k,method='A1B-guarded',beta=beta,status='complete',decision=guarded,evaluation=gs,measured_cpu=0.))
            out['reference_labels']={key:{j:v for j,v in value.items() if j not in ('Q','oracle')} for key,value in evaluations.items()}
        except Exception:out.update(status='failed',error=traceback.format_exc())
    elif p['status']=='ready':out['status']='insufficient_universe'
    out['cpu_seconds']=cpu()-start;write(path,out);return out

def batch(stage):
    from .data import Data
    cfg=read(ROOT/'protocol_frozen.json')
    if cfg['status']!='frozen':raise ValueError('test gate closed')
    jobs=[j for j in read(PRIVATE/'manifest.json') if j['stage']==stage];data=Data()
    for k,j in enumerate(jobs):
        r=solve_job(data,j,cfg)
        print(stage,k+1,len(jobs),j['id'],r['status'],round(r['cpu_seconds'],2),flush=True)
