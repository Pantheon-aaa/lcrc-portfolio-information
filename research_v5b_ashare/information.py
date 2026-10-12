"""Causal natural-missing panels and equal-scalar-budget information experiments."""
import numpy as np
from scipy.special import logsumexp
from .common import *
from .data import Data
from .risk import log_density
from .bridge import dictionary,view_data,evidence
from .decision import DecisionInput,Solver
from .evaluator import label,assess

def create_panels(data):
    out=[];old={};base=read(PRIVATE/'panels.json')
    for t in data.monthends():
        day=data.dates[t]
        if day<'2014' or day[5:7] not in ['03','06','09','12']:continue
        valid=data.a['valid'];n=valid[max(0,t-251):t+1].sum(0);recent=valid[max(0,t-62):t+1].sum(0)
        amount=data.a['amount'][t-19:t+1];cnt=np.isfinite(amount).sum(0);adv=np.nansum(amount,axis=0)/np.maximum(cnt,1)
        common=(recent>=42)&(cnt>=15)&(adv>0)&(data.a['trading_status'][t]==1)&np.isin(data.a['market_type'][t],[1,4,16,32])
        long=np.flatnonzero(common&(n>=240));short=np.flatnonzero(common&(n>=126)&(n<240))
        for seed in [11,29,47]:
            for group in ['long','mixed']:
                prior=set(old.get((seed,group),[]))
                def rank(ids):
                    liquid=sorted(ids,key=lambda j:(-adv[j],data.symbols[j]))[:200]
                    return sorted(liquid,key=lambda j:(0 if j in prior else 1,digest(seed,data.symbols[j])))
                ranked=rank(long)[:50] if group=='long' else rank(long)[:25]+rank(short)[:25]
                ok=len(ranked)==50
                if ok:old[seed,group]=ranked
                out.append(dict(id=f'{day}_allA_{group}_{seed}',t=t,date=day,domain='allA_'+group,seed=seed,phase=phase(day),status='ready' if ok else 'insufficient_universe',ranking=ranked,quarterly=True))
    write(PRIVATE/'natural_panels.json',out)
    inf=[];prior={}
    for p in base:
        if not p['quarterly'] or p['seed']!=11:continue
        t=p['t'];ids=data.members.get((p['date'],p['domain']),[])
        n=data.a['valid'][t-251:t+1].sum(0);older=data.a['valid'][max(0,t-566):t+1].sum(0)
        amount=data.a['amount'][t-19:t+1];cnt=np.isfinite(amount).sum(0);adv=np.nansum(amount,axis=0)/np.maximum(cnt,1)
        eligible=[j for j in ids if n[j]==252 and older[j]>=540 and cnt[j]>=15 and adv[j]>0 and data.a['trading_status'][t,j]==1]
        pool=sorted(eligible,key=lambda j:(-adv[j],data.symbols[j]))[:200];last=set(prior.get(p['domain'],[]))
        ranked=sorted(pool,key=lambda j:(0 if j in last else 1,digest(11,data.symbols[j])))[:50]
        status='unverified_membership' if p['date'] not in data.valid_dates else 'ready' if len(ranked)==50 else 'insufficient_complete_history'
        if status=='ready':prior[p['domain']]=ranked
        inf.append(dict(p,ranking=ranked,status=status,id=p['id']+'_information'))
    write(PRIVATE/'information_panels.json',inf)

def front(data,p,n,view):
    cols=p['ranking'];t=p['t'];history=data.returns(t-252,cols,252);Q,u,a,b=dictionary(history,cols,data.symbols)
    pairs=data.returns(t,cols,2*n).reshape(n,2,len(cols));observations=view_data(pairs,a,b,view)
    ll=evidence(Q,observations);return Q,u,ll,len(cols),observations

def validate_information():
    data=Data();create_panels(data);rows_out=[];eta_grid=[0,.1,.3,1.]
    for p in read(PRIVATE/'information_panels.json'):
        if p['phase']!='validation' or p['status']!='ready' or phase(data.dates[p['t']+1])!='validation':continue
        for n in [32,64,126]:
            for view in ['Separated0','Separated1','Joint0','Joint1']:
                Q,u,ll,N,x=front(data,p,n,view);nextrow=data.returns(p['t']+1,p['ranking'],1)[0]
                densities=np.array([log_density(q,nextrow) for q in Q]);scores=[]
                for eta in eta_grid:
                    lp=eta*ll/N;lp-=logsumexp(lp);scores.append(float(-logsumexp(lp+densities)/N))
                rows_out.append(dict(id=p['id'],domain=p['domain'],date=p['date'],n=n,view=view,nll=scores,log_likelihood=ll))
        print('R3 VALIDATE',p['id'],flush=True)
    if not rows_out:raise ValueError('no information validation examples')
    # Each date/domain has the same fixed n/view grid. Use only dates with both domains.
    joint=[r for r in rows_out if len(set(s['domain'] for s in rows_out if s['date']==r['date']))==2]
    mean=np.mean([r['nll'] for r in joint],axis=0);eta=eta_grid[int(np.argmin(mean))]
    write(PRIVATE/'information_validation.json',rows_out);write(OUT/'information_temperature.json',dict(eta=eta,grid=eta_grid,nll=mean,rows=len(joint)))

def run():
    cfg=read(ROOT/'protocol_frozen.json');data=Data();jobs=[p for p in read(PRIVATE/'information_panels.json') if p['phase'] in ['test','extension']]
    for p in jobs:
        for n in [32,64,126]:
            for view in ['Separated0','Separated1','Joint0','Joint1','Full']:
                path=PRIVATE/'information'/f'{p["id"]}_{n}_{view}.json'
                if path.exists():continue
                start=cpu();out=dict(id=p['id'],date=p['date'],domain=p['domain'],phase=p['phase'],n=n,view=view,status=p['status'],cells=[])
                if p['status']=='ready':
                    Q,u,ll,N,x=front(data,p,n,view);lp=cfg['eta_R3']*ll/N;lp-=logsumexp(lp);post=np.exp(lp)
                    s=Solver(DecisionInput(Q,post,lp,u,cfg['tau'][1]*u,.1,digest(p['id'],n,view)))
                    lab=label(data,p,p['ranking']);out.update(u=u,log_likelihood=ll,posterior=post,scalar_count=int(np.isfinite(x).sum()),columns=p['ranking'])
                    for method,beta in [('MA',0),('A1B',.1),('Greedy',.1),('A1B',.25),('Greedy',.25)]:
                        r=s.solve(method,beta);out['cells'].append(dict(method=method,beta=beta,decision=r,evaluation=assess(lab,r['evaluated_target'],cfg['tau'][1]*u)))
                out['cpu_seconds']=cpu()-start;write(path,out)
        print('R3',p['id'],flush=True)
