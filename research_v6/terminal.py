import itertools,time,gzip,json
import numpy as np
from scipy.special import logsumexp
from research_v4a.api import solve_budgeted_dcl
from research_v4a.baselines import budget_greedy,budget_surrogate
from research_v4a.price_oracle import PriceCache,Geometry as OldGeometry
from research_v4a.mean_budget import MeanBudget
from research_v4a.budgeted_dcl import enumerate_frontier
from .common import *
from .models import specs,make
from .pipeline import load_family,geometry,setup,save_trajectory,readtraj
from .likelihood import constants
from .sequential import simulate
BETAS=(0,.1,.25,np.inf)
def outputs(f,p,eps,truth):
    geo=OldGeometry(f,eps);pc=PriceCache(MeanBudget(f,p),geo);answers=[]
    for beta in BETAS:
        for method in ('A1-B','Greedy','Hinge','CVaR'):
            start=time.process_time()
            try:
                if method=='A1-B':a=solve_budgeted_dcl(f,p,eps,beta,geo,pc,30)
                elif method=='Greedy':a=budget_greedy(f,p,eps,beta,geo)
                else:a=budget_surrogate(f,p,eps,beta,mode='hinge' if method=='Hinge' else 'tail',rho=.5)
                lo,hi=f.regret_bounds(a['weights']);a.update(regret_lower=lo,regret_upper=hi,success_lower=bool(hi[truth]<=eps),success_upper=bool(lo[truth]<=eps),true_regret=float((lo[truth]+hi[truth])/2))
                a['native_budget_certified']=a['budget_certified']
                a['budget_certified']=bool(np.isinf(beta) or a['mean_increment_upper']<=a['budget_lower'])
                answers.append(dict(method=method,beta=beta,cpu=time.process_time()-start,**a))
            except Exception as e:answers.append(dict(method=method,beta=beta,error=repr(e),cpu=time.process_time()-start))
    return answers
def terminal():
    start=time.process_time();prior=sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl') if r['stage']=='terminal');jobs=[]
    for rep,truth in itertools.product(range(10),range(6)):
      for sp in specs('confirmation'):
       name=f'{sp[0]}-{sp[1]}-6-confirmation-{sp[4]}'
       for premium,allowed in itertools.product((1,4),([1],[0,1,2,3])):
        for n in (0,32,128,512,2048):
         # n=0 is a single prior, evaluated across truths; represented once per family.
         if n==0 and (rep or truth or premium!=1 or allowed!=[1]):continue
         for ratio in (.5,.75,1.25):jobs.append(dict(family=name,rep=rep,truth=truth,premium=premium,allowed=allowed,n=n,ratio=ratio))
    write(ROOT/'terminal_manifest.json',jobs);fs={};cc={};datasets={};processed=0
    for j in jobs:
        if time.process_time()-start+prior>4800:break
        key=f"{j['family']}_r{j['rep']}_t{j['truth']}_c{j['premium']}_a{''.join(map(str,j['allowed']))}_n{j['n']}_e{j['ratio']}";path=RUN/'terminal'/f'{key}.json.gz'
        if path.exists():continue
        name=j['family']
        if name not in fs:fs[name]=load_family(name);cc[name]=constants(fs[name])
        f=fs[name];geo,tab=setup(f,.5,j['premium'],j['allowed']);ds=(name,j['truth'],j['rep'],j['premium'],tuple(j['allowed']))
        if j['n']==0:ll=np.zeros(f.M)
        else:
            if ds not in datasets:
                r=simulate(f,geo,tab,cc[name],j['truth'],j['rep'],.5,.05,j['premium'],j['allowed'],'equal_cost',horizon=2048,checkpoints=(0,32,128,512,2048),stream_namespace='confirmation')
                datasets[ds]=r['snapshots'];save_trajectory(RUN/'terminal_streams'/f'{seed(*ds)}.json.gz',r)
            if j['n'] not in datasets[ds]:
                write(RUN/'failures'/f'{key}.json',dict(job=j,error='checkpoint unavailable before cost/action cap'));continue
            ll=np.array(datasets[ds][j['n']]['log_likelihood'])
        p=np.exp(ll-logsumexp(ll));eps=j['ratio']*(geo[63]['lower']+geo[63]['upper'])/2
        a=outputs(f,p,eps,j['truth']);record=dict(job=j,posterior=p,log_likelihood=ll,epsilon=eps,answers=a,zero_prior_truths=list(range(6)) if j['n']==0 else None)
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path.with_suffix('.tmp'),'wt',encoding='utf8') as z:json.dump(clean(record),z,separators=(',',':'))
        path.with_suffix('.tmp').replace(path);processed+=1
        if processed%20==0:print('TERMINAL',processed,'CPU',round(time.process_time()-start),flush=True)
        # Retain only current family/truth/repeat stream set, bounded memory.
        if len(datasets)>60:datasets={ds:datasets[ds]} if ds in datasets else {}
    write(OUT/'terminal_completion.json',dict(planned=len(jobs),processed=len(list((RUN/'terminal').glob('*.json.gz'))),new=processed))
def stress():
    start=time.process_time();prior=sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl') if r['stage']=='stress')
    out=read(OUT/'S3_stress.json') if (OUT/'S3_stress.json').exists() else []
    done={(r['kind'],r['d'],r['M'],str(r['beta']),r['method']) for r in out}
    for d,M,kind in itertools.product((10,20),(6,12),KINDS):
        if prior+time.process_time()-start>900:break
        f=make(kind,d,M,'stress',0);g=__import__('research_v6.certificates',fromlist=['Geometry']).Geometry(f);r=g.gamma((1<<M)-1);eps=.5*(r['lower']+r['upper'])/2
        con=constants(f);from .likelihood import GaussianStream
        stream=GaussianStream(f,0,seed('stress',d,M,kind),con);ll,_=stream.chunk(3,32);p=np.exp(ll.sum(0)-logsumexp(ll.sum(0)));gg=OldGeometry(f,eps);pc=PriceCache(MeanBudget(f,p),gg)
        enum=None
        if M==6 and not (OUT/'frontiers'/f'{kind}-{d}-{M}.json').exists():
            if prior+time.process_time()-start>929:break
            tic=time.process_time()
            try:
                enum,prices=enumerate_frontier(f,p,eps,BETAS,30)
                write(OUT/'frontiers'/f'{kind}-{d}-{M}.json',dict(prices=[dict(subset=G,lower=a.cost_lower,upper=a.cost_upper,epsilon_feasible=a.epsilon_feasible,status=a.status) for G,a in prices.cache.items()],results=enum,cpu=time.process_time()-tic))
            except Exception as e:write(OUT/'frontiers'/f'{kind}-{d}-{M}.json',dict(error=repr(e),cpu=time.process_time()-tic))
        for beta in BETAS:
            for method in ('cold','cached','greedy'):
                if (kind,d,M,str(beta),method) in done:continue
                # Keep the same 30-second query limit and reserve its full allowance.
                if prior+time.process_time()-start>929:
                    write(OUT/'S3_stress.json',out);return
                tic=time.process_time()
                try:
                    a=solve_budgeted_dcl(f,p,eps,beta,gg,pc,30) if method=='cached' else solve_budgeted_dcl(f,p,eps,beta,time_limit_seconds=30) if method=='cold' else budget_greedy(f,p,eps,beta)
                    out.append(dict(kind=kind,d=d,M=M,beta=beta,method=method,cpu=time.process_time()-tic,**a))
                except Exception as e:out.append(dict(kind=kind,d=d,M=M,beta=beta,method=method,error=repr(e),cpu=time.process_time()-tic))
        write(OUT/'S3_stress.json',out);print('STRESS',kind,d,M,flush=True)
def dispatch(stage,*args):{'terminal':terminal,'stress':stress}[stage](*args)
