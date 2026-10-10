"""Frozen finite-model experiment runner; truth enters only simulation/evaluation."""
import json,time,sys,platform
from itertools import combinations
import numpy as np
from scipy.special import logsumexp,expit
from scipy.stats import beta
from .common import OUT,CFG,write,append,rows,seed,timed_stage,source_hash
from .compat_models import e1,geometry,multidirection,Family,information_rows,draw_loglik,kl_matrix
from .compatible_qcqp import Compatibility,solve
from .dcl_master import DCL,greedy
from .evidence_gate import prepare,approve,scalar_decisions
from .baselines import common,evaluate

PAIRS=[(3,3),(3,6),(6,6),(6,10),(10,10),(10,12)]
KINDS=['simplex','asymmetric','hidden_curvature']
RATIOS=[.25,.5,.75,1.]
NATS=[0,1,3,6]

def require_gates():
    if not (OUT/'gate0.json').exists() or not json.loads((OUT/'gate0.json').read_text())['passed']:raise RuntimeError('Gate 0 is required')

def freeze(budget):
    require_gates();timings=[]
    for kind,d,M in [('simplex',3,3),('asymmetric',6,10),('hidden_curvature',10,12)]:
        f=geometry(kind,d,M,'timing');c=Compatibility(f);ga=c.gamma(range(M));epsilon=.5*ga.upper
        masters={r:DCL(c,r*(ga.lower+ga.upper)/2) for r in RATIOS}
        for rep in range(5):
            cpu=time.process_time();ll=draw_loglik(f,rep%M,50,32,np.random.default_rng(seed('timing-data',kind,rep)))
            p,bs=common(f,c,ll)
            for ratio,master in masters.items():
                master.solve(p);solve(f,p,'hinge',epsilon=master.epsilon);greedy(c,p,master.epsilon)
            timings.append({'kind':kind,'d':d,'M':M,'rep':rep,'cpu_seconds':time.process_time()-cpu})
    # Conservative max timing, all cells preserved. Decisions never use performance outcomes.
    cost=max(x['cpu_seconds'] for x in timings)*1.5
    cells=16*len(NATS)*2
    allowed=[n for n in (50,100,200) if cost*cells*n<6300*.85]
    repetitions=max(allowed) if allowed else 50
    f=multidirection(20);c=Compatibility(f);eps=.5*c.gamma(range(f.M)).upper;times4=[]
    for rep in range(4):
        cpu=time.process_time();pilot=draw_loglik(f,rep,16,32,np.random.default_rng(seed('timing4',rep)))
        library=prepare(f,pilot,c,eps);gate=draw_loglik(f,rep,20,0,np.random.default_rng(seed('timing4gate',rep)))
        approve(f,library,gate);common(f,c,pilot+gate);common(f,c,pilot);times4.append(time.process_time()-cpu)
    cost4=max(times4)*1.5;allowed4=[n for n in (50,100,200) if cost4*8*n<5400*.8]
    n4=max(allowed4) if allowed4 else 50
    config=dict(schema=1,baseline_commit='5f3bf8c45b8e0e3d8fa706c57006e45e01e6f147',seed_namespace='v3-frozen-20261010',
      pairs=PAIRS,kinds=KINDS,ratios=RATIOS,information_nats=NATS,singleton_per_asset=[32,256],
      e1_repetitions=200,e2_repetitions=repetitions,e3_repetitions=200,e4_repetitions=n4,
      confirmation_repetitions=50,e4_models=8,e4_pilot_joint=16,e4_t=.1,delta=.05,
      e3_t=[.1,.03,.01],e3_dimensions=[10,20],e3_multipliers=[0,.25,.5,1,2,4,8],
      master_timeout=30,max_joint_rows=100000,cpu_limits=dict(gate_e1=2700,E2=6300,E3=3600,E4=5400,confirmation=3600),
      freeze_hash=source_hash(),timing_max_seconds=cost/1.5,timing4_max_seconds=cost4/1.5,
      primary_comparisons=['A1 vs MA true success','A1 vs CVaR true success','B2 vs LCRC harmful change and improvement','B2 vs pilot MA harmful change and improvement'],
      note='Equal-mass finite prior. Parameter t is public family index. Numerical boundary ratio=1 excluded from rankings.')
    write(CFG/'protocol.json',config,frozen=True);write(OUT/'timing.json',{'E2':timings,'E4':times4})
    print(f'Frozen E2={repetitions}, E4={n4}, confirmation=50 repetitions per cell',flush=True)

def config():return json.loads((CFG/'protocol.json').read_text(encoding='utf-8'))

def record(path,key,method,f,p,epsilon,truth,result=None,detail=None,w=None):
    detail={} if detail is None else detail
    if result is not None:
        w=result.w;detail={**detail,'solver_status':result.status,'objective_lower':result.lower,'objective_upper':result.upper,
          'optimization_gap':result.gap,'solve_seconds':result.seconds}
    append(path,{**key,'method':method,**evaluate(f,w,p,epsilon,truth),**detail})

def A_data(path,key,f,c,masters,ll,truth,extra_pair=False):
    p,base=common(f,c,ll);ma_lower=float(p@f.regret_bounds(base['MA'].w)[0])
    for ratio,master in masters.items():
        epsilon=master.epsilon;k={**key,'epsilon_ratio':ratio,'epsilon':epsilon}
        for name,r in base.items():record(path,k,name,f,p,epsilon,truth,result=r)
        for name,fn in [('A1',lambda:master.solve(p)),('Hinge',lambda:solve(f,p,'hinge',epsilon=epsilon)),('Greedy',lambda:greedy(c,p,epsilon))]:
            try:
                r=fn()
                if name=='A1':
                    w=r.pop('w');r.pop('cut_sets',None);record(path,k,name,f,p,epsilon,truth,w=w,detail=r)
                else:w=r.w;record(path,k,name,f,p,epsilon,truth,result=r)
                assert float(p@f.regret_bounds(w)[1])>=ma_lower-1e-9,'MA optimality violated'
            except Exception as exc:append(path,{**k,'method':name,'status':'failed','error':repr(exc)})
        if extra_pair:
            pair=list(combinations(range(3),2))[seed('pair-randomization',key['dataset'])%3]
            record(path,k,'RandomPair',f,p,epsilon,truth,result=c.gamma(pair),detail={'pair':pair})

def run_e1(budget):
    require_gates();conf=config();path=OUT/'E1.jsonl';done={r['dataset'] for r in rows(OUT/'E1_completed.jsonl')}
    for t in (.005,.01,.02):
        f=e1(t);c=Compatibility(f);epsilon=7*t*t/16;master=DCL(c,epsilon)
        for nats in NATS:
            m,k,actual=information_rows(f,nats)
            for n in (32,256):
                for rep in range(conf['e1_repetitions']):
                    budget.check();truth=rep%3;dataset=f'E1-{t}-{nats}-{n}-{rep}'
                    if dataset in done:continue
                    ll=draw_loglik(f,truth,m,n,np.random.default_rng(seed(conf['seed_namespace'],dataset)))
                    key=dict(dataset=dataset,experiment='E1',split='mechanism',family=f.name,d=3,M=3,t=t,nats=nats,joint_rows=m,
                      actual_information=actual,singleton_per_asset=n,rep=rep,truth=truth)
                    A_data(path,key,f,c,{'fixed':master},ll,truth,True);append(OUT/'E1_completed.jsonl',{'dataset':dataset})
        print(f'E1 t={t} complete; CPU={time.process_time()-budget.cpu:.1f}s',flush=True)

def run_e2(budget,split='development'):
    require_gates();conf=config();nrep=conf['e2_repetitions'] if split=='development' else conf['confirmation_repetitions']
    path=OUT/f'E2_{split}.jsonl';completed=OUT/f'E2_{split}_completed.jsonl';done={r['dataset'] for r in rows(completed)}
    for kind in KINDS:
        for d,M in PAIRS:
            if kind=='hidden_curvature' and d==3:continue
            f=geometry(kind,d,M,split);c=Compatibility(f);ga=c.gamma(range(M));gamma=(ga.lower+ga.upper)/2
            masters={r:DCL(c,r*gamma) for r in RATIOS}
            pair_states=[c.classify(pair,.75*gamma)[0] for pair in combinations(range(M),2)]
            for nats in NATS:
                m,k,actual=information_rows(f,nats)
                for n in (32,256):
                    for rep in range(nrep):
                        budget.check();dataset=f'E2-{split}-{kind}-{d}-{M}-{nats}-{n}-{rep}'
                        if dataset in done:continue
                        # Balanced truth strata, independent data within each stratum.
                        truth=rep%M;ll=draw_loglik(f,truth,m,n,np.random.default_rng(seed(conf['seed_namespace'],dataset)))
                        key=dict(dataset=dataset,experiment='E2',split=split,kind=kind,family=f.name,d=d,M=M,nats=nats,joint_rows=m,
                          actual_information=actual,information_target_met=bool(actual>=nats-1e-10),singleton_per_asset=n,
                          rep=rep,truth=truth,prior='uniform',gamma_lower=ga.lower,gamma_upper=ga.upper)
                        A_data(path,key,f,c,masters,ll,truth);append(completed,{'dataset':dataset})
            write(OUT/'geometry'/f'{f.name}.json',dict(name=f.name,Q=f.Q,oracle=f.oracle,oracle_diagnostics=f.oracle_diagnostics,
              gamma=dict(lower=ga.lower,upper=ga.upper),pairwise_compatible_fraction_at_075=pair_states.count('compatible')/len(pair_states),
              cuts={str(r):ma.cuts for r,ma in masters.items()},cached_subproblems=len(c.cache)))
            print(f'E2 {split} {kind} d={d} M={M} complete; CPU={time.process_time()-budget.cpu:.1f}s',flush=True)

def run_e3(budget):
    require_gates();conf=config();path=OUT/'E3.jsonl';completed=OUT/'E3_completed.jsonl';done={r['dataset'] for r in rows(completed)}
    for d in conf['e3_dimensions']:
        for t in conf['e3_t']:
            f=Family(np.array([[[1,1-t],[1-t,4]],[[1,1.7],[1.7,4]]]))
            KL=kl_matrix(f)[0,1];threshold=np.log(1/(conf['delta']*t));xG=t/(3+2*t)
            for mult in conf['e3_multipliers']:
                m=int(np.ceil(mult*threshold/KL))
                for truth in (0,1):
                    for rep in range(conf['e3_repetitions']):
                        budget.check();dataset=f'E3-{d}-{t}-{mult}-{truth}-{rep}'
                        if dataset in done:continue
                        ll=draw_loglik(f,truth,m,0,np.random.default_rng(seed(conf['seed_namespace'],dataset)))
                        pg=expit(ll[0]-ll[1]);answers,regret=scalar_decisions(pg,t,ll,conf['delta'])
                        answers['Oracle']=xG if truth==0 else 0.
                        k=d//2
                        for name,x in answers.items():
                            r=regret(x);weights=np.r_[np.full(k,(1-x)/k),np.full(k,x/k)]
                            append(path,dict(dataset=dataset,experiment='E3',split='mechanism',method=name,d=d,t=t,truth=truth,rep=rep,
                              multiplier=mult,joint_rows=m,KL=KL,normalized_information=m*KL/np.log(1/t),logLR=float(ll[0]-ll[1]),
                              posterior_good=pg,group2_weight=x,weights=weights,regret=float(r[truth]),regret_over_t2=float(r[truth]/t**2),
                              positive_entry=bool(x>1e-12),meaningful_unlock=bool(x>=xG/2),B1_approval=bool(name=='B1' and x>0),
                              false_unlock=bool(truth==1 and x>=xG/2),false_reject=bool(truth==0 and x<xG/2),target_error=conf['delta']*t))
                        append(completed,{'dataset':dataset})
            print(f'E3 d={d} t={t} complete; CPU={time.process_time()-budget.cpu:.1f}s',flush=True)

def run_e4(budget,split='development'):
    require_gates();conf=config()
    if not (OUT/'E3_completed.jsonl').exists() or not (OUT/'E2_development_completed.jsonl').exists():raise RuntimeError('E2 and E3 prerequisite missing')
    nrep=conf['e4_repetitions'] if split=='development' else conf['confirmation_repetitions'];path=OUT/f'E4_{split}.jsonl'
    completed=OUT/f'E4_{split}_completed.jsonl';done={r['dataset'] for r in rows(completed)}
    for d in (10,20):
        f=multidirection(d,conf['e4_models'],split);c=Compatibility(f);ga=c.gamma(range(f.M));epsilon=.5*(ga.lower+ga.upper)
        master=DCL(c,epsilon)
        for nats in NATS:
            m,kl,actual=information_rows(f,nats)
            for rep in range(nrep):
                budget.check();truth=rep%f.M;dataset=f'E4-{split}-{d}-{nats}-{rep}'
                if dataset in done:continue
                pilot=draw_loglik(f,truth,conf['e4_pilot_joint'],32,np.random.default_rng(seed(conf['seed_namespace'],dataset,'pilot')))
                library=prepare(f,pilot,c,epsilon)
                gate=draw_loglik(f,truth,m,0,np.random.default_rng(seed(conf['seed_namespace'],dataset,'gate')))
                answer=approve(f,library,gate,conf['e4_t'],conf['delta']);p,bs=common(f,c,pilot+gate);_,pilot_bs=common(f,c,pilot)
                key=dict(dataset=dataset,experiment='E4',split=split,family=f.name,d=d,M=f.M,nats=nats,joint_rows=m,
                  pilot_joint_rows=conf['e4_pilot_joint'],actual_information=actual,rep=rep,truth=truth,epsilon=epsilon,epsilon_ratio=1.)
                baseline=float(f.costs(library.baseline)[truth]);pad=1e-11*(1+abs(baseline))
                def safety(w):
                    change=float(f.costs(w)[truth]-baseline)
                    return {'true_change_from_pilot':change,'harmful_change':bool(change>pad),'beneficial_change':bool(change < -pad)}
                for name,r in bs.items():record(path,key,name+'-all',f,p,epsilon,truth,result=r,detail=safety(r.w))
                for name,r in pilot_bs.items():record(path,key,name+'-pilot',f,p,epsilon,truth,result=r,detail=safety(r.w))
                for name,r in [('Hinge-all',solve(f,p,'hinge',epsilon=epsilon))]:record(path,key,name,f,p,epsilon,truth,result=r,detail=safety(r.w))
                a=master.solve(p);record(path,key,'A1-all',f,p,epsilon,truth,w=a['w'],detail={**safety(a['w']),'status':a['status'],'mass_gap':a['mass_upper']-a['mass_lower']})
                harmful_approval=any(f.costs(library.weights[z])[truth]-baseline>pad for z in answer['approved'])
                detail={**safety(answer['w']),'candidate_count':len(library.weights),'approved_count':len(answer['approved']),
                  'harmful_approval':bool(harmful_approval),'eligible_multidirection':len(library.weights)>=2,
                  'optimization_gap':answer['objective_gap'],'target_error':conf['delta']*conf['e4_t']}
                record(path,key,'B2',f,p,epsilon,truth,w=answer['w'],detail=detail)
                append(OUT/f'E4_{split}_evidence.jsonl',{**key,'baseline':library.baseline,'candidate_weights':library.weights,
                  'candidate_labels':library.labels,'generation':library.generation,'log_mixtures':library.log_mixtures,
                  'pilot_loglik':pilot,'gate_loglik':gate,**answer})
                append(completed,{'dataset':dataset})
        write(OUT/'geometry'/f'{f.name}.json',{'Q':f.Q,'oracle':f.oracle,'gamma_lower':ga.lower,'gamma_upper':ga.upper})
        print(f'E4 {split} d={d} complete; CPU={time.process_time()-budget.cpu:.1f}s',flush=True)

def rare(budget):
    conf=config();path=OUT/'E3_rare.jsonl'
    done={(r['t'],r['multiplier']) for r in rows(path)}
    for t in (.01,.03,.1):
        f=Family(np.array([[[1,1-t],[1-t,4]],[[1,1.7],[1.7,4]]]))
        KL=kl_matrix(f)[0,1]
        for mult in (.5,1.,2.):
            if (t,mult) in done:continue
            m=int(np.ceil(mult*np.log(1/(conf['delta']*t))/KL));count=0;n=10000
            from .evidence_gate import two_model_action
            for rep in range(n):
                if rep%500==0:budget.check()
                ll=draw_loglik(f,1,m,0,np.random.default_rng(seed(conf['seed_namespace'],'rare',t,mult,rep)))
                count+=two_model_action(ll,t)[1]
            # Bonferroni simultaneous one-sided interval for nine fixed audits.
            upper=float(beta.ppf(1-.05/9,count+1,n-count)) if count<n else 1.
            append(path,dict(t=t,multiplier=mult,m=m,n=n,errors=count,rate=count/n,simultaneous_95_upper=upper,target=.05*t))
    print('Rare-event calibration complete',flush=True)

def confirmation(budget):
    run_e2(budget,'confirmation');run_e4(budget,'confirmation');rare(budget)

if __name__=='__main__':
    stage=sys.argv[1]
    table={'freeze':('gate_e1',2700,freeze),'E1':('gate_e1',2700,run_e1),'E2':('E2',6300,run_e2),
      'E3':('E3',3600,run_e3),'E4':('E4',5400,run_e4),'confirmation':('confirmation',3600,confirmation)}
    name,limit,fn=table[stage];timed_stage(name,limit,fn)
