"""Frozen three-study scheduler, balanced truth strata, full interval records."""
import json,sys,time,platform,hashlib
from pathlib import Path
import numpy as np
import psutil
from .common import ROOT,OUT,write,append,rows,seed,run,source_hash
from .families import analytic,analytic_price,make,observations,equivalence,affinity
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .budgeted_dcl import solve_budgeted_dcl,result
from .baselines import budget_greedy,budget_surrogate,original_a1

BETAS=[0,.1,.25,.5,np.inf]
KINDS=['asymmetric','hidden_curvature','block']

def require_gate():
    if not json.loads((OUT/'gate0.json').read_text())['passed']:raise RuntimeError('Gate 0 required')
def specs(split):
    n=2 if split=='development' else 4
    return [(kind,12 if kind=='block' else (10 if i%2==0 else 12),6,split,i) for i in range(n) for kind in KINDS]
def save_family(f):
    write(OUT/'families'/f'{f.name}.json',dict(name=f.name,Q=f.Q,upper=f.upper,hash=f.hash,metadata=f.metadata,oracle=f.oracle,oracle_diagnostics=f.oracle_diagnostics))

def record(path,key,method,beta,f,p,epsilon,answer,truths):
    w=answer['weights'];lo,hi=f.regret_bounds(w);maw=answer['ma_weights'] if 'ma_weights' in answer else f.mean(p)['w'];ma_regrets=np.mean(np.array(f.regret_bounds(maw)),axis=0)
    for truth in truths:
        append(path,{**key,'method':method,'beta':beta,**answer,'true_model':truth,
          'true_success_lower':bool(hi[truth]<=epsilon),'true_success_upper':bool(lo[truth]<=epsilon),
          'true_regret_lower':lo[truth],'true_regret_upper':hi[truth],
          'true_regret':float((lo[truth]+hi[truth])/2),'true_regret_MA':float(ma_regrets[truth]),
          'memory_MiB':psutil.Process().memory_info().rss/2**20})

def s1(budget):
    require_gate();path=OUT/'S1.jsonl';done={r['query'] for r in rows(OUT/'S1_completed.jsonl')};errors=[]
    for d in (3,6,10,20):
        for fraction in (.25,.75):
            f,L0=analytic(d,fraction);save_family(f);p=np.ones(d)/d;mb=MeanBudget(f,p)
            for ratio in (.5,.75,.875,.95,1.05):
                eps=ratio*L0;geo=Geometry(f,eps);pc=PriceCache(mb,geo);prices=[pc.get(())]
                # Valid symmetry quotient: all sets of a given cardinality are equivalent.
                for m in range(1,d+1):prices.append(pc.get(range(m),tight=True))
                formulas=analytic_price(d,f.metadata['t'],eps)
                for beta in (0,.01,.025,.05,.1,.25,np.inf):
                    budget.check();query=f'S1-{d}-{fraction}-{ratio}-{beta}'
                    if query in done:continue
                    bl,bu=mb.bounds(beta);available=[m for m,r in enumerate(prices) if r.cost_upper<=bl and r.epsilon_feasible]
                    best=max(available,default=0);w=prices[best].witness_weights
                    allowed=[a['m'] for a in formulas if a['cost']<=beta*L0 and np.isfinite(a['cost'])]
                    exact=1. if eps>=L0 else max([0]+allowed)/d
                    actual=float(p[f.regret_bounds(w)[1]<=eps].sum())
                    upper=max([actual]+[m/d for m,r in enumerate(prices) if np.isfinite(r.cost_lower) and r.cost_lower<=bu])
                    answer=result(mb,eps,beta,w,actual,upper,prices[best].cost_upper-prices[best].cost_lower,'symmetry_quotient',time.perf_counter(),geo,pc)
                    if abs(actual-exact)>1e-8:errors.append(dict(query=query,actual=actual,expected=exact,upper=upper))
                    key=dict(query=query,dataset=query,experiment='S1',split='analytic',family=f.name,family_hash=f.hash,d=d,M=d,t=f.metadata['t'],epsilon=eps,epsilon_scale=ratio,L0=L0,analytic_mass=exact,observed_scalar_count=0)
                    record(path,key,'A1-B-symmetry',beta,f,p,eps,answer,range(d))
                    g=budget_greedy(f,p,eps,beta,geo);record(path,key,'Budget-Greedy',beta,f,p,eps,g,range(d))
                    if d<=6 or (ratio==.875 and beta in (.1,np.inf)):
                        a=solve_budgeted_dcl(f,p,eps,beta,geo,pc);record(path,key,'A1-B-generic',beta,f,p,eps,a,range(d))
                        if a['optimal_mass_lower']>exact+1e-8 or a['optimal_mass_upper']<exact-1e-8:errors.append(dict(query=query,error='generic interval excludes exact'))
                    if beta==0:
                        a=solve_budgeted_dcl(f,p,eps,0);record(path,key,'MA',0,f,p,eps,a,range(d))
                    if np.isinf(beta):record(path,key,'A1-v3',beta,f,p,eps,original_a1(f,p,eps),range(d))
                    append(OUT/'S1_completed.jsonl',{'query':query})
                print(f'S1 d={d}, t fraction={fraction}, epsilon/L0={ratio} complete',flush=True)
    write(OUT/'S1_validation.json',dict(passed=not errors,errors=errors,queries=len(list(rows(OUT/'S1_completed.jsonl')))))
    if errors:raise RuntimeError('S1 analytic mismatch')

def freeze(budget):
    require_gate()
    if not json.loads((OUT/'S1_validation.json').read_text())['passed']:raise RuntimeError('S1 required')
    if (ROOT/'config_frozen.json').exists():print('Existing frozen configuration preserved');return
    geometry=[];diagnostics=[];families=[]
    for spec in specs('development'):
        f=make(*spec);save_family(f);families.append(f);ga=Geometry(f,0).gamma(range(6));uf=float(np.median(f.costs(f.base)))
        geometry.append((ga.lower+ga.upper)/2/uf)
        for design in ('Separated','Joint'):
            B=affinity(f,design);groups=equivalence(f,design);bounds=[]
            for j in range(6):
                C=next(c for c in groups if j in c);outside=[k for k in range(6) if k not in C]
                bounds.append(float(np.sqrt(19)*np.sum(np.exp(-256*B[j,outside])/np.sqrt(len(C)))))
            diagnostics.append(dict(family=f.name,design=design,B=B,classes=groups,worst_bound_256=max(bounds)))
    quantiles=np.quantile(geometry,[.25,.5,.75]);thresholds=sorted(set(float(x) if x>0 else 1e-8 for x in quantiles*np.array([.5,.75,1.25])))
    ns=[0,32,256]+([1024] if any(x['worst_bound_256']>.05 for x in diagnostics) else [])
    timings=[]
    for i in range(20):
        budget.check();f=families[i%6];design=('Separated','Joint')[i%2];n=(32,256)[(i//2)%2]
        _,ll=observations(f,i%6,n,design,seed('v4a-timing',i));p=f.posterior(ll);ga=Geometry(f,0).gamma(range(6));eps=.5*(ga.lower+ga.upper)/2;geo=Geometry(f,eps);pc=PriceCache(MeanBudget(f,p),geo)
        start=time.process_time()
        a=[solve_budgeted_dcl(f,p,eps,b,geo,pc) for b in BETAS]
        timings.append(dict(index=i,family=f.name,cpu_seconds=time.process_time()-start,status=[x['status'] for x in a]))
    import scipy,cvxpy,clarabel,pyscipopt
    config=dict(baseline_commit='edac208d27e8751faab5a2ca94a1cd267cd4439b',seed_namespace='v4a-frozen-20261011',
      betas=[0,.1,.25,.5,'inf'],geometry_ratios=[.25,.5,.75,1.25],common_epsilon_scales=thresholds,development_geometry=geometry,
      n_blocks=ns,development_repetitions=2,confirmation_repetitions=10,prior='uniform',upper=.2,
      source_hash=source_hash(),frozen_algorithm_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py') if p.name in ['mean_budget.py','price_oracle.py','budgeted_dcl.py','families.py','baselines.py']},
      zero_threshold_replacement=1e-8,bootstrap_repetitions=10000,cpu_limit_seconds=21600,query_limit_seconds=30,
      primary_mass_tolerance=1e-8,normalized_price_tolerance=1e-6,primal_tolerance=1e-9,
      environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cvxpy.__version__,clarabel=clarabel.__version__,pyscipopt=pyscipopt.__version__),
      attachment_sha256=hashlib.sha256((ROOT/'INPUT_PROTOCOL_ZH.md').read_bytes()).hexdigest())
    write(ROOT/'config_frozen.json',config);write(OUT/'development_information.json',diagnostics);write(OUT/'timing20.json',timings)
    print(f'Frozen n={ns}, common epsilon scales={thresholds}',flush=True)

def s2(budget,split):
    require_gate();cfg=json.loads((ROOT/'config_frozen.json').read_text());nrep=cfg[f'{split}_repetitions'];path=OUT/f'S2_{split}.jsonl';completed=OUT/f'S2_{split}_completed.jsonl';done={r['dataset'] for r in rows(completed)}
    families=[make(*s) for s in specs(split)];systems={}
    for f in families:
        save_family(f);ga=Geometry(f,0).gamma(range(f.M));gamma=(ga.lower+ga.upper)/2;uf=float(np.median(f.costs(f.base)))
        epsilons=[('geometry',ratio,ratio*gamma) for ratio in cfg['geometry_ratios']]+[('common',ratio,ratio*uf) for ratio in cfg['common_epsilon_scales']]
        systems[f.name]=(epsilons,{(kind,ratio):Geometry(f,eps) for kind,ratio,eps in epsilons},uf)
    # Round-robin completion; a budget stop cannot silently select favorable families.
    jobs=[(0,0,0,design,f) for design in ('Separated','Joint') for f in families]
    jobs += [(rep,truth,n,design,f) for rep in range(nrep) for truth in range(6) for n in cfg['n_blocks'] if n for design in ('Separated','Joint') for f in families]
    for number,(rep,truth,n,design,f) in enumerate(jobs):
        budget.check();dataset=f'S2-{split}-{f.name}-{design}-{n}-{truth}-{rep}'
        if dataset in done:continue
        epsilons,geos,uf=systems[f.name];ds=seed(cfg['seed_namespace'],dataset);Y,ll=observations(f,truth,n,design,ds);p=f.posterior(ll);truths=[truth] if n else range(6)
        key=dict(dataset=dataset,experiment='S2',split=split,family=f.name,family_seed=f.metadata['seed'],data_seed=ds,family_hash=f.hash,
          kind=f.metadata['kind'],d=f.d,M=f.M,mask_design=design,n_blocks=n,observed_scalar_count=n*f.d,rep=rep,prior_name='uniform',risk_unit=uf)
        append(OUT/f'S2_{split}_evidence.jsonl',{**key,'true_model_for_evaluator':truth if n else None,'posterior':p,'log_observation_density':ll})
        for scale,ratio,eps in epsilons:
            geo=geos[(scale,ratio)];pc=PriceCache(MeanBudget(f,p),geo);k={**key,'epsilon':eps,'epsilon_scale':scale,'epsilon_ratio':ratio}
            a=solve_budgeted_dcl(f,p,eps,0,geo,pc);record(path,k,'MA',0,f,p,eps,a,truths)
            try:record(path,k,'A1-v3',np.inf,f,p,eps,original_a1(f,p,eps),truths)
            except Exception as e:
                for j in truths:append(path,{**k,'method':'A1-v3','beta':'inf','status':'failed','failure_reason':repr(e),'true_model':j})
            for beta in BETAS:
                for name,fn in [('A1-B',lambda:solve_budgeted_dcl(f,p,eps,beta,geo,pc)),('Budget-Greedy',lambda:budget_greedy(f,p,eps,beta)),
                  ('Budget-Hinge',lambda:budget_surrogate(f,p,eps,beta)),*[(f'Budget-CVaR-{rho}',lambda rho=rho:budget_surrogate(f,p,eps,beta,'tail',rho)) for rho in (.25,.5,1.)]]:
                    budget.check()
                    try:a=fn();record(path,k,name,beta,f,p,eps,a,truths)
                    except Exception as e:
                        for j in truths:append(path,{**k,'method':name,'beta':beta,'true_model':j,'status':'failed','failure_reason':repr(e)})
            append(OUT/f'S2_{split}_certificates.jsonl',dict(dataset=dataset,epsilon=eps,epsilon_scale=scale,posterior=p,geometry_cuts=geo.cuts,
              prices=[dict(models=G,lower=r.cost_lower,upper=r.cost_upper,status=r.status,witness_weights=r.witness_weights) for G,r in pc.cache.items()]))
        append(completed,{'dataset':dataset});done.add(dataset)
        if number%24==0:print(f'S2 {split}: {len(done)}/{len(jobs)} datasets; CPU {time.process_time():.1f}s',flush=True)
    write(OUT/f'S2_{split}_manifest.json',dict(complete=True,datasets=len(done),expected_datasets=len(jobs),source_hash=source_hash()))

if __name__=='__main__':
    stage=sys.argv[1]
    run(stage,{'S1':s1,'freeze':freeze,'development':lambda b:s2(b,'development'),'confirmation':lambda b:s2(b,'confirmation')}[stage])
