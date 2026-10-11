"""Same-objective generic benchmarks and posterior-perturbation checks."""
import json,time,psutil
import numpy as np
from .common import OUT,ROOT,seed,rows,append,write,run
from .families import make,observations
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache
from .budgeted_dcl import solve_budgeted_dcl,enumerate_frontier
from .baselines import cvx_price
from .scip_reference import solve_scip
from .experiments import BETAS,save_family

def benchmark(budget):
    path=OUT/'S3_computation.jsonl';done={(r['case'],r['method'],str(r['beta'])) for r in rows(path)}
    for d in (10,20):
        for M in (6,12,24):
            for kind in ('asymmetric','hidden_curvature','block'):
                preparation_start=time.perf_counter()
                f=make(kind,d,M,'computation',0);save_family(f);_,ll=observations(f,0,32,'Joint',seed('S3-observations',f.name));p=f.posterior(ll)
                gamma=Geometry(f,0).gamma(range(M));g=(gamma.lower+gamma.upper)/2
                preparation_seconds=time.perf_counter()-preparation_start
                for ratio in (.5,.75):
                    eps=ratio*g;case=f'{f.name}-{ratio}';cache_start=time.perf_counter();geo=Geometry(f,eps);pc=PriceCache(MeanBudget(f,p),geo);cache_setup_seconds=time.perf_counter()-cache_start
                    for beta in BETAS:
                        for method,fn in [('A1-B-cold',lambda:solve_budgeted_dcl(f,p,eps,beta)),('A1-B-cached',lambda:solve_budgeted_dcl(f,p,eps,beta,geo,pc)),('SCIP',lambda:solve_scip(f,p,eps,beta))]:
                            budget.check();key=(case,method,str(beta))
                            if key in done:continue
                            start=time.process_time()
                            try:
                                a=fn();mem=psutil.Process().memory_info();append(path,dict(case=case,family=f.name,d=d,M=M,kind=kind,epsilon=eps,ratio=ratio,posterior=p,loglik=ll,method=method,beta=beta,preparation_seconds=preparation_seconds,cache_setup_seconds=cache_setup_seconds,cpu_seconds=time.process_time()-start,process_peak_MiB=getattr(mem,'peak_wset',mem.rss)/2**20,**a))
                            except Exception as e:append(path,dict(case=case,family=f.name,d=d,M=M,kind=kind,method=method,beta=beta,status='failed',failure_reason=repr(e),cpu_seconds=time.process_time()-start))
                    if M<=10:
                        for beta in BETAS:
                            if (case,'Enumeration',str(beta)) in done:continue
                            budget.check();start=time.process_time()
                            try:
                                a,_=enumerate_frontier(f,p,eps,[beta]);append(path,dict(case=case,family=f.name,d=d,M=M,kind=kind,method='Enumeration',beta=beta,cpu_seconds=time.process_time()-start,**a[0]))
                            except Exception as e:append(path,dict(case=case,method='Enumeration',beta=beta,status='failed',failure_reason=repr(e)))
                print(f'S3 computation {kind} d={d} M={M}; CPU {time.process_time():.1f}s',flush=True)

def perturbation(budget):
    path=OUT/'S3_perturbation.jsonl';done={(r['index'],r['location'],r['delta']) for r in rows(path)}
    for i in range(20):
        kind=('asymmetric','hidden_curvature','block')[i%3];f=make(kind,12,6,'perturbation',i);save_family(f)
        _,ll=observations(f,i%6,32,'Joint',seed('S3-perturb-data',i));p=f.posterior(ll);gamma=Geometry(f,0).gamma(range(6));eps=.25*(gamma.lower+gamma.upper)
        answers,pc=enumerate_frontier(f,p,eps,[np.inf]);mb=pc.mb;Rmax=max(.5*np.linalg.eigvalsh(q)[-1]*f.upper-v for q,v in zip(f.Q,f.VL))
        jumps=sorted(set(r.cost_upper/mb.Lhi for r in pc.cache.values() if np.isfinite(r.cost_upper) and r.cost_upper>1e-8*mb.Lhi))
        if not jumps:append(path,dict(index=i,location='degenerate',delta=0,status='no_positive_price_jump'));continue
        jump=jumps[0];betas={'far':jump*.5,'below':jump*(1-1e-6),'above':jump*(1+1e-6)}
        base_mass=float(p[f.regret_bounds(mb.w)[1]<=eps].sum());jump_intervals={}
        for side,b in betas.items():
            lower=max([base_mass]+[float(p[list(G)].sum()) for G,z in pc.cache.items() if z.epsilon_feasible and z.cost_upper<=b*mb.Llo])
            upper=max([lower]+[float(p[list(G)].sum()) for G,z in pc.cache.items() if np.isfinite(z.cost_lower) and z.cost_lower<=b*mb.Lhi])
            jump_intervals[side]=[lower,upper]
        jump_certified=jump_intervals['above'][0]>jump_intervals['below'][1]
        rdist=np.random.default_rng(seed('S3-perturb-r',i)).dirichlet(np.ones(6))
        for location,beta in betas.items():
            for delta in (0,1e-4,1e-3,1e-2):
                if (i,location,delta) in done:continue
                budget.check();q=(1-delta)*p+delta*rdist;tv=np.sum(abs(p-q))/2;a=(beta+2)*Rmax*tv
                ans=solve_budgeted_dcl(f,q,eps,beta);w=ans['weights'];dl,du=mb.increment(w);lo,hi=f.regret_bounds(w);masslo=float(p[hi<=eps].sum());masshi=float(p[lo<=eps].sum())
                tightened=beta*mb.Llo-a
                target=0.;target_upper=0.
                if tightened>=0:
                    target=max([0.]+[float(p[list(G)].sum()) for G,z in pc.cache.items() if z.cost_upper<=tightened])
                    target_upper=max([target]+[float(p[list(G)].sum()) for G,z in pc.cache.items() if z.cost_lower<=beta*mb.Lhi-a])
                rhs=target-2*tv-ans['primary_mass_gap'];rhs_upper=target_upper-2*tv-ans['primary_mass_gap'];budget_rhs=beta*mb.Lhi+a
                assert dl<=budget_rhs+1e-10*max(f.unit,1e-12),'perturbation budget contradiction'
                assert masshi+1e-8>=rhs,'perturbation success contradiction'
                append(path,dict(index=i,kind=kind,family=f.name,location=location,delta=delta,TV=tv,beta=beta,epsilon=eps,
                  price_jump=jump,success_frontier_jump_certified=jump_certified,unperturbed_frontier_intervals=jump_intervals,Rmax=Rmax,increment_lower=dl,increment_upper=du,budget_bound=budget_rhs,
                  posterior_mass_lower=masslo,posterior_mass_upper=masshi,success_bound=rhs,success_bound_upper=rhs_upper,bound_applicable=tightened>=0,
                  budget_bound_certified=du<=budget_rhs,success_bound_certified=masslo>=rhs_upper,status=ans['status'],posterior_p=p,posterior_q=q,weights=w,loglik=ll,data_seed=seed('S3-perturb-data',i)))
        print(f'S3 perturbation {i+1}/20 complete',flush=True)

if __name__=='__main__':
    import sys
    used=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl') if r['stage'].startswith('S3_'))
    allocation=3600 if '--numerical-repair' in sys.argv else max(1,3600-used)
    run('S3_'+sys.argv[1],benchmark if sys.argv[1]=='benchmark' else perturbation,allocation=allocation)
