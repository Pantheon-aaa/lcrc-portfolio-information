"""Independent conflict-certificate and likelihood distribution audits."""
import json,time
from itertools import combinations
import numpy as np
from scipy.stats import norm
from .common import OUT,write,seed,timed_stage
from .compat_models import Family,geometry,draw_loglik,simulate,kl_matrix
from .compatible_qcqp import Compatibility,cvx_reference
from .dcl_master import DCL,greedy,enumerate_exact

def run(budget):
    certificate_rows=[];errors=[]
    for path in sorted((OUT/'geometry').glob('*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if 'cuts' not in data:continue
        f=Family(np.asarray(data['Q']));c=Compatibility(f);gamma=sum(data['gamma'].values())/2
        for ratio,sets in data['cuts'].items():
            epsilon=float(ratio)*gamma
            for ids in sets:
                budget.check();r=c.gamma(ids,tight=True)
                valid=r.lower>epsilon
                certificate_rows.append({'family':data['name'],'ratio':float(ratio),'indices':ids,
                  'epsilon':epsilon,'lower':r.lower,'upper':r.upper,'gap':r.gap,'valid':bool(valid),
                  'solver_status':r.status,'solver_diagnostics':r.diagnostics,'witness':r.w})
                if not valid:errors.append({'family':data['name'],'ratio':ratio,'indices':ids})
    write(OUT/'conflict_audit.json',{'passed':not errors,'certificate_count':len(certificate_rows),'certificates':certificate_rows,'errors':errors})
    # Expected log likelihood ratio equals m KL; test the actual statistic distribution.
    f=geometry('asymmetric',6,6,'distribution-audit');n=2000;m=20;values=[]
    for rep in range(n):
        if rep%200==0:budget.check()
        ll=draw_loglik(f,0,m,0,np.random.default_rng(seed('wishart-audit',rep)));values.append(ll[0]-ll[1])
    K=kl_matrix(f)[0,1];mean=float(np.mean(values));se=float(np.std(values,ddof=1)/np.sqrt(n))
    passed=abs(mean-m*K)<=5*se
    write(OUT/'likelihood_distribution_audit.json',{'passed':passed,'repetitions':n,'observed_mean_logLR':mean,
      'analytic_mean_logLR':m*K,'standard_error':se,'z':(mean-m*K)/se})
    timings=[]
    for d,M in [(3,3),(6,6),(10,12)]:
        f=geometry('asymmetric',d,M,'timing-reference');c0=Compatibility(f)
        g=c0.gamma(range(M));epsilon=.25*(g.lower+g.upper)
        p=np.random.default_rng(seed('algorithm-timing',d,M)).dirichlet(np.ones(M))
        for name in ('A1','Greedy','Enumeration'):
            if name=='Enumeration' and M>6:continue
            c=Compatibility(f);master=DCL(c,epsilon)
            for warm in (False,True):
                budget.check();cpu=time.process_time();wall=time.perf_counter();calls=c.calls
                if name=='A1':answer=master.solve(p);mass=answer['mass_lower']
                elif name=='Greedy':answer=greedy(c,p,epsilon);mass=float(p[f.regret_bounds(answer.w)[1]<=epsilon].sum())
                else:answer,_=enumerate_exact(c,p,epsilon);mass=answer[0]
                timings.append({'d':d,'M':M,'method':name,'warm_geometry_cache':warm,'cpu_seconds':time.process_time()-cpu,
                  'wall_seconds':time.perf_counter()-wall,'new_gamma_calls':c.calls-calls,'certified_mass':mass})
    write(OUT/'algorithm_timing.json',{'protocol':'same public family and posterior; family construction/oracle preparation excluded; warm repeats exactly same query',
      'epsilon_ratio':.5,'records':timings})
    if errors or not passed:raise RuntimeError('independent audit failed')
    print(f'Independent audit passed: {len(certificate_rows)} conflict cuts',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
