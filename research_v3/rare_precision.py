"""Fresh fixed-size precision audit, independent of the original rare-event draws."""
import numpy as np
from scipy.stats import beta
from .common import OUT,CFG,write,append,rows,seed,timed_stage
from .compat_models import Family,draw_loglik,kl_matrix
from .evidence_gate import two_model_action

def run(budget):
    protocol={'t':.01,'n':100000,'multipliers':[.5,1.,2.],'seed_namespace':'v3-independent-rare-precision-v1',
      'interval':'Bonferroni one-sided 95% across these three new audits; not pooled with previous samples',
      'reason':'Original 10000-repetition intervals too wide for target .0005; no changes to model or gate.'}
    write(CFG/'rare_precision.json',protocol,frozen=True);path=OUT/'E3_rare_precision.jsonl';done={r['multiplier'] for r in rows(path)}
    t=protocol['t'];n=protocol['n'];f=Family(np.array([[[1,1-t],[1-t,4]],[[1,1.7],[1.7,4]]]))
    KL=kl_matrix(f)[0,1]
    for mult in protocol['multipliers']:
        if mult in done:continue
        m=int(np.ceil(mult*np.log(1/(.05*t))/KL));count=0
        for rep in range(n):
            if rep%1000==0:budget.check()
            ll=draw_loglik(f,1,m,0,np.random.default_rng(seed(protocol['seed_namespace'],mult,rep)))
            count+=two_model_action(ll,t)[1]
        upper=float(beta.ppf(1-.05/3,count+1,n-count))
        append(path,{'t':t,'multiplier':mult,'joint_rows':m,'n':n,'errors':count,'rate':count/n,'simultaneous_95_upper':upper,
          'target':.05*t,'upper_below_target':upper<=.05*t})
    print('Fresh fixed-size rare-event precision audit complete',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
