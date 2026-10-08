"""Small five-parameter accuracy/generalization diagnostic, not candidate tuning."""
import time
import numpy as np
from core import solve_qp
from .experiment_v2 import OUT,CFG,write_json,append_row
from .new_generators import make_block_family,importance_scenarios,data_instance,seed_for
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec
from .risk_center import solve_center

def run():
    start=time.perf_counter();cpu=time.process_time();rows=[]
    config={'family_seeds':[91031,91032,91033],'n':64,'m':[0,16],
            'sizes':[128,256,512],'theta':[0,.6,-.6,.3,-.3],
            'scope':'six fixed datasets; diagnostic only, not independent confirmation'}
    write_json(CFG/'five_parameter_audit.json',config)
    specs=[RiskSpec('A-Mean','mean',rho=1),RiskSpec('A-Tail-P:0.5'),
           RiskSpec('B-Tail-P:0.5',conditional=True),RiskSpec('B-Max','max',conditional=True)]
    for seed in config['family_seeds']:
        f=make_block_family(seed,rank=4);Q=f.Q(np.array(config['theta']));oracle=solve_qp(Q,constraints={'upper':.2})
        for m in config['m']:
            Y=data_instance(f,np.array(config['theta']),64,m,seed_for('p5',seed,m))
            for size in config['sizes']:
                sc=importance_scenarios(f,Y,size,seed_for('p5nodes',seed));p=posterior(sc,Y)
                for spec in specs:
                    ans=solve_center(sc,p,spec,.03827751737525126)
                    rows.append({'family_seed':seed,'m':m,'size':size,'method':spec.name,'weights':ans.weights,
                        'regret':float(.5*ans.weights@Q@ans.weights-(oracle['value_lb']+oracle['value_ub'])/2),
                        'status':ans.status,'gap':ans.gap,'posterior':p.diagnostics,'seconds':ans.seconds})
            print('five-parameter',seed,m,flush=True)
    write_json(OUT/'five_parameter_audit.json',rows)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'five_parameter','wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})

if __name__=='__main__':run()
