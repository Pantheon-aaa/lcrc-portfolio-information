"""Accuracy escalation: more inner nodes and exact hidden-box vertices for B-Max."""
import json,time,itertools
import numpy as np
from .experiment_v2 import OUT,CFG,write_json,append_row,load_rows,single_method
from .new_generators import family,truth_points,data_instance,importance_scenarios,seed_for
from .scenario_inference import ScenarioSet,posterior,posterior_from_loglik,mixture_logprior
from .risk_envelopes import RiskSpec,Envelope
from .risk_center import solve_center

def vertices(fam,sc,post):
    nodes=np.array([np.r_[sc.nodes[sc.groups==s][0,0],eta] for s in np.unique(sc.groups)
                    for eta in itertools.product((-1.,1.),repeat=fam.eta_rank)])
    count=2**fam.eta_rank;groups=np.repeat(np.unique(sc.groups),count)
    vs=ScenarioSet(np.array([fam.Q(x) for x in nodes]),nodes,groups,np.full(len(nodes),-np.log(len(nodes)))).prepare()
    # The posterior outer marginal is held fixed. Inner masses are immaterial to max.
    lp=np.repeat(post.log_alpha,count)-np.log(count)
    vp=posterior_from_loglik(vs,lp-vs.log_reference)
    return vs,vp

def run():
    begin=time.perf_counter();cpu=time.process_time();records=[]
    setup=json.loads((OUT/'development_setup.json').read_text());unit=setup['risk_unit'];eb=np.array(setup['eb_weights'])
    panel=json.loads((CFG/'resolution_manifest.json').read_text());old=load_rows('resolution')
    specs=[RiskSpec(**s) for s in json.loads((CFG/'selection.json').read_text())['confirmation_specs']]
    write_json(CFG/'high_resolution_audit.json',{'panel':panel,'size':4096,'shape':[32,128],
        'purpose':'triggered by initial scramble sensitivity, never fed back into candidate selection',
        'B_Max_reference':'exact eta-box corners for each sampled phi; phi integration remains approximate'})
    for r in panel:
        f=family('G2',r['family_seed']);Y=data_instance(f,truth_points(r['point_seed'])[0],r['n'],r['m'],r['data_seed'])
        sc=importance_scenarios(f,Y,4096,seed_for('high-resolution',r['family_seed']));post=posterior(sc,Y)
        vs,vp=vertices(f,sc,post)
        for spec in specs:
            if spec.name=='B-Max':ss,pp=vs,vp
            else:ss,pp=sc,post
            if spec.name=='REF-EB':pp=posterior_from_loglik(ss,post.loglik,mixture_logprior(ss,eb))
            result=solve_center(ss,pp,spec,unit)
            env=Envelope(ss,pp,spec,unit);V=0 if spec.absolute else (ss.oracle_lower+ss.oracle_upper)/2
            compare=[]
            for row in old:
                if row['instance_id']!=r['instance_id'] or row['method']!=spec.name or row['failure']:continue
                w=np.array(row['weights']);obj=env.support(ss.costs(w)-V)[0]
                compare.append({'size':row['size'],'scramble':row['scramble'],
                    'high_reference_excess':obj-result.objective,'weight_distance':float(np.linalg.norm(w-result.weights))})
            records.append(r|{'method':spec.name,'high_size':len(ss.Q),'gap':result.gap,'status':result.status,
                'objective':result.objective,'weights':result.weights,'seconds':result.seconds,
                'posterior':post.diagnostics,'comparison':compare,
                'scope':'exact hidden maximization, approximate outer integration' if spec.name=='B-Max' else '4096-node approximate integration; no formal global accuracy bound'})
            print('high-resolution',r['family_id'],r['m'],spec.name,result.status,flush=True)
        write_json(OUT/'high_resolution_audit.json',records)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'high_resolution','wall_seconds':time.perf_counter()-begin,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})

if __name__=='__main__':run()
