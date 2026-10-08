"""Fixed audit panel: prior/floor sensitivity and posterior-sampling approximation."""
import json,time
import numpy as np
from .experiment_v2 import OUT,CFG,write_json,append_row,single_method
from .new_generators import family,truth_points,data_instance,importance_scenarios,seed_for
from .scenario_inference import posterior,posterior_from_loglik,mixture_logprior
from .risk_envelopes import RiskSpec,Envelope,tail_weights
from .risk_center import solve_center

def from_mass(sc,p):
    with np.errstate(divide='ignore'): lp=np.log(p)
    return posterior_from_loglik(sc,lp-sc.log_reference)

def run():
    start=time.perf_counter();cpu=time.process_time()
    setup=json.loads((OUT/'development_setup.json').read_text());unit=setup['risk_unit'];eb=np.array(setup['eb_weights'])
    sel=json.loads((CFG/'selection.json').read_text());specs=[RiskSpec(**s) for s in sel['confirmation_specs'] if s['name']!='REF-EB']
    source=json.loads((CFG/'G2_manifest.json').read_text())
    panel=[r for r in source if r['point_id']==0 and r['rep']==0 and r['n']==256 and r['m'] in (0,16)]
    path=OUT/'ablation_audit.jsonl'
    if path.exists():raise RuntimeError('preserve existing audit')
    write_json(CFG/'ablation_manifest.json',panel);mc=[]
    for r in panel:
        fam=family('G2',r['family_seed']);theta=truth_points(r['point_seed'])[0]
        Y=data_instance(fam,theta,r['n'],r['m'],r['data_seed'])
        sc=importance_scenarios(fam,Y,256,seed_for('ablation',r['family_seed']));base=posterior(sc,Y)
        reference=np.exp(sc.log_reference)
        for prior in ('uniform','empirical-bayes'):
            post=base if prior=='uniform' else posterior_from_loglik(sc,base.loglik,mixture_logprior(sc,eb))
            for spec in specs:
                original=solve_center(sc,post,spec,unit)
                for floor in (0.,.01):
                    mass=post.p.copy()
                    if floor:
                        if spec.conditional:
                            for s in np.unique(sc.groups):
                                ix=sc.groups==s;alpha=np.exp(post.log_alpha[s]);b=reference[ix]/reference[ix].sum()
                                mass[ix]=(1-floor)*mass[ix]+floor*alpha*b
                        else:mass=(1-floor)*mass+floor*reference
                    changed=post if not floor else from_mass(sc,mass)
                    ans=original if not floor else solve_center(sc,changed,spec,unit)
                    env=Envelope(sc,post,spec,unit)
                    oracle=0 if spec.absolute else (sc.oracle_lower+sc.oracle_upper)/2
                    target=env.support(sc.costs(ans.weights)-oracle)[0]
                    append_row(path,r|{'method':spec.name,'prior':prior,'floor':floor,'status':ans.status,
                        'gap':ans.gap,'seconds':ans.seconds,'weights':ans.weights,'unfloored_objective':target,
                        'distance_to_unfloored':float(np.linalg.norm(ans.weights-original.weights)),
                        'floor_definition':'conditional mass floor preserving outer marginal' if spec.conditional else 'joint mass floor'})
        # Fixed-action empirical CVaR diagnostic; no second likelihood weighting.
        w=solve_center(sc,base,RiskSpec('mean','mean',rho=1),unit).weights
        loss=sc.costs(w)-(sc.oracle_lower+sc.oracle_upper)/2
        for conditional in (False,True):
            spec=RiskSpec('sampling-target',conditional=conditional);env=Envelope(sc,base,spec,unit);target=env.support(loss)[0]
            for count in (16,64,256,1024,4096):
                errors=[]
                for rep in range(20):
                    rng=np.random.default_rng(seed_for('sampling-audit',r['instance_id'],conditional,count,rep));value=0.
                    for ix,alpha in env.blocks():
                        if alpha==0:continue
                        values=loss[rng.choice(ix,size=count,p=base.p[ix]/base.p[ix].sum())]
                        value+=alpha*np.sort(values)[-count//2:].mean()
                    errors.append(value-target)
                mc.append(r|{'conditional':conditional,'draws_per_group':count,'repetitions':20,'target':target,
                    'rmse':float(np.sqrt(np.mean(np.square(errors)))),'bias':float(np.mean(errors))})
        print('ablation',r['family_id'],r['m'],flush=True)
    write_json(OUT/'posterior_sampling_tail.json',mc)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'ablation_audit','wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})

if __name__=='__main__':run()
