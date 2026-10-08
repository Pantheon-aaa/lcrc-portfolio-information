"""Independent numerical Gate V2. Failures prevent performance execution."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.linalg import null_space
from core import solve_qp
from .scenario_inference import ScenarioSet,posterior,posterior_from_loglik,observation_operator
from .new_generators import make_block_family,scenarios,data_instance
from .risk_envelopes import RiskSpec,Envelope,tail_weights
from .risk_center import solve_center,conic_reference

ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def run():
    rng=np.random.default_rng(93081); results={}
    fam=make_block_family(8201,d=6); sc=scenarios(fam,32,610,upper=.6)
    Y=data_instance(fam,np.array([.1,.4,-.3]),64,0,6101); post=posterior(sc,Y)
    for s in np.unique(sc.groups):
        ix=sc.groups==s
        assert np.ptp(post.loglik[ix])<1e-10
        assert np.max(abs(np.exp(post.log_conditional[ix])-1/ix.sum()))<1e-12
    results['unidentified_conditional_update']=True
    p=posterior(sc,data_instance(fam,np.array([.1,.4,-.3]),64,16,6102))
    unit=.1
    specs=[RiskSpec('mean','mean',rho=1),RiskSpec('bmean','mean',rho=1,conditional=True),
           RiskSpec('tail',rho=.5),RiskSpec('ctail',rho=.5,conditional=True),
           RiskSpec('max','max'),RiskSpec('cmax','max',conditional=True),
           RiskSpec('lp','max',penalty=.1),RiskSpec('lpt',penalty=.1,reference=True),
           RiskSpec('blend',blend=.5),RiskSpec('soft','soft',temperature=.1),
           RiskSpec('csoft','soft',temperature=.1,conditional=True),
           RiskSpec('lr','lr',reference=True,radius=.1),RiskSpec('budget','budget',reference=True),
           RiskSpec('filter','filter'),RiskSpec('sample','sample'),RiskSpec('pr','pr')]
    audits=[]
    for spec in specs:
        a=solve_center(sc,p,spec,unit); env=Envelope(sc,p,spec,unit)
        _,_,ref,status=conic_reference(sc,env)
        assert a.lower-2e-7*unit<=ref<=a.upper+2e-7*unit,(spec.name,a.lower,ref,a.upper,a.diagnostics)
        assert a.gap<3e-6*unit,(spec.name,a.gap,a.diagnostics)
        assert a.residual<1e-8
        audits.append({'method':spec.name,'gap':a.gap,'reference':ref,'status':a.status})
        print('gate',spec.name,a.gap,flush=True)
    results['primal_dual_checks']=audits
    ma=solve_center(sc,p,specs[0],unit); bm=solve_center(sc,p,specs[1],unit)
    qp=solve_qp(np.einsum('m,mij->ij',p.p,sc.Q),constraints={'upper':sc.upper})
    assert np.linalg.norm(ma.weights-bm.weights)<1e-7
    assert np.linalg.norm(ma.weights-qp['w'])<1e-7
    results['mean_identities']=True
    errors=[]
    for _ in range(30):
        w=rng.dirichlet(np.ones(6)); x=sc.costs(w)-(sc.oracle_lower+sc.oracle_upper)/2
        vals=[Envelope(sc,p,s,unit).support(x)[0] for s in [specs[0],specs[3],specs[2],specs[4]]]
        assert np.min(np.diff(vals))>=-1e-12
        uniform=np.ones(len(x))/len(x)
        assert abs(tail_weights(x,uniform,.5)@x-np.sort(x)[-len(x)//2:].mean())<1e-12
        env=Envelope(sc,p,specs[3],unit); _,q,_=env.support(x)
        for g,mass in enumerate(env.alpha): assert abs(q[sc.groups==g].sum()-mass)<1e-12
        # Duplicate each atom and split its probability; value must not change.
        first=tail_weights(x,p.p,.25)@x
        second=tail_weights(np.repeat(x,2),np.repeat(p.p/2,2),.25)@np.repeat(x,2)
        errors.append(abs(first-second)); assert errors[-1]<1e-12
    results['ordering_replication_topk_group_mass']={'passed':True,'max_error':max(errors)}
    a=solve_center(sc,post,RiskSpec('bp',conditional=True),unit)
    b=solve_center(sc,post,RiskSpec('bu',conditional=True,reference=True),unit)
    assert np.linalg.norm(a.weights-b.weights)<1e-7
    # Unobservable directions hidden by coordinate rotation.
    from core import make_family
    affine=make_family(6,12); masks=np.isfinite(Y)
    A=observation_operator(affine.directions,masks); O=np.linalg.qr(rng.normal(size=(3,3)))[0]
    rotated=np.einsum('rs,rij->sij',O,affine.directions)
    Ar=observation_operator(rotated,masks)
    assert np.linalg.norm(Ar-A@O)<1e-12
    N=null_space(A); Nr=O@null_space(Ar)
    assert np.linalg.norm(N@N.T-Nr@Nr.T)<1e-10
    results['rotated_observation_kernel']=True
    def naive(x):
        r=np.array([(x-np.sqrt(3))**2,(x+1)**2,(x-1)**2]); p0=np.array([.1,.8,.1])
        ix=np.argsort(-r,kind='stable')[:2]; return p0[ix]@r[ix]/p0[ix].sum()
    assert abs(naive(-1e-9)-2)<1e-7 and abs(naive(1e-9)-11/9)<1e-7
    results['naive_discontinuity']={'left':naive(-1e-9),'right':naive(1e-9)}
    odd=np.full(len(sc.Q),.01/(len(sc.Q)-3));odd[:3]=[.34,.33,.32]
    oddpost=posterior_from_loglik(sc,np.log(odd)-sc.log_reference)
    oddenv=Envelope(sc,oddpost,RiskSpec('odd-filter','filter'),unit)
    losses=np.arange(len(sc.Q),dtype=float)
    assert abs(oddenv.support(losses)[0]-1.5)<1e-12
    assert abs(oddenv.block_rho(np.arange(len(sc.Q)))-2/3)<1e-12
    results['filtered_integer_topk']=True
    # API information permissions: no evaluator fields in the solver signature.
    import inspect
    assert set(inspect.signature(solve_center).parameters)=={'sc','post','spec','unit','reference','time_limit'}
    results['public_input_signature']=True
    # Re-preparing a mutable scenario object must not reuse another QP's oracle.
    cached=sc.oracle_upper.copy()
    sc.kappa=.003;sc.previous=np.zeros(sc.Q.shape[1]);sc.previous[:2]=.5
    sc.prepare()
    assert np.max(np.abs(sc.oracle_upper-cached))>1e-6
    check=solve_qp(sc.Q[0],sc.mu[0],{'upper':sc.upper},sc.kappa,sc.previous)
    assert sc.oracle_lower[0]<=check['value_ub'] and sc.oracle_upper[0]>=check['value_lb']
    sc.upper=.25;sc.prepare()
    check=solve_qp(sc.Q[0],sc.mu[0],{'upper':sc.upper},sc.kappa,sc.previous)
    assert sc.oracle_lower[0]<=check['value_ub'] and sc.oracle_upper[0]>=check['value_lb']
    results['oracle_cache_problem_signature']=True
    results['passed']=True
    (OUT/'gate_v2.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Gate V2 passed',flush=True)


if __name__=='__main__': run()
