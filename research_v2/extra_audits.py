"""Predetermined accuracy, information and matched-baseline diagnostics."""
import json,traceback
from dataclasses import replace
from time import perf_counter,process_time,time
import numpy as np
from scipy.optimize import minimize
from core import solve_qp,em_cov,affine_mle,lcrc_affine,MaskLikelihood
from .experiment_v2 import OUT,CFG,write_json,append_row,load_rows,source_hash,single_method
from .new_generators import family,scenarios,truth_points,data_instance,seed_for,make_block_family,importance_scenarios
from .scenario_inference import ScenarioSet,posterior,posterior_from_loglik,component_logmasses,fisher_information
from .risk_envelopes import RiskSpec,Envelope
from .risk_center import solve_center


def baseline_audit():
    setup=json.loads((OUT/'development_setup.json').read_text()); unit=setup['risk_unit']; start=perf_counter(); cpu=process_time()
    manifest=json.loads((CFG/'G1_manifest.json').read_text()); path=OUT/'baseline_audit.jsonl'
    done={(x['instance_id'],x['method']) for x in load_rows('baseline_audit')};cache={}
    for i,r in enumerate(manifest):
        key=r['family_seed']
        if key not in cache:
            fam=family('G1',key); points=truth_points(r['point_seed'])
            sc=ScenarioSet(np.array([fam.Q(x) for x in points]),points,np.zeros(len(points),int),np.full(len(points),-np.log(len(points)))).prepare()
            cache[key]=fam,points,sc
        fam,points,sc=cache[key]; theta=points[r['point_id']];Y=data_instance(fam,theta,r['n'],r['m'],r['data_seed'])
        Q=fam.Q(theta);oracle=solve_qp(Q,constraints={'upper':.2});like=MaskLikelihood(Y)
        names=['Equal-Weight','EM-shrink','Affine-MLE']+[f'OracleGridPrior-{j}' for j in range(4)]
        if r['rep']==0:names+=['REF-LCRC']
        for name in names:
            if (r['instance_id'],name) in done:continue
            tic=perf_counter(); details={}
            try:
                if name=='Equal-Weight':w=np.ones(20)/20
                elif name=='EM-shrink':
                    q,diag=em_cov(Y);w=solve_qp(np.diag(np.diag(q)),constraints={'upper':.2})['w'];details=diag
                elif name=='Affine-MLE':
                    x,details=affine_mle(fam,like);w=solve_qp(fam.Q(x),constraints={'upper':.2})['w']
                elif name=='REF-LCRC':
                    ans,_,_=lcrc_affine(fam,Y,upper=.2,risk_unit=unit);w=ans.weights;details=ans.diagnostics
                else:
                    j=int(name.split('-')[-1]);p=posterior(sc,Y,component_logmasses(sc)[j]);ans=solve_center(sc,p,RiskSpec(name,'mean',rho=1),unit);w=ans.weights
                    details={'privileged':True,'prior':'known finite test-grid distribution; not continuous-prior oracle'}
                append_row(path,r|{'stage':'baseline_audit','method':name,'regret_scaled':float((.5*w@Q@w-(oracle['value_lb']+oracle['value_ub'])/2)/unit),
                    'weights':w,'seconds':perf_counter()-tic,'failure':False,'details':details})
            except Exception as exc: append_row(path,r|{'stage':'baseline_audit','method':name,'failure':True,'error':repr(exc)})
        if i%40==0:print('baseline audit',i+1,'/',len(manifest),flush=True)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'baseline_audit','wall_seconds':perf_counter()-start,'process_cpu_seconds':process_time()-cpu,'unix':time()})


def resolution_audit():
    start=perf_counter();cpu=process_time();setup=json.loads((OUT/'development_setup.json').read_text());unit=setup['risk_unit']
    selection=json.loads((CFG/'selection.json').read_text()); specs=[RiskSpec(**s) for s in selection['confirmation_specs']]
    source=json.loads((CFG/'G2_manifest.json').read_text())
    manifest=[r for r in source if r['point_id']==0 and r['rep']==0 and r['n']==256 and r['m'] in (0,16)]
    write_json(CFG/'resolution_manifest.json',manifest)
    path=OUT/'resolution.jsonl';done={(x['instance_id'],x['size'],x['scramble'],x['method']) for x in load_rows('resolution')}
    for r in manifest:
        fam=family('G2',r['family_seed']);theta=truth_points(r['point_seed'])[r['point_id']]
        Y=data_instance(fam,theta,r['n'],r['m'],r['data_seed']);trueQ=fam.Q(theta);oracle=solve_qp(trueQ,constraints={'upper':.2})
        ref=importance_scenarios(fam,Y,1024,seed_for('audit_reference',r['family_seed']));refpost=posterior(ref,Y)
        for size in (128,256,512):
            for scramble in range(3):
                sc=importance_scenarios(fam,Y,size,seed_for('resolution',r['family_seed'],scramble));post=posterior(sc,Y)
                for spec in specs:
                    key=(r['instance_id'],size,scramble,spec.name)
                    if key in done:continue
                    try:
                        ans=single_method(sc,post,spec,unit,np.array(setup['eb_weights']))
                        targetpost=refpost
                        if spec.name=='REF-EB':
                            from .scenario_inference import mixture_logprior
                            targetpost=posterior_from_loglik(ref,refpost.loglik,mixture_logprior(ref,np.array(setup['eb_weights'])))
                        env=Envelope(ref,targetpost,spec,unit)
                        oracle_constant=0 if spec.absolute else (ref.oracle_lower+ref.oracle_upper)/2
                        val=env.support(ref.costs(ans.weights)-oracle_constant)[0]
                        append_row(path,r|{'method':spec.name,'size':size,'scramble':scramble,'weights':ans.weights,
                            'reference_objective':val,'posterior_mean':post.p@sc.nodes,'posterior':post.diagnostics,
                            'regret_scaled':float((.5*ans.weights@trueQ@ans.weights-(oracle['value_lb']+oracle['value_ub'])/2)/unit),
                            'status':ans.status,'gap':ans.gap,'seconds':ans.seconds,'failure':False})
                    except Exception as exc:append_row(path,r|{'method':spec.name,'size':size,'scramble':scramble,'failure':True,'error':repr(exc)})
                print('resolution',r['family_id'],r['m'],size,scramble,flush=True)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'resolution','wall_seconds':perf_counter()-start,'process_cpu_seconds':process_time()-cpu,'unix':time()})


def mechanisms():
    setup=json.loads((OUT/'development_setup.json').read_text()); unit=setup['risk_unit'];rows=[]
    from core import make_family
    # Gaussian affinity: analytic versus Monte Carlo, to audit the information formula.
    rng=np.random.default_rng(99311); Q0=np.array([[1.,.2],[.2,1.3]]);Q1=np.array([[1.,-.4],[-.4,1.3]])
    z=rng.normal(size=(20000,2))@np.linalg.cholesky(Q0).T
    llratio=-.5*(np.linalg.slogdet(Q1)[1]-np.linalg.slogdet(Q0)[1]+np.einsum('ni,ij,nj->n',z,np.linalg.inv(Q1)-np.linalg.inv(Q0),z))
    lam=.5;J=(1-lam)*np.linalg.inv(Q0)+lam*np.linalg.inv(Q1)
    affinity=np.exp(-.5*((1-lam)*np.linalg.slogdet(Q0)[1]+lam*np.linalg.slogdet(Q1)[1]+np.linalg.slogdet(J)[1]))
    mc=np.exp(lam*llratio);assert abs(mc.mean()-affinity)<5*mc.std()/np.sqrt(len(mc))
    for seed in (31,32):
        fam=make_family(20,seed);sc=scenarios(fam,128,seed)
        for m in (0,4,16,64):
            Y=data_instance(fam,np.zeros(3),256,m,seed_for('mechanism',seed,m));post=posterior(sc,Y)
            fit=post.p@sc.nodes;F,eff=fisher_information(fam.Q(fit),fam.directions,np.isfinite(Y),np.ones(3))
            a=solve_center(sc,post,RiskSpec('conditional',conditional=True),unit)
            b=solve_center(sc,post,RiskSpec('global'),unit)
            # Estimated Fisher eigenbasis changes the scenario grouping (exploratory only).
            _,U=np.linalg.eigh(F); projected=sc.nodes@U[:,-1]
            order=np.argsort(projected,kind='stable'); groups=np.zeros(len(sc.Q),int)
            for j,chunk in enumerate(np.array_split(order,8)):groups[chunk]=j
            altered=ScenarioSet(sc.Q,sc.nodes,groups,sc.log_reference,oracle_lower=sc.oracle_lower,oracle_upper=sc.oracle_upper).prepare()
            ap=posterior_from_loglik(altered,post.loglik); aa=solve_center(altered,ap,RiskSpec('fisher-binned',conditional=True),unit)
            rows.append({'seed':seed,'m':m,'fisher_eigenvalues':np.linalg.eigvalsh(F),'efficient_eigenvalues':np.linalg.eigvalsh(eff),
                'conditional_global_distance':float(np.linalg.norm(a.weights-b.weights)),
                'fisher_group_distance':float(np.linalg.norm(aa.weights-a.weights)),
                'fisher_grouping_interpretation':'data-derived binned approximation, not exact conditional integration'})
    # L1 shared turnover, independent conic reference handled by solver.
    fam=make_block_family(91,d=6);sc=scenarios(fam,32,9,upper=.5);sc.kappa=.001;sc.previous=np.ones(6)/6
    sc.oracle_lower=None;sc.prepare();post=posterior(sc,data_instance(fam,np.zeros(3),64,16,71))
    a=solve_center(sc,post,RiskSpec('turnover',conditional=True),unit)
    assert a.gap<2e-6*unit
    # Five-dimensional family, legal known-reference quadrature.
    fam5=make_block_family(95,d=20,rank=4);sc5=scenarios(fam5,128,32)
    Y=data_instance(fam5,np.zeros(5),64,16,901);post5=posterior(sc5,Y)
    a5=solve_center(sc5,post5,RiskSpec('p5',conditional=True),unit)
    write_json(OUT/'mechanism_audit.json',{'information_geometry':rows,'gaussian_affinity':{'analytic':affinity,'monte_carlo':mc.mean(),'mc_se':mc.std()/np.sqrt(len(mc))},
        'turnover':{'gap':a.gap,'status':a.status},'five_parameter':{'gap':a5.gap,'status':a5.status,'min_eigenvalue':np.linalg.eigvalsh(sc5.Q).min()}})


if __name__=='__main__':
    import sys
    {'baselines':baseline_audit,'resolution':resolution_audit,'mechanisms':mechanisms}[sys.argv[1]]()
