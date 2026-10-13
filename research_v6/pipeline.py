import itertools,time,inspect,gzip
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linprog
from core import MaskLikelihood
from research_v4a.mean_budget import Family,MeanBudget
from research_v4a.price_oracle import PriceCache,Geometry as OldGeometry
from research_v4a.families import analytic,make as old_make
from research_v3.compat_models import e1
from research_v3.compatible_qcqp import cvx_reference
from .common import *
from .models import make,specs,equivalence,kl,actions
from .certificates import Geometry,mat,mv,F
from .information import tables,randomized_platform
from .likelihood import constants,increments,GaussianStream,verified_exclusions
from .sequential import simulate,POLICIES
def base_control(d=10,M=6):
    Q=np.repeat(np.eye(d)[None],M,axis=0)
    for j in range(M):
        i=j%(d-3);a,b,c,e=i,i+1,i+2,i+3
        for x,y,v in [(a,b,1),(b,c,-1),(c,e,1),(e,a,-1)]:Q[j,x,y]=Q[j,y,x]=v/16
    return Family(Q,.2,'common-exact',dict(kind='common_control'))
def save_family(f):
    p=RUN/'families'/f'{f.name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():np.savez_compressed(p,Q=f.Q,upper=f.upper)
    write(RUN/'families'/f'{f.name}.json',dict(name=f.name,metadata=f.metadata,hash=f.hash))
def load_family(name):
    a=np.load(RUN/'families'/f'{name}.npz');m=read(RUN/'families'/f'{name}.json')
    return Family(a['Q'],float(a['upper']),name,m['metadata'])
def geometry(f):
    path=RUN/'geometry'/f'{f.name}.json'
    if path.exists():return {int(k):v for k,v in read(path).items()}
    g=Geometry(f);data=g.all();write(path,data);save_family(f);return data
def setup(f,ratio,premium,allowed):
    geo=geometry(f);allr=geo[(1<<f.M)-1];eps=ratio*(allr['lower']+allr['upper'])/2
    key=f'{f.name}-{ratio}-{premium}-'+''.join(map(str,allowed));path=RUN/'tables_klv2'/f'{key}.npz'
    if path.exists():
        a=np.load(path);tab={k:a[k] for k in a.files};tab['classes']=equivalence(f,allowed)
    else:
        tab=tables(f,geo,eps,[f.d]*3+[f.d*premium],allowed);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{k:v for k,v in tab.items() if k!='classes'})
    return geo,tab
def gate():
    checks=[]
    def check(name,condition,**detail):
        checks.append(dict(name=name,passed=bool(condition),**detail));write(OUT/'gate0.json',dict(passed=all(x['passed'] for x in checks),checks=checks))
        if not condition:raise AssertionError(name)
    f=make('block',10,6,'gate',0);g=geometry(f);r=g[63];eps=.5*(r['lower']+r['upper']);tab=tables(f,g,eps,[10]*4,[0,1,2,3]);con=constants(f)
    check('exact_covariance_symmetry',np.array_equal(f.Q,f.Q.transpose(0,2,1)))
    check('rational_gamma_bracket',r['lower']<=r['upper'] and r['lower']>0)
    cv,w=cvx_reference(f,range(6));check('independent_cvx_geometry',r['lower']-1e-9<=cv<=r['upper']+1e-9,cvx=cv,interval=[r['lower'],r['upper']])
    rng=np.random.default_rng(22);X=rng.normal(size=(7,f.d));aa=actions(f.d)
    for a in range(4):
        ll,err=increments(X,con[a]);Y=np.full((len(X)*len(aa[a].masks),f.d),np.nan)
        for k,s in enumerate(aa[a].masks):Y[k*len(X):(k+1)*len(X),s]=X[:,s]
        expected=np.array([MaskLikelihood(Y)(Q) for Q in f.Q]);actual=ll.sum(0)-.5*len(X)*f.d*np.log(2*np.pi)
        check('complete_mask_likelihood_'+str(a),np.max(abs(actual-expected))<=err.sum(0).max()+1e-10)
    ll,err=increments(X,con[1]);check('class_odds_invariance',np.array_equal(ll[:,0],ll[:,1]) and np.array_equal(ll[:,3],ll[:,4]))
    check('method_has_no_truth_argument','truth' not in inspect.signature(__import__('research_v6.sequential',fromlist=['advance']).advance.py_func).parameters)
    check('future_action_not_in_density',all(a.scalar_count==f.d for a in aa))
    check('separated_classes',equivalence(f,[1])==[[0,1,2],[3,4,5]])
    K=kl(f)
    check('structural_zero_KL',all(K[1,i,j]==0 for cl in equivalence(f,[1]) for i in cl for j in cl))
    try:Geometry(f).gamma(0);empty=False
    except ValueError:empty=True
    check('empty_set_rejected',empty)
    ff=base_control();gg=geometry(ff);check('exact_common_optimizer',gg[63]['upper']==0 and gg[63]['status']=='exact_common_optimizer')
    z=e1(.01);e=Family(z.Q,1.,'e1-gate',dict(kind='simplex'));eg=geometry(e);epsilon=7*.01**2/16
    ok=[False]+[eg[s]['upper']<=epsilon for s in range(1,8)];lp=randomized_platform(3,ok,[0,1,2]);check('pairwise_not_joint',all(ok[s] for s in (3,5,6)) and not ok[7]);check('randomized_one_third',abs(lp['failure']-1/3)<1e-10)
    for C in tab['conflicts']:
        if C:check('conflict_'+str(C),g[int(C)]['lower']>eps and int(C).bit_count()<=f.d)
    check('boundary_not_relaxed',not(g[63]['upper']<=g[63]['lower']))
    p=np.ones(6)/6;pc=PriceCache(MeanBudget(f,p),OldGeometry(f,eps))
    try:pc.validate(f,np.array([.2,.2,.15,.15,.15,.15]),eps);rejected=False
    except ValueError:rejected=True
    check('cross_posterior_cache_rejected',rejected)
    for j in range(6):
        check('oracle_interval_'+str(j),g[1<<j]['regret_upper'][j]<1e-9)
    # Exclusions are permanent; the path mixture is recomputed from cumulative ll.
    ll=np.array([0.,-20.,-1.]);ex=verified_exclusions(ll,np.full(3,1e-12),7,.05);check('interval_evidence_threshold',1 in ex and 0 not in ex)
    s=simulate(f,g,tab,con,0,0,.5,.05,1,[0,1,2,3],'equal_cost',horizon=1000)
    check('sequential_smoke',not s['false_certification'] and all(b['t']<=s['stop_t'] if s['stop_t'] else True for b in s['events']))
    check('persistent_exclusions',all((a['confidence_mask']&b['confidence_mask'])==b['confidence_mask'] for a,b in zip(s['events'],s['events'][1:])))
    # Duplicate atoms split the mixture mass; permutation merely reorders atoms.
    l=np.array([-.2,-5.,-3.]);q=logsumexp(l)-np.log(3);qd=logsumexp(np.repeat(l,2))-np.log(6)
    check('mixture_copy_invariance',abs(q-qd)<1e-14);check('mixture_permutation',abs(q-(logsumexp(l[::-1])-np.log(3)))<1e-14)
    copied=Family(np.repeat(f.Q,2,axis=0),f.upper,'copy-gate');cr=Geometry(copied).gamma((1<<12)-1)
    check('geometric_copy_invariance',max(r['lower'],cr['lower'])<=min(r['upper'],cr['upper'])+1e-10)
    perm=Family(f.Q[::-1],f.upper,'perm-gate');pr=Geometry(perm).gamma(63)
    check('geometric_permutation',max(r['lower'],pr['lower'])<=min(r['upper'],pr['upper'])+1e-10)
    from research_v4a.api import solve_budgeted_dcl
    b0=solve_budgeted_dcl(f,np.ones(6)/6,eps,0)
    check('beta_zero_MA',np.array_equal(b0['weights'],b0['ma_weights']))
    # The two independent block times must not expose unobserved cross-products.
    x2=X.copy();x2[:,5:]*=-1
    a0,_=increments(X,con[0]);a1,_=increments(x2,con[0]);check('singleton_hidden_signs',np.array_equal(a0,a1))
    c0,_=increments(X,con[1]);c1,_=increments(x2,con[1]);check('separated_hidden_cross_signs',np.array_equal(c0,c1))
    write(OUT/'gate0.json',dict(passed=True,checks=checks,smoke={k:v for k,v in s.items() if k!='actions'}));print('GATE',len(checks),'passed',flush=True)
def diagnose():
    start=time.process_time();out=[];local=[]
    families=[Family(e1(.01).Q,1,'old-e1',dict(kind='simplex'))]
    for k in KINDS:
        for i in range(2):
            f=old_make(k,10 if i%2==0 else 12,6,'development',i);f.name='old-'+f.name;families.append(f)
    families.append(base_control())
    for f in families:
        check_budget(start,5400);g=geometry(f);gamma=g[(1<<f.M)-1];unit=float(np.median(f.costs(f.base)))
        ratios=[.25,.5,.75,1.,1.25];epses=[('relative',x,x*(gamma['lower']+gamma['upper'])/2) for x in ratios]
        epses += [('old_common',x,x*unit) for x in read(ROOT.parent/'research_v4a/config_frozen.json')['common_epsilon_scales']]
        for kind,x,eps in epses:
            ok=[True]+[g[s]['upper']<=eps for s in range(1,1<<f.M)]
            for allowed in ([0],[1],[0,1,2,3]):
                classes=equivalence(f,allowed)
                for cl in classes:
                    mask=sum(1<<j for j in cl);r=g[mask];platform=randomized_platform(f.M,ok,cl) if any(ok[1:]) else None
                    out.append(dict(family=f.name,kind=f.metadata.get('kind'),threshold_type=kind,threshold=x,epsilon=eps,gamma=gamma,gamma_over_u=(gamma['lower']+gamma['upper'])/2/unit,allowed=allowed,group=cl,class_gamma=r,randomized=platform))
        if f.metadata.get('kind')=='hidden_curvature':
            H=f.Q-np.eye(f.d)-.08*np.diag(np.linspace(-1,1,f.d));base=np.eye(f.d)
            for strength in (0,.01,.03,.08):
                z=Family(base+H+strength*np.diag(np.linspace(-1,1,f.d)),.2,f'{f.name}-diagonal-{strength}',dict(kind='hidden_ablation'))
                zz=geometry(z);w0=np.ones(f.d)/f.d;P=np.eye(f.d)-np.ones((f.d,f.d))/f.d
                tangent=np.linalg.eigh(P)[1][:,1:];pred=[]
                for Q in z.Q:
                    Q0=Q-strength*np.diag(np.linspace(-1,1,f.d));dw=-tangent@np.linalg.solve(tangent.T@Q0@tangent,tangent.T@(Q-Q0)@w0);pred.append(w0+dw)
                local.append(dict(family=f.name,strength=strength,gamma=zz[(1<<f.M)-1],oracle_spread=float(np.max(np.linalg.norm(z.oracle-z.oracle[0],axis=1))),row_sum_residual=float(np.max(abs(H.sum(2)))),prediction_error=float(np.max(np.linalg.norm(z.oracle-np.array(pred),axis=1))),active_counts=((z.oracle<1e-8)|(z.oracle>z.upper-1e-8)).sum(1)))
        print('DIAGNOSE',f.name,flush=True)
    # Symmetric price geometry is a historical algebraic control, not confirmation.
    sym=[]
    for d in (3,6,10,20):
        for fraction in (.25,.75):
            f,L=analytic(d,fraction);gg=Geometry(f)
            for m in range(1,d+1):sym.append(dict(d=d,fraction=fraction,m=m,L0=L,gamma=gg.gamma((1<<m)-1)))
    write(OUT/'S0_diagnostics.json',out);write(OUT/'S0_hidden_ablation.json',local);write(OUT/'S0_symmetry.json',sym)
    active=[]
    for regime,diag in [('interior',np.ones(6)),('fixed_active',np.array([.25,.8,1,1.2,1.4,1.6])),('changing_active',np.ones(6))]:
        upper=.2;E=np.diag([-8,0,0,0,0,0]) if regime=='changing_active' else np.diag(np.linspace(-1,1,6))
        base=Family(np.array([np.diag(diag)]),upper,'active-base');w0=base.oracle[0];free=(w0>1e-8)&(w0<upper-1e-8)
        Q0=np.diag(diag);inds=np.flatnonzero(free);A=np.block([[Q0[np.ix_(inds,inds)],np.ones((len(inds),1))],[np.ones((1,len(inds))),np.zeros((1,1))]])
        dw=np.zeros(6);dw[inds]=np.linalg.solve(A,np.r_[-(E@w0)[inds],0])[:-1]
        for strength in (0,.01,.03,.08):
            z=Family(np.array([Q0+strength*E]),upper,'active-'+regime);w=z.oracle[0];same=np.array_equal((w>1e-8)&(w<upper-1e-8),free)
            active.append(dict(regime=regime,strength=strength,same_active_set=bool(same),prediction_error=float(np.linalg.norm(w-(w0+strength*dw))),weights=w,linear_term=float((w-w0)@Q0@w0)))
    write(OUT/'S0_active_geometry.json',active)
def prepare(split='development'):
    start=time.process_time()
    for spec in specs(split):
        check_budget(start,5400);f=make(*spec);save_family(f);geometry(f)
        for ratio,premium,allowed in itertools.product((.5,.75,1.25),(1,4),([1],[0,1,2,3])):setup(f,ratio,premium,allowed)
        print('PREPARED',f.name,flush=True)
def save_trajectory(path,r):
    # Complete action sequence + reproducible action-stream seed, cumulative
    # interval evidence at every exclusion, and stop certificate are lossless.
    acts=r.pop('actions');r['action_rle']=[[int(a),len(list(g))] for a,g in itertools.groupby(acts)]
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path.with_suffix(path.suffix+'.tmp'),'wt',encoding='utf8') as z:json.dump(clean(r),z,ensure_ascii=False,separators=(',',':'))
    path.with_suffix(path.suffix+'.tmp').replace(path)
def jobkey(f,truth,rep,ratio,delta,premium,allowed,policy):return f'{f.name}_t{truth}_r{rep}_e{ratio}_d{delta}_c{premium}_a'+''.join(map(str,allowed))+'_'+policy
def readtraj(path):
    with gzip.open(path,'rt',encoding='utf8') as z:return json.load(z)
def develop():
    start=time.process_time();timings=[];families=[make(*s) for s in specs('development')];cons={f.name:constants(f) for f in families};cache={}
    # Rotate all six families before advancing any development cell.
    for rep,truth,ratio,delta,premium,allowed in itertools.product(range(10),range(6),(.5,.75,1.25),(.05,.1,.01),(1,4),([0,1,2,3],[1])):
      for f in families:
        tk=(f.name,ratio,premium,tuple(allowed))
        if tk not in cache:cache[tk]=setup(f,ratio,premium,allowed)
        geo,tab=cache[tk]
        for policy in (('equal_cost',) if allowed==[1] else POLICIES):
            if time.process_time()-start>1800:
                write(OUT/'development_timing.json',dict(records=timings,censored=True));return
            key=jobkey(f,truth,rep,ratio,delta,premium,allowed,policy);path=RUN/'development'/f'{key}.json.gz'
            if path.exists():r=readtraj(path)
            else:
                r=simulate(f,geo,tab,cons[f.name],truth,rep,ratio,delta,premium,allowed,policy,stream_namespace='development');save_trajectory(path,r)
            timings.append(dict(key=key,cpu=r['cpu_seconds'],actions=sum(r['counts']),certified=r['certified'],kind=f.metadata['kind']))
        if len(timings)%100==0:print('DEVELOP',len(timings),flush=True)
    write(OUT/'development_timing.json',dict(records=timings,censored=False))
def freeze():
    if (ROOT/'manifest.json').exists():raise RuntimeError('freeze already exists')
    if not read(OUT/'gate0.json')['passed']:raise RuntimeError('gate required')
    t=read(OUT/'development_timing.json')['records'];p90=float(np.quantile([x['cpu'] for x in t],.9)) if t else np.inf
    measured=sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl') if r['stage']=='develop')
    p90=max(p90,measured/max(1,len(t)))  # includes serialization and shared setup
    # 12 families, 6 truths, 3 eps, 3 delta, 2 costs, 6 policy/design records.
    per_rep=12*6*3*3*2*6;remaining=min(18000,34560-used()-5760)
    chosen=50
    for n in (50,100,200):
        if p90*per_rep*n*1.25<=remaining:chosen=n
    cfg=dict(base=BASE,seed_namespace='v6-20261013',confirmation_repetitions=chosen,development_repetitions=10,delta=[.1,.05,.01],ratios=[.5,.75,1.25],premiums=[1,4],horizon_actions=100000,horizon_cost_units=100000,cost_unit='d scalars',policies=POLICIES,separated_equivalent_policies='one equal_cost representative; all policies have the sole available action',timing_p90=p90,estimated_confirmation_cpu=p90*per_rep*chosen*1.25,confirmation_allocation_seconds=18000,terminal_allocation_seconds=5760,core_hashes={k:v for k,v in sources().items() if k in ('models.py','certificates.py','information.py','likelihood.py','sequential.py')},bootstrap_seed=20261013,bootstrap_repetitions=10000)
    write(ROOT/'config_frozen.json',cfg)
    # Rotate structure, family and truth before advancing repeat, as prespecified.
    jobs=[]
    for rep in range(chosen):
      for truth in range(6):
       for spec in specs('confirmation'):
        name=f'{spec[0]}-{spec[1]}-6-confirmation-{spec[4]}'
        for ratio,delta,premium,allowed in itertools.product(cfg['ratios'],cfg['delta'],cfg['premiums'],([1],[0,1,2,3])):
         for policy in (('equal_cost',) if allowed==[1] else POLICIES):jobs.append(dict(family=name,truth=truth,rep=rep,ratio=ratio,delta=delta,premium=premium,allowed=allowed,policy=policy))
    write(ROOT/'manifest.json',jobs);write(OUT/'freeze_summary.json',dict(planned=len(jobs),config_sha256=sha(ROOT/'config_frozen.json'),manifest_sha256=sha(ROOT/'manifest.json'),timing_records=len(t)))
def confirm():
    cfg=read(ROOT/'config_frozen.json');revised=read(ROOT/'implementation_revision.json')['approved_core_hashes'] if (ROOT/'implementation_revision.json').exists() else cfg['core_hashes']
    for k,v in revised.items():
        if sha(ROOT/k)!=v:raise RuntimeError('frozen core changed: '+k)
    start=time.process_time();fs={};cons={};ts={};done=0
    # Stage resumptions share the same cumulative stage CPU allocation.
    prior=sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl') if r['stage']=='confirm')
    for j in read(ROOT/'manifest.json'):
        if time.process_time()-start+prior>18000 or used()+time.process_time()-start>28700:break
        name=j['family'];key=name+f"_t{j['truth']}_r{j['rep']}_e{j['ratio']}_d{j['delta']}_c{j['premium']}_a"+''.join(map(str,j['allowed']))+'_'+j['policy'];path=RUN/'confirmation'/f'{key}.json.gz'
        if path.exists():continue
        if name not in fs:fs[name]=load_family(name);cons[name]=constants(fs[name])
        f=fs[name];k=(name,j['ratio'],j['premium'],tuple(j['allowed']))
        if k not in ts:ts[k]=setup(f,j['ratio'],j['premium'],j['allowed'])
        geo,tab=ts[k]
        try:
            r=simulate(f,geo,tab,cons[name],j['truth'],j['rep'],j['ratio'],j['delta'],j['premium'],j['allowed'],j['policy']);save_trajectory(path,r)
        except Exception as e:write(RUN/'failures'/f'{key}.json',dict(job=j,error=repr(e)));raise
        done+=1
        if done%100==0:print('CONFIRM',done,'CPU',round(time.process_time()-start),flush=True)
    write(OUT/'confirmation_stage.json',dict(new_records=done,cpu=time.process_time()-start,stage_budget=18000))
def calibrate():
    start=time.process_time();zs=e1(.01);families=[Family(zs.Q,1.,'e1-calibration',dict(kind='simplex')),make('block',10,6,'development',0),base_control()]
    for f in families:
        con=constants(f);geo=geometry(f)
        for premium in (1,):
            _,tab=setup(f,.5,premium,[0,1,2,3])
            for rep in range(1000):
                for truth in range(f.M):
                    path=RUN/'calibration'/f'{f.name}-{truth}-{rep}.json.gz'
                    if path.exists():continue
                    check_budget(start,3600)
                    r=simulate(f,geo,tab,con,truth,rep,.5,.05,1,[0,1,2,3],'conflict_kl',stream_namespace='calibration');save_trajectory(path,r)
                if rep%100==0:print('CALIBRATE',f.name,rep,flush=True)
def dispatch(stage,*args):
    {'gate':gate,'diagnose':diagnose,'prepare':prepare,'develop':develop,'freeze':freeze,'confirm':confirm,'calibrate':calibrate,'cache-check':cache_check}[stage](*args)
def cache_check():
    from .stream_cache import verify
    checks=[]
    for i in range(20):
        f=make(KINDS[i%3],10,6,'development',0);checks.append(verify(f,constants(f),seed('cache-check',i)))
    write(OUT/'stream_cache_check.json',dict(passed=all(checks),cases=len(checks),scope='two reads of original and cached action streams; LL and rounding enclosures bitwise equal'))
