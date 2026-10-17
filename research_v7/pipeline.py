"""Fixed synthetic manifests, paired observations and conditional E3 gate."""
import time,traceback,itertools
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp
from research_v4a.mean_budget import Family
from research_v6.certificates import Geometry
from .common import *
from .models import make,specs,cases,observations,prefix
from .radius import calibrate,radius_bounds,oracle_bounds,envelope
from .decision import DecisionInput,solve
NGRID=(32,256,2048);RATIOS=(.5,1.25)

def prepare(split='development'):
    for kind,d,index in specs(split):
        f=make(kind,d,index,split);path=RUN/'families'/f'{f.name}.json'
        if path.exists():continue
        cc=cases(f);libraries={}
        for c in cc:
            key='all' if c['kind']!='leave_one' else c['id'];c['library']=key
            if key in libraries:continue
            ff=Family(c['Q'],.2,f'{f.name}-{key}');gg=Geometry(ff);g=gg.gamma((1<<ff.M)-1)
            libraries[key]=dict(Q=ff.Q,gamma=g,oracle=ff.oracle,oracle_lower=ff.VL,oracle_upper=ff.VU)
        write(path,dict(name=f.name,kind=kind,d=d,index=index,split=split,libraries=libraries,cases=cc))
        print('PREPARE',f.name,flush=True)

@lru_cache(32)
def load(name):
    raw=read(RUN/'families'/f'{name}.json');lib={};truths={}
    for key,a in raw['libraries'].items():
        f=Family(np.array(a['Q']),.2,name+'-'+key);g=Geometry(f);g.cache[(1<<f.M)-1]=a['gamma'];lib[key]=(f,g)
    for c in raw['cases']:truths[c['id']]=Family(np.array([c['truth']]),.2)
    return raw,lib,truths

def block_ll(X,Q,design):
    d=Q.shape[-1];n=len(X) if design=='Joint' else len(X)//2;ll=np.zeros((n,len(Q)))
    masks=[np.arange(d)] if design=='Joint' else [np.arange(d//2),np.arange(d//2,d)]
    for k,S in enumerate(masks):
        Y=X[:n,S] if design=='Joint' else X[k*n:(k+1)*n,S]
        for j,q in enumerate(Q):
            sub=q[np.ix_(S,S)];inv=np.linalg.inv(sub);ld=np.linalg.slogdet(sub)[1]
            ll[:,j]-=.5*(len(S)*np.log(2*np.pi)+ld+np.einsum('ni,ij,nj->n',Y,inv,Y))
    return ll

def manifest(split,repetitions):
    names=[f'{k}-{d}-{split}-{i}' for k,d,i in specs(split)]
    # Round-robin across family before moving to the next truth or replicate.
    for rep,case_index,design in itertools.product(range(repetitions),range(22),('Separated','Joint')):
      for name in names:
        yield dict(family=name,case_index=case_index,rep=rep,design=design)

def jobkey(j):return f"{j['family']}_c{j['case_index']}_r{j['rep']}_{j['design']}"

@lru_cache(128)
def prior_decisions(name,key,ratio):
    raw,lib,_=load(name);f,g=lib[key];eps=raw['libraries'][key]['gamma']['upper']*ratio;p=np.ones(f.M)/f.M
    return solve(DecisionInput(f.Q,p,np.log(p),f.upper,eps),family=f)

def run_job(j,split):
    tick=time.process_time();raw,lib,truths=load(j['family']);case=raw['cases'][j['case_index']];f,geom=lib[case['library']];true=truths[case['id']];Qstar=true.Q[0]
    g=raw['libraries'][case['library']]['gamma'];w=np.array(g['weights']);reglo,reghi=true.regret_bounds(w);rows=[];term=[]
    # Two independent streams; neither calibration observations nor truth enter decisions.
    sx=seed(j['family'],case['id'],j['rep'],j['design'],'pilot');sy=seed(j['family'],case['id'],j['rep'],j['design'],'calibration')
    X=observations(Qstar,1024,j['design'],sx);Y=observations(Qstar,1024,j['design'],sy)
    lpx=block_ll(X,f.Q,j['design']);lpy=block_ll(Y,f.Q,j['design']);cum=np.cumsum(lpx,axis=0)
    loge=logsumexp(cum,axis=1)[:,None]-np.log(f.M)-cum
    for n in NGRID:
        m=n//2;lp=cum[m-1]-logsumexp(cum[m-1]);p=np.exp(lp);cal=calibrate(prefix(Y,m,j['design']),j['design'])
        rho=radius_bounds(cal,f.Q,p,f.upper);oracle=oracle_bounds(Qstar,f.Q,p,f.upper)
        cs=np.flatnonzero(np.max(loge[:m],axis=0)<np.log(20));cs_gamma=geom.gamma(sum(1<<int(i) for i in cs)) if len(cs) else None
        # Naive finite-library confidence set is reported without a misspecified guarantee.
        if cs_gamma is not None:cl,ch=true.regret_bounds(np.array(cs_gamma['weights']))
        for ratio in RATIOS:
            eps=g['upper']*ratio
            for method,r in [('library',0.),('oracle_decision',oracle['upper']['upper']),('oracle_operator',oracle['operator']['upper']),('data_decision',rho['decision']['upper']),('data_operator',rho['operator']['upper'])]:
                cert=bool(g['upper']+r<=eps);fail=bool(cert and reglo[0]>eps);unknown=bool(cert and reglo[0]<=eps<reghi[0])
                rows.append(dict(n=n,ratio=ratio,method=method,epsilon=eps,rho=r,gamma_lower=g['lower'],gamma_upper=g['upper'],certified=cert,false_certification=fail,truth_indeterminate=unknown,truth_regret_lower=reglo[0],truth_regret_upper=reghi[0],confidence_event=cal.encloses(Qstar),weights=w,
                    numerical_qualification='rational_library_geometry_plus_outward_float_radius',scalars=n*f.d,independent_times=n*(1 if j['design']=='Joint' else 2),cost=n*f.d,rho_witness=rho.get(method.removeprefix('data_'),{}).get('witness')))
            rows.append(dict(n=n,ratio=ratio,method='v6_pilot_CS',epsilon=eps,certified=bool(cs_gamma and cs_gamma['upper']<=eps),false_certification=bool(cs_gamma and cs_gamma['upper']<=eps and cl[0]>eps),truth_indeterminate=bool(cs_gamma and cs_gamma['upper']<=eps and cl[0]<=eps<ch[0]),confidence_set=cs.tolist(),gamma_upper=None if cs_gamma is None else cs_gamma['upper'],truth_regret_upper=None if cs_gamma is None else ch[0],weights=None if cs_gamma is None else cs_gamma['weights'],cost=n*f.d,guarantee='correct finite family only'))
        if j['rep']<10:
            lp_full=cum[m-1]+lpy[:m].sum(0);lp_full-=logsumexp(lp_full);pfull=np.exp(lp_full)
            for ratio in RATIOS:
                eps=g['upper']*ratio;ans=solve(DecisionInput(f.Q,p,lp,f.upper,eps),family=f)
                mafull=f.mean(pfull)['w'];ans.append(dict(method='MA_full_data',beta=0.,weights=mafull,budget_applies=False))
                for a in ans:
                    a=dict(a,n=n,ratio=ratio,epsilon=eps)
                    if 'weights' in a:
                        tl,tu=true.regret_bounds(np.array(a['weights']));a.update(truth_regret_lower=tl[0],truth_regret_upper=tu[0],true_success_lower=bool(tu[0]<=eps),true_success_upper=bool(tl[0]<=eps))
                        if 'max_regret_upper' in a:a['external_certified']=bool(a['max_regret_upper']+rho['decision']['upper']<=eps)
                    term.append(a)
            if j['rep']==0 and j['design']=='Separated':
                for ratio in RATIOS:
                    for a in prior_decisions(j['family'],case['library'],ratio):
                        a=dict(a,n=0,ratio=ratio,epsilon=g['upper']*ratio)
                        if 'weights' in a:
                            tl,tu=true.regret_bounds(np.array(a['weights']));a.update(truth_regret_lower=tl[0],truth_regret_upper=tu[0],true_success_lower=bool(tu[0]<=a['epsilon']),true_success_upper=bool(tl[0]<=a['epsilon']))
                        if not any(t['n']==0 and t['ratio']==ratio and t['method']==a['method'] and t['beta']==a['beta'] for t in term):term.append(a)
    return dict(job=j,kind=raw['kind'],case=case['id'],panel=case['panel'],perturbation=case['kind'],strength=case['strength'],status='complete',rows=rows,terminal=term,pilot_seed=sx,calibration_seed=sy,log_likelihood_pilot=cum[[15,127,1023]],log_likelihood_calibration=np.cumsum(lpy,axis=0)[[15,127,1023]],cpu=time.process_time()-tick)

def batch(split,reps,limit):
    tick=time.process_time();count=0;cost=[]
    for j in manifest(split,reps):
        path=RUN/split/f'{jobkey(j)}.json.gz'
        if path.exists():continue
        if time.process_time()-tick>limit or used()+time.process_time()-tick>34400:break
        start=time.process_time()
        try:r=run_job(j,split)
        except Exception:r=dict(job=j,status='failed',error=traceback.format_exc(),cpu=time.process_time()-start)
        write(path,r);count+=1;cost.append(r['cpu'])
        if count%16==0:print(split,count,round(time.process_time()-tick,1),r['status'],flush=True)
    write(OUT/f'{split}_run.json',dict(new=count,cpu=time.process_time()-tick,costs=cost))

def develop():batch('development',10,3400)
def freeze():
    if (ROOT/'config_frozen.json').exists():raise RuntimeError('already frozen')
    records=[read(p) for p in (RUN/'development').glob('*.json.gz')];cost=[r['cpu'] for r in records];p90=float(np.quantile(cost,.9));count_base=8*22*2
    if not read(OUT/'gate0.json')['passed']:raise RuntimeError('correctness gate not passed')
    if not read(OUT/'parallel_smoke.json')['passed']:raise RuntimeError('parallel reproducibility not verified')
    if len(records)!=len(list(manifest('development',10))) or any(r['status']!='complete' for r in records):raise RuntimeError('development incomplete or contains failed datasets')
    if any('error' in a for r in records for a in r['terminal']):raise RuntimeError('unresolved terminal implementation errors')
    # Includes expensive first-ten terminal decisions; later records cheaper.
    estimates={r:p90*count_base*r*1.25 for r in (20,50,100)}
    reps=max([r for r,v in estimates.items() if v<=16560] or [20]);jobs=list(manifest('confirmation',reps))
    write(ROOT/'manifest.json.gz',jobs);write(ROOT/'config_frozen.json',dict(base=BASE,namespace='v7-20261017',repetitions=reps,n=NGRID,ratios=RATIOS,delta=.05,terminal_repetitions=10,p90=p90,estimates=estimates,minimum_tier_over_budget=estimates[20]>16560,workers=2,core_hashes={k:v for k,v in sources().items() if k in ('models.py','radius.py','decision.py','pipeline.py','executor.py','experiments.py','supervisor.py','common.py')},manifest_sha256=sha(ROOT/'manifest.json.gz'),proposal_sha256=sha(ROOT/'INPUT_PROPOSAL_ZH.md')))
    print('FROZEN',reps,'datasets',len(jobs),'estimate',estimates[reps],flush=True)
def confirm():
    cfg=read(ROOT/'config_frozen.json')
    for name,h in cfg['core_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen source changed '+name)
    from .executor import batch as parallel_batch
    prior=sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl') if r['stage']=='confirm');parallel_batch(cfg['repetitions'],max(0,16560-prior))
def e3():
    from .real_extension import run
    run()
def dispatch(stage,*args):globals()[stage](*args)
