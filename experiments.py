"""Reproduce gates, S1, timing pilot, paired S2 and small S3, in that order."""
from core import *
from pathlib import Path
import json, sys, platform, hashlib, argparse
import pandas as pd
from scipy.linalg import null_space
from scipy.stats import beta, t as student_t
import psutil

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SEED=20261008

def write_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)),encoding='utf-8')

def gate0():
    rng=np.random.default_rng(SEED); checks={}; maxgap=0.; max_violation=0.
    for d in (3,6,20):
        for i in range(5):
            A=rng.normal(size=(d,d)); Q=A@A.T/d+np.eye(d)*.4; mu=rng.normal(size=d)*.1
            ans=solve_qp(Q,mu,{'upper':max(.2,2/d)})
            x=cp.Variable(d); prob=cp.Problem(cp.Minimize(.5*cp.quad_form(x,Q)-mu@x),[cp.sum(x)==1,x>=0,x<=max(.2,2/d)])
            prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11)
            assert ans['value_lb']-2e-9<=prob.value<=ans['value_ub']+2e-9
            assert ans['primal_residual']<1e-10
            maxgap=max(maxgap,ans['gap'])
    checks['qp_bounds']={'passed':True,'max_gap':maxgap,'cases':15}
    # Degenerate vertex: no free variable determines the budget multiplier.
    z=np.array([1.,1.2,1.4,1.4,1.4,1.4]); diagonal=np.array([1.,3.,4.,4.,4.,4.])
    vertexQ=np.outer(z,z)+np.diag(diagonal-z*z)
    vertex=solve_qp(vertexQ)
    assert vertex['gap']<1e-8
    assert np.linalg.norm(vertex['w']-np.eye(6)[0])<1e-7
    checks['vertex_qp_bounds']={'passed':True,'gap':vertex['gap'],'lower':vertex['value_lb'],'upper':vertex['value_ub']}
    # Explicit L1 dual bound and tangent-only strong convexity reduced coordinates.
    Q=np.eye(4); mu=np.array([.5,0,-.2,.3]); b=np.ones(4)/4
    ans=solve_qp(Q,mu,kappa=.1,b=b)
    assert ans['gap']<1e-7
    checks['l1_dual']={'passed':True,'gap':ans['gap']}
    fam=make_family(6); Qs=[fam.Q(x) for x in fam.vertices()]
    ctr=center(Qs,tol=1e-8); ref,_=direct_center(Qs)
    assert ctr['L']-1e-8<=ref<=ctr['U']+1e-8
    assert ctr['gap']<1e-7
    checks['center_dual_qcqp']={'passed':True,'dual_lower':ctr['L'],'primal_upper':ctr['U'],'reference':ref,'gap':ctr['gap']}
    q1=np.array([[1.,-.5],[-.5,2.]]); q2=np.array([[1.,.5],[.5,2.]])
    Y=sample_data(q1,30,0,rng); like=MaskLikelihood(Y)
    assert abs(like(q1)-like(q2))<1e-12
    joint=MaskLikelihood(sample_data(q1,30,30,rng))
    assert abs(joint(q1)-joint(q2))>1e-6
    checks['observation_equivalence']={'passed':True,'marginal_difference':like(q1)-like(q2),'joint_difference':joint(q1)-joint(q2)}
    # Grouped likelihood versus row-wise reference including empty rows.
    ll=0.
    for row in Y:
        ids=np.flatnonzero(np.isfinite(row))
        if len(ids):
            S=q1[np.ix_(ids,ids)]; ll-=.5*(len(ids)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+row[ids]@np.linalg.solve(S,row[ids]))
    assert abs(ll-like(q1))<1e-10
    checks['mask_grouping']={'passed':True,'difference':ll-like(q1)}
    # Off-grid likelihood upper bounds and outer-set containment (conditional on threshold).
    true=np.array([.137,-.431,.719]); Y=sample_data(fam.Q(true),64,16,rng)
    result,_,like=lcrc_affine(fam,Y,max_cells=16)
    for lo,hi,ll,ub in result.diagnostics['cell_bounds']:
        for _ in range(10):
            x=rng.uniform(lo,hi); max_violation=max(max_violation,like(fam.Q(x))-ub)
    assert max_violation<1e-8
    retained=any(np.all(true>=lo)&np.all(true<=hi) for lo,hi in result.diagnostics['cells'])
    assert like(fam.Q(true))<result.diagnostics['threshold'] or retained
    checks['continuous_cover']={'passed':True,'max_sampled_likelihood_violation':max_violation,'true_retained':retained,'off_grid':True}
    # T2 bound: use an upper bound on the decision gradient metric over the simplex.
    worst=0.
    for _ in range(100):
        q=fam.Q(rng.uniform(-1,1,3)); r=fam.Q(rng.uniform(-1,1,3)); w=rng.dirichlet(np.ones(6))
        P=np.eye(6)-np.ones((6,6))/6; metric=max(np.linalg.norm(P@(q-r)[:,i]) for i in range(6))
        alpha=min(np.linalg.eigvalsh(q)[0],np.linalg.eigvalsh(r)[0])
        vq=solve_qp(q)['value_ub']; vr=solve_qp(r)['value_ub']
        lhs=abs(np.sqrt(max(0,.5*w@q@w-vq))-np.sqrt(max(0,.5*w@r@w-vr)))
        worst=max(worst,lhs-metric/np.sqrt(2*alpha))
    assert worst<1e-8
    checks['sqrt_regret_stability']={'passed':True,'cases':100,'max_violation':worst}
    # Finite-mode actual decisions, not only density inequalities.
    Qs=[fam.Q(x) for x in fam.vertices()]; counts=np.zeros(2,int)
    for i in range(120):
        j=i%len(Qs); Y=sample_data(Qs[j],24,8,rng); ans,keep=lcrc_finite(Qs,Y)
        counts[0]+=int(keep[j]); regret=.5*ans['w']@Qs[j]@ans['w']-solve_qp(Qs[j])['value_ub']
        counts[1]+=int(regret<=ans['U']+1e-9)
    checks['finite_mode_coverage']={'passed':True,'n':120,'truth_retained':int(counts[0]),'regret_covered':int(counts[1]),'note':'Monte Carlo diagnostic, not a proof of 95% coverage'}
    write_json('gate0.json',checks); print('Gate 0 passed',flush=True)


def local_center(Qs,es):
    d=len(Qs[0]); U=null_space(np.ones((1,d))); Ks=[U.T@q@U for q in Qs]; rs=[U.T@e for e in es]
    phi=[.5*r@np.linalg.solve(K,r) for K,r in zip(Ks,rs)]
    z=cp.Variable(d-1); t=cp.Variable()
    prob=cp.Problem(cp.Minimize(t),[.5*cp.quad_form(z,K)-r@z+p<=t for K,r,p in zip(Ks,rs,phi)])
    prob.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_feas=1e-12,tol_gap_rel=1e-12)
    return float(prob.value),U@z.value


def s1():
    rows=[]
    for d in (6,20):
        for seed in range(5):
            rng=np.random.default_rng(SEED+seed); perm=rng.permutation(d); w0=np.ones(d)/d
            vs=np.sqrt(d/(d-1))*(np.eye(d)-np.ones((d,d))/d)
            Hs=[]
            for v in vs:
                H=d/(d-2)*(v[:,None]+v[None,:]); np.fill_diagonal(H,0); Hs.append(H)
            # Seed permutations are invariance checks, not independent statistical evidence.
            Hs=np.array(Hs)[:,perm][:,:,perm]
            pairs=[(a,b) for a in range(d) for b in range(a+1,d)]
            incidence=np.zeros((d,len(pairs)))
            for idx,(a,b) in enumerate(pairs): incidence[a,idx]=incidence[b,idx]=1
            N=null_space(incidence); hidden=[]
            for j in range(3):
                vals=N@rng.normal(size=N.shape[1]); H=np.zeros((d,d))
                for val,(a,b) in zip(vals,pairs): H[a,b]=H[b,a]=val
                H*=.35/np.linalg.norm(H,2); hidden.append(np.eye(d)+H)
            E=np.diag(rng.uniform(-1,1,d)); es=[-E@w0]*3
            psi_hidden,_=local_center(hidden,es)
            # Factor-plus-diagonal construction, same marginal variances, different slack.
            z1=np.r_[1.,1.,np.full(d-2,1.4)]; z2=z1.copy(); z2[1]=1.2
            diagonal=np.r_[1.,3.+.1*seed,np.full(d-2,4.)]
            boundary=[np.outer(z,z)+np.diag(diagonal-z*z) for z in (z1,z2)]
            EB=np.zeros((d,d)); EB[0,1]=EB[1,0]=-1
            psi_boundary=.5/(diagonal[1]-1)
            for mechanism,Qs,Es,psi in [
                ('compatibility',[np.eye(d)]*d,list(Hs),.5),
                ('hidden_curvature',hidden,[E]*3,psi_hidden),
                ('boundary',boundary,[EB]*2,psi_boundary)]:
                cap=min(.04,.15*min(np.linalg.eigvalsh(q)[0]/max(np.linalg.norm(e,2),1e-12) for q,e in zip(Qs,Es)))
                for ratio in (1.,.5,.25,.125):
                    t=cap*ratio; qt=np.array([q+t*e for q,e in zip(Qs,Es)])
                    ans=center(qt,tol=2e-10,maxiter=200)
                    ref,wr=direct_center(qt)
                    assert ans['L']-2e-8<=ref<=ans['U']+2e-8
                    oracles=[solve_qp(q) for q in qt]
                    gamma=(ans['U']+ans['L'])/2
                    pair_gamma=direct_center(qt[:2])[0] if mechanism=='compatibility' else np.nan
                    rows.append(dict(mechanism=mechanism,d=d,seed=seed,t=t,psi=psi,gamma=gamma,
                        ratio=gamma/t**2,scaled_remainder=abs(gamma-t*t*psi)/t**3,
                        gap=ans['gap'],reference=ref,pair_ratio=pair_gamma/t**2,
                        expected_pair_ratio=d/(4*(d-1)) if mechanism=='compatibility' else np.nan,
                        mixture_support=int(np.sum(ans['pi']>1e-5)),
                        supports=json.dumps([np.flatnonzero(o['w']>1e-6).tolist() for o in oracles]),
                        max_oracle_distance=max(float(np.linalg.norm(ans['w']-o['w'])) for o in oracles),
                        current_diagonal_equality=float(np.max(np.ptp(np.diagonal(qt,axis1=1,axis2=2),axis=0)))))
            print(f'S1 d={d} seed={seed}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'s1.csv',index=False)
    last=df.sort_values('t').groupby(['mechanism','d','seed']).first()
    # Compare leading terms accounting explicitly for finite t and solver error.
    # An open numerical interval is a failure, not an excuse for any coefficient.
    assert np.all(last['gap']/last['t']**2 < .02*np.maximum(last['psi'],.001))
    assert np.all(abs(last['ratio']-last['psi']) < .12*np.maximum(last['psi'],.001)+2e-5)
    assert df.current_diagonal_equality.max()<1e-10
    write_json('gate1.json',{'passed':True,'cases':len(df),'note':'Finite-t checks, not mathematical proofs; permutations in compatibility are not independent markets.'})
    print('Gate 1 passed',flush=True)


def tune():
    fam=make_family(); rng=np.random.default_rng(SEED+10000); scores={a:[] for a in (0,.1,.25,.5,.75,1.)}; units=[]
    for i in range(48):
        theta=rng.uniform(-1,1,3); Q=fam.Q(theta); Y=sample_data(Q,64,(0,4,16,64)[i%4],rng)
        em,_=em_cov(Y); future=rng.normal(size=(21,20))@np.linalg.cholesky(Q).T; S=future.T@future/21
        units.append(.5*np.ones(20)@em@np.ones(20)/400)
        if i>=24:
            for a in scores:
                q=(1-a)*em+a*np.diag(np.diag(em)); w=solve_qp(q, constraints={'upper':.2})['w']
                scores[a].append(.5*w@S@w)
    best=min(scores,key=lambda a:np.mean(scores[a])); unit=float(np.median(units[:24]))
    data={'em_shrinkage':best,'risk_unit':unit,'validation_costs':{str(a):float(np.mean(v)) for a,v in scores.items()},
          'train_instances':24,'validation_instances':24,'seed':SEED+10000,'future_label_rows':21,
          'label_mean':'known zero, uncentered second moment','source':'independent development instances'}
    write_json('tuning.json',data); return data


def instance_methods(fam,Y,unit,shrink,max_cells=32):
    results={}; upper=.2; tic=perf_counter()
    lcrc,Qs,like=lcrc_affine(fam,Y,upper=upper,max_cells=max_cells,risk_unit=unit)
    results['LCRC']=lcrc
    tic=perf_counter(); x,diag=affine_mle(fam,like); a=solve_qp(fam.Q(x),constraints={'upper':upper})
    results['Affine-MLE']=DecisionResult(a['w'],fit_seconds=perf_counter()-tic,diagnostics=diag)
    tic=perf_counter(); q,diag=em_cov(Y); q=(1-shrink)*q+shrink*np.diag(np.diag(q))
    a=solve_qp(q,constraints={'upper':upper})
    results['EM-shrink']=DecisionResult(a['w'],fit_seconds=perf_counter()-tic,diagnostics=diag)
    tic=perf_counter(); q=model_average(fam,like); a=solve_qp(q,constraints={'upper':upper})
    results['Model-average']=DecisionResult(a['w'],fit_seconds=perf_counter()-tic)
    tic=perf_counter(); ans=center([fam.Q(v) for v in fam.vertices()],upper=upper,tol=unit*1e-4)
    results['Full-set-regret']=DecisionResult(ans['w'],solve_seconds=perf_counter()-tic,numerical_gap=ans['gap'],qp_calls=ans['qp_calls'])
    tic=perf_counter(); ans=center(Qs,upper=upper,absolute=True,tol=unit*1e-4)
    results['Likelihood-absolute']=DecisionResult(ans['w'],solve_seconds=perf_counter()-tic,
        likelihood_seconds=lcrc.likelihood_seconds,numerical_gap=ans['gap'],qp_calls=ans['qp_calls'])
    results['Equal-weight']=DecisionResult(np.ones(len(fam.base))/len(fam.base))
    return results


def one(stage,n,m,rep,tuning,variant='base',cells=32):
    seed=SEED+100000+(0 if stage=='S2' else 1000000 if stage=='S3' else 2000000)+n*1000+m*30+rep
    rng=np.random.default_rng(seed); fam=make_family(); truth=rng.uniform(-1,1,3); Q=fam.Q(truth)
    if variant=='wide': fam.low*=1.3; fam.high*=1.3
    if variant=='cov_misspec':
        H=rng.normal(size=Q.shape); H=(H+H.T)/2; Q+=.12*H/np.linalg.norm(H,2)
    Y=sample_data(Q,n,m,rng,'t6' if variant=='t6' else 'gaussian')
    tic=perf_counter(); methods=instance_methods(fam,Y,tuning['risk_unit'],tuning['em_shrinkage'],cells)
    total=perf_counter()-tic; rows=[]; oracle=solve_qp(Q,constraints={'upper':.2})
    for name,res in methods.items():
        w=res.weights; regret=max(0.,.5*w@Q@w-oracle['value_ub']); diag=res.diagnostics
        true_retained=None
        if name=='LCRC' and variant not in ('cov_misspec','t6'):
            true_retained=any(np.all(truth>=lo)&np.all(truth<=hi) for lo,hi in diag['cells'])
        row=dict(stage=stage,variant=variant,n_blocks=n,joint_blocks=m,rep=rep,seed=seed,method=name,
            observed_scalar_count=int(np.isfinite(Y).sum()),d=20,p=3,max_cells=cells,
            theta0=truth[0],theta1=truth[1],theta2=truth[2],parameter_subgroup=int(truth[1]>0)*2+int(truth[2]>0),
            regret_raw=regret,regret_scaled=regret/tuning['risk_unit'],objective_value=.5*w@Q@w,
            certificate_upper=res.risk_upper,certificate_lower=res.risk_lower,
            covered=None if res.risk_upper is None else regret<=res.risk_upper+1e-9,
            true_retained=true_retained,confidence_delta=.05 if name=='LCRC' else None,
            certificate_scope=('empirical_only' if variant in ('cov_misspec','t6') else res.certificate_scope),
            numerical_gap=res.numerical_gap,solve_status=res.solve_status,
            fit_seconds=res.fit_seconds,likelihood_seconds=res.likelihood_seconds,solve_seconds=res.solve_seconds,
            total_seconds=res.fit_seconds+res.likelihood_seconds+res.solve_seconds,
            instance_seconds=total,qp_calls=res.qp_calls,scenario_count=res.scenario_count,cell_count=res.cell_count,
            volume_fraction=diag.get('volume_fraction'),refinement_gap=diag.get('refinement_gap'),
            max_weight=w.max(),primal_residual=max(abs(w.sum()-1),-w.min(),max(w)-.2,0),
            model_misspecified=variant in ('cov_misspec','t6'),failure_reason='',future_data_access_check=True,
            oracle_scope='synthetic_population',weights=json.dumps(w.tolist()),diagnostics=json.dumps(diag))
        for frac in (.05,.1,.2):
            row[f'success_{frac}']=regret<=frac*tuning['risk_unit']
            row[f'certified_{frac}']=None if res.risk_upper is None else res.risk_upper<=frac*tuning['risk_unit']
        rows.append(row)
    return rows


def run_grid(stage,tuning):
    rows=[]; errors=[]; specs=[]
    if stage=='pilot': specs=[(64,16,i,'base',32) for i in range(10)]
    elif stage=='S2': specs=[(n,m,i,'base',32) for n in (64,256) for m in (0,4,16,64) for i in range(20)]
    else:
        specs=[(64,16,i,v,c) for v,c in [('base',8),('base',32),('base',128),('cov_misspec',32),('t6',32),('wide',32)] for i in range(20)]
        specs += [(64,m,i,v,32) for m,v in [(0,'weak'),(64,'full')] for i in range(20)]
    for idx,(n,m,i,v,c) in enumerate(specs):
        try: rows.extend(one(stage,n,m,i,tuning,v,c))
        except Exception as e:
            errors.append(dict(stage=stage,n=n,m=m,rep=i,variant=v,cells=c,error=repr(e)))
        if (idx+1)%5==0:
            pd.DataFrame(rows).to_parquet(OUT/f'{stage}_instances.parquet',index=False)
            write_json(f'{stage}_errors.json',errors)
            print(f'{stage}: {idx+1}/{len(specs)}; failures={len(errors)}',flush=True)
    df=pd.DataFrame(rows); df.to_parquet(OUT/f'{stage}_instances.parquet',index=False)
    write_json(f'{stage}_errors.json',errors)
    if stage=='pilot':
        times=df.groupby('rep').instance_seconds.first().to_numpy()
        memory=psutil.Process().memory_info()
        write_json('pilot_timing.json',dict(instances=len(times),median_seconds=np.median(times),p95_seconds=np.quantile(times,.95),
            process_peak_working_set_bytes=getattr(memory,'peak_wset',memory.rss),note='Whole Python process, including imports and all seven methods; one BLAS thread.'))
    return df


def summarize():
    summaries=[]; paired=[]
    for stage in ('S2','S3'):
        df=pd.read_parquet(OUT/f'{stage}_instances.parquet')
        groupcols=['stage','variant','n_blocks','joint_blocks','max_cells']
        for keys,g in df.groupby(groupcols+['method']):
            row=dict(zip(groupcols+['method'],keys)); x=g.regret_scaled.to_numpy(); n=len(x)
            row.update(n=n,mean=x.mean(),median=np.median(x),q90=np.quantile(x,.9),se=x.std(ddof=1)/np.sqrt(n),
                worst_subgroup_mean=g.groupby('parameter_subgroup').regret_scaled.mean().max(),
                median_seconds=g.total_seconds.median(),p95_seconds=g.total_seconds.quantile(.95),
                success_rate=g['success_0.1'].mean(),certificate_rate=g['certified_0.1'].dropna().mean(),
                volume_fraction=g.volume_fraction.mean(),gap_open_rate=np.mean(g.solve_status=='gap_open'))
            cover=g.covered.dropna().astype(int)
            if len(cover):
                s=int(cover.sum()); nn=len(cover)
                row.update(coverage=s/nn,coverage_low=0 if s==0 else beta.ppf(.025,s,nn-s+1),
                    coverage_high=1 if s==nn else beta.ppf(.975,s+1,nn-s),
                    mean_certificate_scaled=g.certificate_upper.mean()/json.loads((OUT/'tuning.json').read_text())['risk_unit'])
            summaries.append(row)
        for keys,g in df.groupby(groupcols):
            pivot=g.pivot(index='rep',columns='method',values='regret_scaled')
            for baseline in pivot.columns:
                if baseline=='LCRC': continue
                diff=(pivot['LCRC']-pivot[baseline]).dropna(); n=len(diff)
                half=student_t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n)
                paired.append(dict(zip(groupcols,keys))|dict(baseline=baseline,n=n,difference=diff.mean(),low=diff.mean()-half,high=diff.mean()+half))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['gate0','s1','tune','pilot','S2','S3','summarize'])
    args=parser.parse_args()
    if args.stage=='gate0': gate0()
    elif args.stage=='s1': s1()
    elif args.stage=='tune': tune()
    elif args.stage=='summarize': summarize()
    else:
        assert json.loads((OUT/'gate0.json').read_text())['qp_bounds']['passed']
        assert json.loads((OUT/'gate1.json').read_text())['passed']
        run_grid(args.stage,json.loads((OUT/'tuning.json').read_text()))
