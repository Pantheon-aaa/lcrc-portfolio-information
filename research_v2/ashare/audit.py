"""Same-scene objective comparisons, leakage checks, and independently chosen audits."""
import numpy as np
import pandas as pd
from .common import PRIVATE,ROOT,read,dump
from .run import scene,tasks
from .data import Data,select
from ..scenario_inference import posterior_from_loglik
from ..risk_envelopes import Envelope,RiskSpec
from ..risk_center import solve_center,conic_reference

def temporal():
    data=Data(); rows=read(PRIVATE/'manifest_all.json');report={}
    assert all(r['t']>=251 and data.dates[r['t']]==r['date'] for r in rows)
    val=tasks('scores'); assert all(t['row']['evaluation_end']<='2019-12-31' for t in val)
    assert all(t['row']['date']>='2020-01-01' for t in tasks('test'))
    # A data facade denies any read beyond decision time, including eligibility fields.
    class PrefixArray:
        def __init__(self,a,t):self.a=a;self.t=t
        def __getitem__(self,index):
            key=index[0] if isinstance(index,tuple) else index
            if isinstance(key,slice):assert key.stop is not None and key.stop<=self.t+1
            else:assert int(key)<=self.t
            return self.a[index]
    for cohort in ('long','limited'):
        row=next(r for r in rows if r['cohort']==cohort and r['d']==20)
        t=row['t']; D=Data();D.a={k:PrefixArray(v,t) for k,v in D.a.items()}
        a,diag=select(D,t,cohort,0,d=20)
        assert a==row['ids']
        assert D.past(t,a).shape==(252,20)
    report.update(prefix_read_guard=True,validation_end_bound=True,test_date_bound=True,
        validation_windows=len(val),test_windows=len(tasks('test')),
        overlapping_validation_end_excluded=[r['key'] for r in rows if r['d']==20 and r['period']=='validation' and r['evaluation_end']>'2019-12-31'])
    dump(ROOT/'output/temporal_gate.json',report);print(report)

def integration():
    cfg=read(ROOT/'configs/base.json');choice=read(ROOT/'configs/selected.json');rows={r['key']:r for r in read(PRIVATE/'manifest_all.json')}
    records=[read(p) for p in (PRIVATE/'records/audit').glob('*.json')];out=[]
    for key in sorted(set(r['key'] for r in records)):
        group=[r for r in records if r['key']==key]
        if len(group)!=9:continue
        reference=next(r for r in group if r['size']==512 and r['scramble']==0)
        sc,ll,nom,mean,Y,diag=scene(rows[key],512,0)
        post=posterior_from_loglik(sc,ll,power=choice['power'])
        base={a['method']:a for a in reference['methods']}
        p_ref=np.einsum('m,mij->ij',post.p,sc.Q)
        for r in group:
            sc2,ll2,*_=scene(rows[key],r['size'],r['scramble'])
            p2=posterior_from_loglik(sc2,ll2,power=choice['power'])
            matrixdiff=float(np.linalg.norm(np.einsum('m,mij->ij',p2.p,sc2.Q)-p_ref)/np.linalg.norm(p_ref))
            for a in r['methods']:
                method=a['method']
                if method not in ('MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25') or not a.get('weights'):continue
                spec=RiskSpec(method,'mean') if method=='MA' else RiskSpec(method,rho=.25 if method.endswith('0.25') else .5,conditional=method.startswith('B'))
                env=Envelope(sc,post,spec,cfg['risk_unit']);w=np.array(a['weights']);wref=np.array(base[method]['weights'])
                diff=env.support(sc.costs(w)-sc.oracle_lower)[0]-env.support(sc.costs(wref)-sc.oracle_lower)[0]
                out.append(dict(key=key,cohort=r['cohort'],method=method,size=r['size'],scramble=r['scramble'],
                    l1_weight=float(abs(w-wref).sum()),common_objective_difference_unit=float(diff/cfg['risk_unit']),
                    covariance_relative_difference=matrixdiff,ess=p2.diagnostics['ess'],gap=a.get('gap'),
                    realized_risk_difference=a['evaluation']['second_moment']-base[method]['evaluation']['second_moment']))
    frame=pd.DataFrame(out);frame.to_csv(PRIVATE/'evaluation/integration_instances.csv',index=False)
    frame.groupby(['cohort','method','size']).agg(weight_l1_median=('l1_weight','median'),weight_l1_max=('l1_weight','max'),
        common_objective_abs_max=('common_objective_difference_unit',lambda x:abs(x).max()),
        covariance_relative_median=('covariance_relative_difference','median'),ess_median=('ess','median'),
        realized_risk_abs_max=('realized_risk_difference',lambda x:abs(x).max())).to_csv(ROOT/'output/integration.csv')
    within=[];contrasts=[]
    for key in sorted(set(r['key'] for r in records)):
      group=[r for r in records if r['key']==key]
      if len(group)!=9:continue
      for scramble in range(3):
        ref=next(r for r in group if r['size']==512 and r['scramble']==scramble)
        base={a['method']:a for a in ref['methods']}
        sc,ll,*_=scene(rows[key],512,scramble);post=posterior_from_loglik(sc,ll,power=choice['power'])
        for r in [z for z in group if z['scramble']==scramble]:
          amap={a['method']:a for a in r['methods']}
          for method in ('MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25'):
            a=amap[method];w=np.array(a['weights']);v=np.array(base[method]['weights'])
            spec=RiskSpec(method,'mean') if method=='MA' else RiskSpec(method,rho=.25 if method.endswith('0.25') else .5,conditional=method.startswith('B'))
            env=Envelope(sc,post,spec,cfg['risk_unit'])
            delta=env.support(sc.costs(w)-sc.oracle_lower)[0]-env.support(sc.costs(v)-sc.oracle_lower)[0]
            within.append(dict(cohort=r['cohort'],method=method,size=r['size'],scramble=scramble,key=key,
                weight_l1=float(abs(w-v).sum()),objective_delta_unit=float(delta/cfg['risk_unit'])))
            if method!='MA':
                contrasts.append(dict(cohort=r['cohort'],method=method,size=r['size'],scramble=scramble,key=key,
                    relative_risk=a['evaluation']['second_moment']/amap['MA']['evaluation']['second_moment']-1))
    pd.DataFrame(within).groupby(['cohort','method','size']).agg(weight_l1_median=('weight_l1','median'),weight_l1_max=('weight_l1','max'),
        objective_abs_max=('objective_delta_unit',lambda x:abs(x).max())).to_csv(ROOT/'output/integration_within_seed.csv')
    contrast=pd.DataFrame(contrasts);contrast.to_csv(PRIVATE/'evaluation/audit_contrasts.csv',index=False)
    cr=contrast[contrast['size']==256].groupby(['cohort','method','key']).relative_risk.agg(['min','max','mean'])
    cr['sign_changes']=(cr['min']<0)&(cr['max']>0)
    cr.groupby(['cohort','method']).agg(windows=('mean','size'),sign_change_windows=('sign_changes','sum'),
        relative_min=('min','min'),relative_max=('max','max'),mean_relative=('mean','mean')).to_csv(ROOT/'output/audit_rank_stability.csv')
    return frame

def reference():
    cfg=read(ROOT/'configs/base.json');choice=read(ROOT/'configs/selected.json');rows=read(PRIVATE/'manifest_all.json');out=[]
    for c in ('long','limited'):
        row=next(r for r in rows if r['d']==20 and r['panel']==0 and r['cohort']==c)
        sc,ll,*_=scene(row);post=posterior_from_loglik(sc,ll,power=choice['power'])
        for spec in [RiskSpec('MA','mean'),RiskSpec('A-Tail',rho=.5),RiskSpec('B-Tail',rho=.5,conditional=True),RiskSpec('B-Max','max',conditional=True)]:
            ans=solve_center(sc,post,spec,cfg['risk_unit']);env=Envelope(sc,post,spec,cfg['risk_unit'])
            w,q,value,status=conic_reference(sc,env)
            margin=3e-7*cfg['risk_unit']
            assert ans.lower-margin<=value<=ans.upper+margin,(c,spec.name,ans.lower,value,ans.upper)
            out.append(dict(cohort=c,method=spec.name,gap_unit=ans.gap/cfg['risk_unit'],reference_objective=value,
                            lower=ans.lower,upper=ans.upper,reference_status=status,margin=margin))
    dump(ROOT/'output/real_reference_gate.json',out);print('real-reference comparisons passed',len(out))

if __name__=='__main__':
    import sys
    if len(sys.argv)<2 or sys.argv[1]=='temporal':temporal()
    elif sys.argv[1]=='reference':reference()
    else:integration()
