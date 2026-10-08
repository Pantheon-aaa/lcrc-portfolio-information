"""Evaluator-only fixed-grid priors, path-cluster summaries and numeric audits."""
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t
from .experiment_v2 import OUT,CFG,load_rows,write_json
from .scenario_inference import prior_components

def weighted_grid(stage):
    rows=load_rows(stage)
    if stage=='G1': rows+=load_rows('baseline_audit')
    df=pd.DataFrame(rows)
    if not len(df):return
    if stage=='confirm': df=df[df.kind=='G1']
    df=df[~df.failure].copy()
    points=pd.DataFrame(load_rows('G1' if stage=='G1' else 'confirm'))
    points=points[points.kind=='G1'].drop_duplicates(['family_id','point_id'])
    masses={}
    for fid,g in points.groupby('family_id'):
        lp=prior_components(np.array(g.theta.tolist()));p=np.exp(lp-logsumexp(lp,axis=1)[:,None])
        for j,name in enumerate(('uniform','center','edge-bimodal','skew')):
            for point,weight in zip(g.point_id,p[j]):masses[fid,point,name]=weight
    point=df.groupby(['family_id','point_id','method']).regret_scaled.agg(['mean','count']).reset_index()
    point.to_csv(OUT/f'{stage}_point_risk.csv',index=False)
    records=[]
    for (fid,method),g in point.groupby(['family_id','method']):
        for j,name in enumerate(('uniform','center','edge-bimodal','skew')):
            records.append(dict(stage=stage,family_id=fid,method=method,distribution=name,
                grid_points=len(g),weighted_risk=sum(masses[fid,p,name]*v for p,v in zip(g.point_id,g['mean'])),
                worst_point=float(g['mean'].max()),grid_p90=float(g['mean'].quantile(.9)),
                interpretation='weighted finite eight-point grid, not continuous-prior integration'))
    pd.DataFrame(records).to_csv(OUT/f'{stage}_four_prior_grid.csv',index=False)

def path_summary(stage):
    df=pd.DataFrame(load_rows(stage))
    if not len(df):return
    df=df[~df.failure]
    records=[]
    for variant,g in df.groupby('variant'):
        for metric in ('regret_scaled','realized_risk'):
            pivot=g.pivot(index=['path_id','window'],columns='method',values=metric)
            for method in pivot:
                diff=(pivot[method]-pivot['A-Mean']).dropna()
                means=diff.groupby(level='path_id').mean();se=means.std(ddof=1)/np.sqrt(len(means))
                ci=t.ppf(.975,len(means)-1)*se
                records.append(dict(stage=stage,variant=variant,method=method,metric=metric,
                    difference=float(means.mean()),low=float(means.mean()-ci),high=float(means.mean()+ci),
                    paths=len(means),paired_windows=len(diff),
                    interval='paired whole-path means; three paths, t approximation; exploratory'))
    pd.DataFrame(records).to_csv(OUT/f'{stage}_path_paired.csv',index=False)

def point_and_uncertainty(stage):
    df=pd.DataFrame(load_rows(stage))
    if not len(df):return
    df=df[~df.failure]
    point=df.groupby(['kind','family_id','point_id','n','m','method']).regret_scaled.agg(['mean','count']).reset_index()
    point.to_csv(OUT/f'{stage}_all_point_risk.csv',index=False)
    point.groupby(['kind','method'])['mean'].agg(['mean','max',lambda x:x.quantile(.9)]).to_csv(OUT/f'{stage}_point_risk_summary.csv')
    if stage!='confirm':return
    rows=[];rng=np.random.default_rng(625001)
    for kind,g in df.groupby('kind'):
        pivot=g.pivot(index=['family_id','point_id','n','m','rep'],columns='method',values='regret_scaled')
        for base in ('A-Mean','REF-EB'):
            for name in pivot.columns:
                if name in ('A-Mean','REF-EB'):continue
                diff=(pivot[name]-pivot[base]).dropna();families=sorted(diff.index.get_level_values('family_id').unique())
                if len(families)!=3:continue
                family_boot=[]
                for fid in families:
                    cells=[]
                    for _,values in diff.xs(fid,level='family_id').groupby(level=['point_id','n','m']):
                        a=values.to_numpy();cells.append(rng.choice(a,size=(3000,len(a)),replace=True).mean(1))
                    family_boot.append(np.mean(cells,axis=0))
                family_boot=np.array(family_boot)
                selected=rng.integers(0,3,size=(1000,3))
                bootstrap=family_boot[selected,np.arange(3000).reshape(1000,3)].mean(1)
                low,high=np.quantile(bootstrap,[.025,.975])
                rows.append({'kind':kind,'method':name,'baseline':base,'difference':float(diff.mean()),
                    'low':low,'high':high,'resamples':1000,
                    'interpretation':'descriptive hierarchical bootstrap; family resampling, fixed parameter/information strata, paired repetition resampling; only three families'})
    pd.DataFrame(rows).to_csv(OUT/'confirmation_hierarchical_bootstrap.csv',index=False)

def integrity():
    checks=[]
    for stage in ('development','G1','G2','confirm','G3_d20','G3_d50','resolution','baseline_audit'):
        rows=load_rows(stage)
        if not rows:continue
        methods=sorted({r['method'] for r in rows})
        keys=[(r['instance_id'],r['method'],r.get('size'),r.get('scramble')) for r in rows]
        assert len(keys)==len(set(keys)),(stage,'duplicates')
        valid=[r for r in rows if not r['failure']]
        for r in valid:
            assert np.isfinite(r['weights']).all(),(stage,r['instance_id'],r['method'])
            w=np.array(r['weights']);assert abs(w.sum()-1)<1e-7 and w.min()>-1e-8 and w.max()<.20000001
        drift=[]
        for r in valid:
            if 'adversary' not in r or 'alpha' not in r:continue
            if not (r['method'].startswith('B-') or r['method'].startswith('H-Conditional') or r['method'].startswith('C-Soft-Cond')):continue
            if r['method']=='B-Plugin' or 'pR' in r['method']:continue
            q=np.array(r['adversary']);alpha=np.array(r['alpha'])
            if len(q)%len(alpha)==0:drift.append(float(abs(q.reshape(len(alpha),-1).sum(1)-alpha).max()))
        checks.append(dict(stage=stage,rows=len(rows),instances=len({r['instance_id'] for r in rows}),
            methods=methods,failures=sum(r['failure'] for r in rows),
            statuses=dict(pd.Series([r.get('status','baseline') for r in rows]).value_counts()),
            max_conditional_mass_residual=max(drift,default=0),
            min_rows_per_instance=min(pd.Series([r['instance_id'] for r in rows]).value_counts())))
    write_json(OUT/'delivery_integrity.json',checks)

def run():
    for stage in ('G1','confirm'):weighted_grid(stage)
    for stage in ('G1','G2','confirm'):point_and_uncertainty(stage)
    for stage in ('G3_d20','G3_d50'):path_summary(stage)
    integrity()

if __name__=='__main__':run()
