import numpy as np
import pandas as pd
from .common import ROOT,PRIVATE,read
from .run import scene
from ..scenario_inference import posterior_from_loglik

def run():
    cfg=read(ROOT/'configs/selected.json');lookup={r['key']:r for r in read(PRIVATE/'manifest_all.json')};out=[]
    for p in sorted((PRIVATE/'records/test').glob('*.json')):
        r=read(p);methods={a['method']:a for a in r.get('methods',[])}
        if 'MA' not in methods:continue
        sc,ll,*_=scene(lookup[r['key']]);post=posterior_from_loglik(sc,ll,power=cfg['power'])
        alpha=np.exp(post.log_alpha);inner=[]
        for s in range(len(alpha)):
            ix=sc.groups==s;pcond=np.exp(post.log_conditional[ix]);inner.append(float(1/(pcond@pcond)))
        for m in ('A-Tail-0.5','B-Tail-0.5','B-Tail-0.25','B-Max','A-Max','C-LP-Max','C-Soft','MA-power1','B-Tail-0.5-power1'):
            if m not in methods or not methods[m].get('weights'):continue
            a=methods[m];w=np.array(a['weights']);ma=np.array(methods['MA']['weights'])
            ag=np.array(methods['A-Tail-0.5']['weights'])
            out.append(dict(cohort=r['cohort'],period=r['period'],method=m,
                l1_to_ma=float(abs(w-ma).sum()),l1_to_global_half=float(abs(w-ag).sum()),
                support=int((w>1e-6).sum()),at_cap=int((w>.2-1e-6).sum()),
                alpha_ess=float(1/(alpha@alpha)),conditional_ess=float(alpha@inner),
                isolated_risk=a['evaluation']['second_moment'],ma_isolated_risk=methods['MA']['evaluation']['second_moment'],
                lp_remaining=256-a.get('diagnostics',{}).get('exact_dominated_atoms_removed',0) if m=='C-LP-Max' else np.nan))
    df=pd.DataFrame(out);df.to_csv(PRIVATE/'evaluation/mechanism_instances.csv',index=False)
    summary=df.groupby(['cohort','period','method']).agg(l1_ma_mean=('l1_to_ma','mean'),l1_global_half_mean=('l1_to_global_half','mean'),
        support_mean=('support','mean'),cap_mean=('at_cap','mean'),alpha_ess_median=('alpha_ess','median'),
        conditional_ess_median=('conditional_ess','median'),isolated_risk=('isolated_risk','mean'),
        ma_isolated_risk=('ma_isolated_risk','mean'),lp_single_fraction=('lp_remaining',lambda x:float((x.dropna()==1).mean()) if x.notna().any() else np.nan))
    summary.to_csv(ROOT/'output/mechanism.csv');print(summary.to_string())

if __name__=='__main__':run()
