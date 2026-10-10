"""Numerical-only repair: archive original validation records, use exact dominance."""
import shutil
import time
import numpy as np
from .common import PRIVATE,ROOT,read,dump,ledger
from .run import scene,tasks,data
from .model import PARAMS,stable_solve
from .book import isolated_risk
from ..scenario_inference import posterior_from_loglik
from ..risk_envelopes import RiskSpec,Envelope

def run():
    start=time.perf_counter();cpu=time.process_time();cfg=read(ROOT/'configs/base.json');choice=read(ROOT/'configs/selected.json')
    folder=PRIVATE/'records_archive/tune_before_exact_lp';folder.mkdir(parents=True,exist_ok=True)
    count=0;maxgap=0.
    for task in tasks('tune'):
        row=task['row'];path=PRIVATE/'records/tune'/f"{row['key']}_256_0.json";r=read(path)
        if r.get('lp_repaired'):continue
        archive=folder/path.name
        if not archive.exists():shutil.copy2(path,archive)
        sc,ll,*_=scene(row);post=posterior_from_loglik(sc,ll,power=choice['power'])
        for a in r['methods']:
            if a['method'].startswith('lambda=') or a['method']=='C-LP-Max':
                lam=float(a['method'].split('=')[1]) if '=' in a['method'] else choice['lam']
                spec=RiskSpec(a['method'],'max',penalty=lam)
                ans=stable_solve(sc,post,spec,cfg['risk_unit'])
                full=Envelope(sc,post,spec,cfg['risk_unit']).support(sc.costs(ans.weights)-sc.oracle_lower)[0]
                assert abs(full-ans.upper)<1e-7*cfg['risk_unit']
                a.update(weights=ans.weights.tolist(),gap=ans.gap,status=ans.status,seconds=ans.seconds,
                         objective=ans.objective,lower=ans.lower,upper=ans.upper,diagnostics=ans.diagnostics,
                         evaluation=isolated_risk(data(),row['t'],row['ids'],ans.weights))
                maxgap=max(maxgap,ans.gap/cfg['risk_unit'])
        r['lp_repaired']=True;dump(path,r);count+=1
    dump(ROOT/'output/lp_repair.json',dict(windows=count,max_gap_unit=maxgap,
        reason='Exact redundant-atom elimination fixes large evidence-offset scaling; original records archived privately',
        performance_selection_used=False))
    ledger('repair_lp',time.perf_counter()-start,time.process_time()-cpu,dict(windows=count))
    print('repaired',count,'max gap/u',maxgap)

if __name__=='__main__':run()
