"""Unknown nominal family: rolling factor-market stress, paired information access."""
import argparse,json,traceback
from time import perf_counter,process_time,time
import numpy as np
from core import solve_qp,em_cov
from .experiment_v2 import ROOT,OUT,CFG,write_json,load_rows,append_row,source_hash,single_method
from .new_generators import factor_path,estimated_scenarios,seed_for
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec


def run(d=20):
    setup=json.loads((OUT/'development_setup.json').read_text()); unit=setup['risk_unit']
    selection=json.loads((CFG/'selection.json').read_text()); cfg=json.loads((CFG/'protocol.json').read_text())
    specs=[RiskSpec(**s) for s in selection['confirmation_specs'] if s['name']!='REF-EB']
    # EB fitted to a supplied parametric family cannot silently become a bootstrap prior.
    stage=f'G3_d{d}'; path=OUT/f'{stage}.jsonl'; done={(x['instance_id'],x['method']) for x in load_rows(stage)}
    manifest=[]
    for variant in cfg['g3_variants']:
        for path_id in range(3):
            seed=seed_for('G3',d,path_id)
            for window in range(12):
                manifest.append({'variant':variant,'path_id':path_id,'window':window,'seed':seed,
                                 'instance_id':f'{d}-{variant}-{path_id}-{window}'})
    manifest_path=CFG/f'{stage}_manifest.json'
    if not manifest_path.exists(): write_json(manifest_path,manifest)
    start=perf_counter(); cpu=process_time(); sha=source_hash(); cached={}
    budget_spent=sum(x['wall_seconds'] for x in load_rows('runtime_ledger'))
    for i,r in enumerate(manifest):
        names=[s.name for s in specs]+['Equal-Weight','EM-shrink']
        if all((r['instance_id'],n) in done for n in names): continue
        key=(r['variant'],r['path_id'])
        if key not in cached: cached[key]=factor_path(r['seed'],d,r['variant'])
        Y,full,Qs=cached[key]; end=252+21*r['window']; past=Y[end-252:end]; future=full[end:end+21]
        trueQ=Qs[end:end+21].mean(axis=0); oracle=solve_qp(trueQ,constraints={'upper':.2})
        tic=perf_counter(); sc,score,nominal=estimated_scenarios(past,128,seed_for('G3scenario',d,r['path_id'],r['window']))
        build=perf_counter()-tic; tic=perf_counter(); post=posterior(sc,score); inference=perf_counter()-tic
        for name in names:
            if (r['instance_id'],name) in done: continue
            base=r|{'stage':stage,'d':d,'source_hash':sha,'unit':unit,'scenario_build_seconds':build,
                    'inference_seconds':inference,'scenario_oracle_seconds':sc.oracle_seconds,
                    'posterior':post.diagnostics,'alpha':np.exp(post.log_alpha),'mass_source':sc.source,
                    'guarantee':'empirical only; unknown family and bootstrap reference',
                    'mask_scalar_count':int(np.isfinite(past).sum())}
            try:
                if name=='Equal-Weight':
                    w=np.ones(d)/d; details={'status':'baseline','seconds':0.,'gap':None,'residual':0.}
                elif name=='EM-shrink':
                    # Fixed diagonal shrinkage selected in the independent v1 development,
                    # retained as a descriptive comparator, not a newly tuned strong baseline.
                    tic=perf_counter(); q=np.diag(np.diag(nominal)); ans=solve_qp(q,constraints={'upper':.2})
                    w=ans['w']; details={'status':'baseline','seconds':perf_counter()-tic,'gap':ans['gap'],'residual':ans['primal_residual']}
                else:
                    spec=next(s for s in specs if s.name==name); ans=single_method(sc,post,spec,unit)
                    w=ans.weights; details={'status':ans.status,'seconds':ans.seconds,'gap':ans.gap,'residual':ans.residual,
                        'objective':ans.objective,'lower':ans.lower,'upper':ans.upper,'adversary':ans.adversary,'diagnostics':ans.diagnostics}
                truecost=.5*w@trueQ@w
                base.update(details); base.update(method=name,weights=w,
                    regret_scaled=float((truecost-(oracle['value_lb']+oracle['value_ub'])/2)/unit),
                    realized_risk=float(.5*np.mean((future@w)**2)),failure=False)
            except Exception as exc:
                base.update(method=name,failure=True,status='exception',error=repr(exc),traceback=traceback.format_exc())
            append_row(path,base)
        if i%6==0: print(stage,i+1,'/',len(manifest),'elapsed',round(perf_counter()-start,1),flush=True)
        if budget_spent+perf_counter()-start>=cfg['budget_seconds']: break
    append_row(OUT/'runtime_ledger.jsonl',{'stage':stage,'wall_seconds':perf_counter()-start,
        'process_cpu_seconds':process_time()-cpu,'source_hash':sha,'unix':time()})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--d',type=int,default=20); a=p.parse_args(); run(a.d)
