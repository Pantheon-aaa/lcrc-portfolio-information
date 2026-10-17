"""Read frozen v5b inputs and labels; never refit or modify their source."""
import time
import numpy as np
from scipy.special import logsumexp
from research_v4a.mean_budget import Family,MeanBudget
from research_v3.compatible_qcqp import solve as gamma
from .common import *
def run():
    start=time.process_time();cfg=read(ROOT.parent/'research_v5b_ashare/protocol_frozen.json');jobs=read(OLD/'manifest.json')
    jobs=[j for j in jobs if j['stage']=='R2' and j['panel']['phase']=='test'];write(OUT/'e1_manifest.json',[{k:j[k] for k in ('id','stage','N','M')} for j in jobs])
    out=[];hashes=read(RUN/'e1_source_hashes.json') if (RUN/'e1_source_hashes.json').exists() else {}
    for k,j in enumerate(jobs):
        path=RUN/'e1'/f"{j['id']}.json";source=OLD/'decisions'/f"{j['id']}.json"
        if path.exists():out.append(read(path));continue
        if time.process_time()-start>4800:break
        raw=read(source);hashes[str(source)]=sha(source);r={z:raw.get(z) for z in ('id','date','domain','seed','status','N','M','u')}
        if raw['status']=='complete':
            tick=time.process_time();key=raw['shared_cost_id'];npz=OLD/'frontends'/f'{key}.npz';meta=read(npz.with_suffix('.json'));hashes[str(npz)]=sha(npz)
            with np.load(npz) as z:Q=z['Q'];scores=z['scores'];unit=float(z['u'])
            lp=cfg['eta']*scores.sum(0);lp-=logsumexp(lp);p=np.exp(lp)
            f=Family(Q/unit,min(.2,5/len(Q[0])));mb=MeanBudget(f,p);g=gamma(f,tight=True,time_limit=30)
            cell=next(c for c in raw['cells'] if c['method']=='MA' and c['tau_index']==1);w=np.array(cell['decision']['evaluated_target']);lo,hi=f.regret_bounds(w);eps=cfg['tau'][1]
            spread=np.sum(abs(f.oracle[:,None]-f.oracle[None,:]),axis=2)
            em=meta['current_models'];r.update(A_lower=float(max(lo)/eps),A_upper=float(max(hi)/eps),gamma_lower_over_epsilon=g.lower/eps,gamma_upper_over_epsilon=g.upper/eps,gamma_status=g.status,
                Lstar_lower_over_u=mb.Llo,Lstar_upper_over_u=mb.Lhi,oracle_L1_spread=float(spread.max()),entropy=float(-np.sum(p[p>0]*lp[p>0])),ESS=float(1/(p@p)),
                internal_all_covered=bool(max(hi)<=eps),model_regret_lower=lo,model_regret_upper=hi,em_max_iterations=sum(a['status']=='max_iterations' for a in em),score_fit_status=meta['score_fit_status'],
                evaluation=cell['evaluation'],old_MA_recompute_L1=float(abs(w-mb.w).sum()),a1_same_MA=[],cpu_seconds=time.process_time()-tick)
            for c in raw['cells']:
                if c.get('method')=='A1B' and c['tau_index']==1 and c['beta'] in (.1,.25):r['a1_same_MA'].append(dict(beta=c['beta'],same=bool(np.array_equal(c['decision']['evaluated_target'],w)),budget_certified=c['decision']['budget_certified'],status=c['decision']['raw_status']))
            write(RUN/'e1_private'/f"{j['id']}.json",dict(weights=w,Q=Q,p=p,log_p=lp,source_hash=hashes[str(source)],matrix_hash=hashes[str(npz)],columns=raw['columns']))
        write(path,r);out.append(r)
        if k%24==0:print('E1',k+1,len(jobs),flush=True)
    # Source-level extreme returns are an audit, not a reason to rewrite prices.
    cache=OLD.parent/'ashare_v5/data';arrays={k:np.load(cache/f'{k}.npy',mmap_mode='r') for k in ('return_total','close','open','prev_close_exchange','total_adjustment_suspect','observed','trading_status')}
    ix=read(cache/'index.json');ret=arrays['return_total'];flat=np.asarray(ret).ravel();idx=np.argpartition(np.nan_to_num(flat,nan=-np.inf),-10)[-10:];ext=[]
    for z in idx:
        i,col=np.unravel_index(z,ret.shape);ext.append(dict(date=ix['dates'][i],symbol=ix['symbols'][col],**{k:float(v[i,col]) for k,v in arrays.items()},reason='unverified corporate-action or source-return anomaly; not auto-corrected'))
    write(RUN/'e1_source_extremes.json',ext)
    audit=read(ROOT.parent/'research_v5b_ashare/output/valuation_audit.json')
    write(OUT/'e1_source_audit.json',dict(valuation_audit=audit,source_return_max=max(x['return_total'] for x in ext),extreme_rows=10,classification='unresolved source anomaly; exact rows private',source_index_sha256=sha(cache/'index.json'),new_data_downloaded=False,old_files_modified=False))
    write(RUN/'e1_source_hashes.json',hashes);write(OUT/'e1_records.json',out)
    write(OUT/'e1_completion.json',dict(planned=len(jobs),processed=len(out),old_ma_certification_is_not_external_guarantee=True))
    print('E1 DONE',len(out),flush=True)
