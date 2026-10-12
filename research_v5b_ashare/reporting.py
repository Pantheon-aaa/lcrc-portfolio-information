"""Public aggregates and synchronized calendar-block inference; no security detail."""
import json,time,hashlib,tarfile,importlib.metadata
import numpy as np
from .common import *

def num(x):return float(x) if x is not None else np.nan
def key(cell):return cell['method']+':'+str(cell['beta'])

def collect():
    from collections import Counter,defaultdict
    manifest=read(PRIVATE/'manifest.json');public=[];fail=[];complete=0;counts=defaultdict(Counter)
    for job in manifest:
        path=PRIVATE/'decisions'/f'{job["id"]}.json';p=job['panel']
        if not path.exists():fail.append(dict(id=job['id'],stage=job['stage'],status='not_run'));counts[job['stage']]['not_run']+=1;continue
        r=read(path)
        counts[job['stage']][r['status']]+=1
        if r['status']!='complete':fail.append(dict(id=job['id'],stage=job['stage'],status=r['status'],error=r.get('error')));continue
        complete+=1
        for cell in r['cells']:
            if cell['status']!='complete':fail.append(dict(id=job['id'],method=key(cell),status=cell['status']));continue
            d=cell['decision'];rr=d.get('raw_result') or {}
            if d.get('fallback_reason') or d['raw_status'] not in ['optimal','MA','greedy','greedy_1swap','surrogate','hinge_zero_MA_secondary','equal_weight','practical_baseline']:
                fail.append(dict(id=job['id'],method=key(cell),status=d['raw_status'],fallback=d.get('fallback_reason')))
            for view,e in cell['evaluation'].items():
                public.append(dict(id=job['id'],stage=job['stage'],date=p['date'],phase=p['phase'],domain=p['domain'],seed=p['seed'],N=job['N'],M=job['M'],
                   tau=cell['tau_index'],method=key(cell),view=view,u=r['u'],status=e['status'],success_lower=e.get('success_lower'),success_upper=e.get('success_upper'),signed_excess_risk=e.get('signed_excess_risk'),
                   regret_lower=e.get('regret_lower'),regret_upper=e.get('regret_upper'),loss=e.get('loss'),quality=e.get('quality'),
                   internal_success_lower=d['success_mass_lower'],internal_success_upper=d['success_mass_upper'],budget_certified=d['budget_certified'],
                   mean_increment_upper=d['mean_increment_upper'],budget_lower=d['budget_lower'],primary_gap=rr.get('primary_mass_gap'),secondary_gap=rr.get('secondary_price_gap'),
                   fallback=d.get('fallback_reason'),solver_status=d['raw_status'],residual=d['primal_residual'],solver_cpu=cell['measured_cpu']))
    import gzip
    with gzip.open(OUT/'risk_records_aggregate.json.gz','wt',encoding='utf-8') as f:json.dump(clean(public),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    write(OUT/'failure_and_unfinished.json',fail)
    unrun=sum(c['not_run'] for c in counts.values())
    write(OUT/'completion.json',dict(planned=len(manifest),processed=len(manifest)-unrun,completed=complete,unrun=unrun,remaining=unrun,unresolved_or_failed=len(manifest)-complete-unrun,
       by_stage={s:dict(planned=sum(c.values()),processed=sum(c.values())-c['not_run'],complete=c['complete'],statuses=dict(c)) for s,c in counts.items()}))
    return public

def inference(records):
    manifest=[j for j in read(PRIVATE/'manifest.json') if j['stage']=='R2' and j['panel']['phase']=='test']
    dates=sorted(set(j['panel']['date'] for j in manifest));lookup={(r['id'],r['method']):r for r in records if r['stage']=='R2' and r['phase']=='test' and r['tau']==1 and r['view']=='H63_stale63'}
    comparisons=[];B=10000
    for beta in [.1,.25]:
        for other in ['MA:0.0',f'Greedy:{beta}']:
            lo=[];hi=[];common=[]
            for day in dates:
                pairs=[]
                for job in [j for j in manifest if j['panel']['date']==day]:
                    a=lookup.get((job['id'],f'A1B:{beta}'));b=lookup.get((job['id'],other))
                    pairs.append((-1.,1.) if a is None or b is None else (a['success_lower']-b['success_upper'],a['success_upper']-b['success_lower']))
                    if a and b and a['status']!='unknown_future_returns' and b['status']!='unknown_future_returns' and a['success_lower']==a['success_upper'] and b['success_lower']==b['success_upper']:common.append(a['success_lower']-b['success_lower'])
                lo.append(np.mean([z[0] for z in pairs]));hi.append(np.mean([z[1] for z in pairs]))
            low=np.array(lo);high=np.array(hi);entry=dict(endpoint='success',beta=beta,baseline=other,lower=float(low.mean()),upper=float(high.mean()),months=len(dates),common_evaluable_exploratory=float(np.mean(common)) if common else None,block_results={})
            for block in [6,3,12]:
                rng=np.random.default_rng(20261012);starts=rng.integers(0,len(dates),size=(B,int(np.ceil(len(dates)/block))))
                idx=((starts[:,:,None]+np.arange(block))%len(dates)).reshape(B,-1)[:,:len(dates)]
                dl=low[idx].mean(1);dh=high[idx].mean(1)
                entry['block_results'][block]=dict(basic_ci=[float(2*low.mean()-np.quantile(dl,.975)),float(2*high.mean()-np.quantile(dh,.025))],
                    percentile_ci=[float(np.quantile(dl,.025)),float(np.quantile(dh,.975))],p_one_sided_conservative=float((1+np.sum(dl-low.mean()>=low.mean()))/(B+1)))
            entry['p']=entry['block_results'][6]['p_one_sided_conservative'];comparisons.append(entry)
    for beta in [.1,.25]:
        vals=[];missing=0
        for job in manifest:
            a=lookup.get((job['id'],f'A1B:{beta}'));b=lookup.get((job['id'],'MA:0.0'))
            if a is None or b is None or a['regret_lower'] is None or b['regret_lower'] is None:missing+=1;continue
            vals.append(((1+beta)*num(b['regret_lower'])-num(a['regret_upper']))/a['u'])
        comparisons.append(dict(endpoint='mean_cost',beta=beta,p=None,status='unbounded_missing_no_full_sample_test' if missing else 'complete',missing_instances=missing,
          common_evaluable_exploratory=float(np.mean(vals)) if vals else None))
    pp=np.array([c['p'] if c['p'] is not None else 1. for c in comparisons]);order=np.argsort(pp);adj=np.maximum.accumulate((6-np.arange(6))*pp[order]);adj=np.minimum(adj,1)
    for j,i in enumerate(order):comparisons[i]['holm_p']=None if comparisons[i]['p'] is None else float(adj[j])
    write(OUT/'six_tests.json',dict(comparisons=comparisons,months=dates,bootstrap=10000,seed=20261012,
      missing_policy='fixed calendar denominator, paired success intervals; unavailable cost hypotheses treated as p=1 only for six-family multiplicity adjustment'))

def tables(records):
    table=[]
    for stage in sorted(set(r['stage'] for r in records)):
        selected=[r for r in records if r['stage']==stage and r['phase']=='test' and r['tau']==1 and r['view']=='H63_stale63']
        for method in sorted(set(r['method'] for r in selected)):
            rr=[r for r in selected if r['method']==method];finite=[r for r in rr if r['regret_lower'] is not None]
            table.append(dict(stage=stage,method=method,recorded=len(rr),evaluable=len(finite),internal_success=float(np.mean([r['internal_success_lower'] for r in rr])),
              success_common=float(np.mean([r['success_lower'] for r in finite])) if finite else None,
              regret_over_u_common=float(np.mean([(r['regret_lower']+r['regret_upper'])/(2*r['u']) for r in finite])) if finite else None,
              p95_regret_over_u_common=float(np.quantile([r['regret_upper']/r['u'] for r in finite],.95)) if finite else None,
              failed_conditional_regret_over_u_common=float(np.mean([r['regret_upper']/r['u'] for r in finite if r['status']=='failure'])) if any(r['status']=='failure' for r in finite) else None,
              loss_over_u_common=float(np.mean([r['loss']/r['u'] for r in finite])) if finite else None,
              budget_certified_fraction=float(np.mean([r['budget_certified'] for r in rr])),fallback_fraction=float(np.mean([bool(r['fallback']) for r in rr]))))
    write(OUT/'method_tables.json',table)
    sensitivity=[]
    for method in ['MA:0.0','A1B:0.1','A1B:0.25','Greedy:0.1','Greedy:0.25']:
        for view in ['H63_stale0','H63_stale20','H63_stale63','H63_stale126','H126_stale63']:
            rr=[r for r in records if r['stage']=='R2' and r['phase']=='test' and r['tau']==1 and r['method']==method and r['view']==view];good=[r for r in rr if r['regret_lower'] is not None]
            sensitivity.append(dict(method=method,view=view,recorded=len(rr),evaluable=len(good),success_common=float(np.mean([r['success_lower'] for r in good])) if good else None,
                regret_over_u_common=float(np.mean([r['regret_upper']/r['u'] for r in good])) if good else None))
    write(OUT/'valuation_sensitivity.json',sensitivity)
    return table

def bridge_summary():
    aggregate={}
    for path in (PRIVATE/'semisynthetic').glob('*/*.json'):
        if path.name.startswith('platform'):continue
        r=read(path)
        for cell in r['results']:
            ids=[r['truth']] if r['truth']>=0 else list(range(6));ev=cell['evaluation'];k=(r['domain'],r['n'],r['view'],cell['threshold'],cell['method'],str(cell['beta']))
            aggregate.setdefault(k,[]).extend([(ev['success_lower'][j],ev['success_upper'][j],ev['regret_upper'][j]) for j in ids])
    out=[dict(domain=k[0],n=k[1],view=k[2],threshold=k[3],method=k[4],beta=k[5],observations=len(v),success_lower=float(np.mean(np.array(v)[:,0])),success_upper=float(np.mean(np.array(v)[:,1])),mean_regret=float(np.mean(np.array(v)[:,2]))) for k,v in aggregate.items()]
    write(OUT/'bridge_summary.json',out)
    info=[]
    for path in (PRIVATE/'information').glob('*.json'):
        r=read(path)
        if r['status']!='ready':continue
        for cell in r['cells']:info.append({k:r[k] for k in ['date','domain','phase','n','view','scalar_count']}|dict(method=cell['method'],beta=cell['beta'],**cell['evaluation']))
    write(OUT/'information_summary.json',info)

def report():
    records=collect();inference(records);table=tables(records);bridge_summary()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for prefix in ['A1B','Greedy','Hinge','CVaR05','MA','EMGMV','EW']:
        rr=[r for r in table if r['stage']=='R2' and r['method'].split(':')[0]==prefix and r['success_common'] is not None]
        if rr:ax.plot([r['regret_over_u_common'] for r in rr],[r['success_common'] for r in rr],'o-',label=prefix)
    ax.set_xlabel('Mean empirical regret / historical risk unit');ax.set_ylabel('Empirical success on evaluable records');ax.set_title('Approximate A-share evaluation; descriptive available-case frontier');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'risk_frontier.png',dpi=160);plt.close(fig)
    tests=read(OUT/'six_tests.json');completion=read(OUT/'completion.json');cfg=read(ROOT/'protocol_frozen.json')
    lines=['# research_v5b：近似真实 A 股实验与机制验证','',
      '**阶段性汇总：仍有冻结实例未执行，完成规模实验后更新。**' if completion['unrun'] else '**冻结清单已全部处理；数据不可用和失败项仍保留，下文逐项说明。**','',
      '本报告的真实评价采用总收益估值近似：缺失期间沿用上一估值，恢复日计入累计变化。训练端仍保留 NaN。结果不代表精确可交易回测或真实市场风险保证。','',
      '## 配置与完成情况','',f'冻结 τ={cfg["tau"]}；η_F1={cfg["eta"]}，η_R3={cfg["eta_R3"]}，η_F2={cfg["eta_F2"]}。主估值陈旧期63日，敏感性20/126日，严格完整口径另列。',
      f'主跑批计划 {completion["planned"]} 项，已处理 {completion["processed"]} 项，形成有效决策 {completion["completed"]} 项，数据未决/失败 {completion["unresolved_or_failed"]} 项，未运行 {completion["unrun"]} 项；完整清单保留，不根据胜负删减。',
      '半合成、信息视图、R1枚举另有独立记录，未计入上述主跑批数量。2019年末目标用于初始建仓；2026单列。','',
      '## 真实风险比较（公共可评价记录的描述性结果）','',
      '下表不能替代固定月历上的区间推断。不同估计器可用性可能不同，LW缺失时不插补。正式主要比较使用同一月历、同一数据配对。','',
      '|方法|可评价记录|达标率|平均 regret/u|p95 regret/u|模型内达标质量|','|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['stage']=='R2' and r['success_common'] is not None:lines.append(f'|{r["method"]}|{r["evaluable"]}|{r["success_common"]:.3%}|{r["regret_over_u_common"]:.5f}|{r["p95_regret_over_u_common"]:.5f}|{r["internal_success"]:.3%}|')
    lines+=['','![风险前沿](output/risk_frontier.png)','','## 六项预设推断','','|比较|全月历识别区间|95%基本bootstrap区间，块长6|Holm p|','|---|---|---|---|']
    for r in tests['comparisons']:
        interval=f'[{r["lower"]:.4f}, {r["upper"]:.4f}]' if 'lower' in r else '连续损失未决，未计算全样本检验'
        ci=r.get('block_results',{}).get('6',{}).get('basic_ci')
        cis=f'[{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '未计算'
        lines.append(f'|{r["endpoint"]} β={r["beta"]} {r.get("baseline","")}|{interval}|{cis}|{r["holm_p"] if r["holm_p"] is not None else "未计算"}|')
    lines+=['','采用10,000次同步循环月份块重采样，主块长6，敏感性3/12；缺失成功率按区间传播。无法有限界定的连续损失不输出全样本p值。六项检验中的不可计算项不被解释为零效应。',
      '','## 估值、账本与机制边界','',
      '20/63/126日和严格完整口径见 `output/valuation_sensitivity.json`；同一目标不重新调参。账户以可分割总收益单位计价、下一交易日收盘调整，成本每边0/10/20/50bp。无记录证券不成交，池外遗留持仓不删除，不重复发放分红。未知估值暂停交易并保留净值未决。',
      '账本聚合见 `output/book_summary.json`，半合成见 `output/bridge_summary.json`，共同观测见 `output/information_summary.json`。模型内目标改善、已知高斯真模型改善、未来经验改善分别解释，不能相互替代。',
      '','## 复现与未完成项','',
      '私有运行目录保存矩阵、似然、目标、执行单位及原始证书。公开目录只保存代码、配置、聚合记录和哈希。原v5、v4a及源数据保持不变；本轮默认本地提交。',
      '完整清单见 `output/scheduled_manifest.json`、`output/failure_and_unfinished.json`；累计CPU见 `output/runtime_snapshot.json`，包含此前v5用量。对失败不放宽ε或预算；开放证书与回退单列。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from .report_details import extend_report
    extend_report(records,table)
    print('REPORT',completion,flush=True)

def archive():
    import zstandard
    root=PRIVATE/'archives';root.mkdir(exist_ok=True);path=root/'v5b_records.tar.zst'
    files=[p for p in PRIVATE.rglob('*') if p.is_file() and p.relative_to(PRIVATE).parts[0] not in ['archives','valuation'] and '.tmp' not in p.name]
    expected={p.relative_to(PRIVATE).as_posix():sha(p) for p in files}
    with path.open('wb') as out,zstandard.ZstdCompressor(level=3,threads=0).stream_writer(out) as z,tarfile.open(fileobj=z,mode='w|') as t:
        for p in files:t.add(p,arcname=p.relative_to(PRIVATE).as_posix(),recursive=False)
    actual={}
    with path.open('rb') as src,zstandard.ZstdDecompressor().stream_reader(src) as z,tarfile.open(fileobj=z,mode='r|') as t:
        for m in t:
            f=t.extractfile(m)
            if f:actual[m.name]=hashlib.file_digest(f,'sha256').hexdigest()
    assert expected==actual;write(root/'manifest.json',expected);write(OUT/'archive_verification.json',dict(files=len(files),sha256=sha(path),bytes=path.stat().st_size,verified=True,
       excluded='rebuildable valuation cache, archive itself and final live resource footer'))

def delivery_check():
    import ast
    for file in ROOT.glob('*.py'):ast.parse(file.read_text(encoding='utf-8'))
    assert sha(SOURCE)==read(CACHE/'index.json')['source_sha256']
    assert PRIOR_CPU+sum(r['cpu_seconds'] for r in rows(PRIVATE/'runtime_ledger.jsonl'))<86400
    assert read(OUT/'gate0.json')['passed']
    cfg=read(ROOT/'protocol_frozen.json')
    assert all(sha(MEMBERSHIP/name)==value for name,value in cfg['membership_hashes'].items())
    assert digest(json.dumps(clean(read(PRIVATE/'manifest.json')),sort_keys=True))==cfg['manifest_hash']
    core=['risk.py','decision.py','data.py','valuation.py','evaluator.py','bridge.py','information.py','setup.py']
    assert all(sha(ROOT/name)==cfg['source_hashes'][name] for name in core)
    forbidden={'symbol','symbols','security_id','ranking','columns','raw_target','evaluated_target','guarded_target','executed_units','Q','log_likelihood','weights','ma_weights'}
    def inspect_public(x):
        if isinstance(x,dict):
            assert not forbidden.intersection(x),f'Private field in public aggregate: {forbidden.intersection(x)}'
            for value in x.values():inspect_public(value)
        elif isinstance(x,list):
            for value in x:inspect_public(value)
    for path in OUT.glob('*.json'):inspect_public(read(path))
    import gzip
    for path in OUT.glob('*.json.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:inspect_public(json.load(f))
    prefetch=[]
    if (PRIVATE/'prefetch_log.json').exists():
        for r in read(PRIVATE/'prefetch_log.json'):
            with np.load(PRIVATE/'frontends'/f'{r["shared_cost_id"]}.npz') as z:
                assert array_hash(z['Q'])==r['Q_hash'] and array_hash(z['scores'])==r['scores_hash']
            prefetch.append(r['shared_cost_id'])
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','pyarrow','cvxpy','clarabel','highspy','psutil','matplotlib','zstandard','scikit-learn','mpmath']}
    import platform,os
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n',encoding='utf-8')
    write(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
        blas_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        resource_note='At most two numerical worker processes; cumulative process CPU includes native solver threads. CPU is not wall time.'))
    write(OUT/'delivery_check.json',dict(passed=True,source_unchanged=True,membership_unchanged=True,manifest_unchanged=True,frozen_core_verified=core,public_security_detail_scan_passed=True,prefetch_cache_hashes_verified=len(prefetch),
       versions=versions,current_source_hashes=source_hashes(),frozen_source_hashes=cfg['source_hashes']))
