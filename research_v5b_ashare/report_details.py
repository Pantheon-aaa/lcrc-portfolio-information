"""Read-only additional summaries; this module never feeds experiment decisions."""
from collections import Counter,defaultdict
import numpy as np
from .common import *

def finite(x):
    try:return np.isfinite(float(x))
    except (TypeError,ValueError):return False

def mean(x):return float(np.mean(x)) if len(x) else None

def r1_audit():
    rows=[];frontiers=[]
    for path in (PRIVATE/'r1').glob('*.json'):
        r=read(path)
        if r['status']!='complete':continue
        for z in r['prices']:
            cv=z['cvx'];v=cv.get('value');lo=z['lower'];hi=z['upper']
            # CVXPY is an independent formulation, not a certified dual bound.
            distance=max(float(lo)-float(v),float(v)-float(hi),0.) if all(finite(x) for x in [lo,hi,v]) else None
            rows.append(dict(id=r['id'],threshold=r['threshold'],subset_size=len(z['G']),mass=z['mass'],lower=lo,upper=hi,
                             feasible=z['feasible'],status=z['status'],cvx_status=cv['status'],cvx_interval_distance=distance))
        for d in r['results']:
            if d['method']!='A1B':continue
            blo=d['budget_lower']/r['u'] if finite(d['budget_lower']) else np.inf
            bhi=d['budget_upper']/r['u'] if finite(d['budget_upper']) else np.inf
            feasible=[z for z in r['prices'] if z['feasible'] and float(z['upper'])<=blo]
            possible=[z for z in r['prices'] if z['status']!='infeasible' and float(z['lower'])<=bhi]
            lower=max([z['mass'] for z in feasible]+[0.]);upper=max([z['mass'] for z in possible]+[0.])
            raw=d.get('raw_result') or {}
            frontiers.append(dict(id=r['id'],domain=r['domain'],threshold=r['threshold'],beta=d['beta'],
                actual_lower=d['success_mass_lower'],actual_upper=d['success_mass_upper'],enumeration_lower=lower,enumeration_upper=upper,
                increment_over_u=d['mean_increment_upper']/r['u'],budget_certified=d['budget_certified'],status=d['raw_status'],
                primary_gap=raw.get('primary_mass_gap'),secondary_gap=raw.get('secondary_price_gap'),
                interval_conflict=bool(d['success_mass_lower']>upper+1e-8)))
    write(OUT/'r1_price_audit.json',rows);write(OUT/'r1_frontiers.json',frontiers)
    return rows,frontiers

def detailed_bridge():
    grouped=defaultdict(list);platform=[]
    for path in (PRIVATE/'semisynthetic').glob('*/*.json'):
        r=read(path)
        if path.name.startswith('platform'):
            platform.append(dict(domain=path.parent.name,h=r['h'],threshold=r['threshold'],prices=[{k:z[k] for k in ['G','mass','lower','upper','feasible','status']} for z in r['prices']]))
            continue
        for c in r['results']:
            k=(r['domain'],r['n'],r['view'],c['threshold'],c['method'],str(c['beta']))
            ev=c['evaluation']
            for t in ([r['truth']] if r['truth']>=0 else range(6)):
                grouped[k].append((t,ev['success_lower'][t],ev['success_upper'][t],ev['regret_upper'][t],c['decision']['budget_certified']))
    result=[]
    for k,v in grouped.items():
        a=np.asarray(v);bytruth=[a[a[:,0]==t] for t in range(6)]
        result.append(dict(domain=k[0],n=k[1],view=k[2],threshold=k[3],method=k[4],beta=k[5],count=len(a),
            success_lower=float(a[:,1].mean()),success_upper=float(a[:,2].mean()),worst_truth_success_lower=min(float(b[:,1].mean()) for b in bytruth if len(b)),
            mean_regret=float(a[:,3].mean()),p95_regret=float(np.quantile(a[:,3],.95)),budget_certified_fraction=float(a[:,4].mean())))
    write(OUT/'bridge_detailed.json',result);write(OUT/'bridge_platform_prices.json',platform)
    return result

def additional_tables(records):
    groups=defaultdict(list)
    for r in records:
        if r['view']=='H63_stale63':groups[(r['stage'],r['phase'],r['N'],r['M'],r['tau'],r['method'])].append(r)
    allgrid=[]
    for k,rr in groups.items():
        good=[r for r in rr if r['regret_upper'] is not None]
        allgrid.append(dict(stage=k[0],phase=k[1],N=k[2],M=k[3],tau=k[4],method=k[5],recorded=len(rr),evaluable=len(good),
            success_lower_recorded=mean([r['success_lower'] for r in rr]),success_upper_recorded=mean([r['success_upper'] for r in rr]),
            success_evaluable=mean([r['success_lower'] for r in good]),regret_over_u=mean([r['regret_upper']/r['u'] for r in good]),
            loss_over_u=mean([r['loss']/r['u'] for r in good]),solver_cpu_sum=sum(r['solver_cpu'] for r in rr)))
    write(OUT/'all_threshold_budget_scale_tables.json',allgrid)
    selected=[r for r in records if r['stage']=='R2' and r['phase']=='test' and r['tau']==1]
    common=[];crossfit=[];quality=[]
    for method in sorted(set(r['method'] for r in selected)):
        a={r['id']:r for r in selected if r['method']==method and r['view']=='H63_stale63' and r['regret_upper'] is not None}
        b={r['id']:r for r in selected if r['method']==method and r['view']=='H126_stale63' and r['regret_upper'] is not None}
        ids=sorted(a.keys()&b.keys())
        common.append(dict(method=method,common_instances=len(ids),H63_success=mean([a[i]['success_lower'] for i in ids]),H126_success=mean([b[i]['success_lower'] for i in ids]),
            H63_regret_over_u=mean([a[i]['regret_upper']/a[i]['u'] for i in ids]),H126_regret_over_u=mean([b[i]['regret_upper']/b[i]['u'] for i in ids])))
        z=[r for r in selected if r['method']==method and r['view']=='H126_crossfit63' and r['signed_excess_risk'] is not None]
        crossfit.append(dict(method=method,evaluable=len(z),signed_excess_over_u=mean([r['signed_excess_risk']/r['u'] for r in z])))
    for view in ['H63_stale0','H63_stale20','H63_stale63','H63_stale126']:
        rr=[r for r in selected if r['method']=='MA:0.0' and r['view']==view]
        quality.append(dict(view=view,recorded=len(rr),status=dict(Counter(r['status'] for r in rr)),quality_fields=[r['quality'] for r in rr if r['quality']]))
    write(OUT/'common_H63_H126.json',common);write(OUT/'crossfit_signed_excess.json',crossfit);write(OUT/'evaluation_quality.json',quality)
    return allgrid

def private_audit():
    hashes=[];overlap=[];costs={};status=Counter();fail=[]
    for job in read(PRIVATE/'manifest.json'):
        path=PRIVATE/'decisions'/f'{job["id"]}.json'
        if not path.exists():continue
        r=read(path)
        if r['status']!='complete':continue
        costs.setdefault(r['shared_cost_id'],r['frontend_cpu'])
        cells=[c for c in r['cells'] if c['status']=='complete']
        for tau in set(c['tau_index'] for c in cells):
            cc=[c for c in cells if c['tau_index']==tau]
            hs={c['decision']['input_hash'] for c in cc};hashes.append(len(hs)==1)
            ma=next(c for c in cc if c['method']=='MA')['decision']
            for c in cc:
                d=c['decision'];status[d['raw_status']]+=1
                if c['method']=='A1B':overlap.append(dict(stage=job['stage'],phase=job['panel']['phase'],tau=tau,beta=c['beta'],max_weight_difference=float(np.max(np.abs(np.array(d['evaluated_target'])-ma['evaluated_target']))),internal_mass_gain=d['success_mass_lower']-ma['success_mass_lower']))
                if d['primal_residual']>1e-9:fail.append(dict(id=job['id'],method=c['method'],reason='evaluated_primal'))
    write(OUT/'postrun_integrity.json',dict(common_inputs=all(hashes),shared_inputs_checked=len(hashes),unique_frontends=len(costs),
        cold_frontend_cpu_sum=sum(costs.values()),statuses=dict(status),failures=fail,overlap=overlap))
    return dict(common_inputs=all(hashes),failures=len(fail),statuses=dict(status))

def extend_report(records,table):
    prices,front=r1_audit();bridge=detailed_bridge();grid=additional_tables(records);audit=private_audit()
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,domain in zip(axes,['csi300','csi500']):
        for method,beta in [('MA','0.0'),('A1B','0.1'),('Greedy','0.1')]:
            for view,style in [('Separated0','--'),('Joint0','-')]:
                rr=sorted([r for r in bridge if r['domain']==domain and r['threshold']=='geo0.75' and r['method']==method and r['beta']==beta and r['view']==view],key=lambda r:r['n'])
                if rr:ax.plot([r['n'] for r in rr],[r['success_lower'] for r in rr],style,marker='o',label=f'{method} {view}')
        ax.set_title(domain);ax.set_xlabel('Two-day blocks');ax.set_ylabel('Known-model success');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'bridge_information.png',dpi=160);plt.close(fig)
    lines=['','## R1：历史风险库的价格台阶与数值核验','',
        f'独立价格记录 {len(prices)} 个，预算前沿记录 {len(front)} 个；A1-B 持仓成功质量下界超出枚举可能质量上界的冲突 {sum(r["interval_conflict"] for r in front)} 个。CVXPY 的值只是独立建模参考，不当作严格对偶证书。',
        '逐集合价格、独立模型状态及区间距离见 `output/r1_price_audit.json`，前沿见 `output/r1_frontiers.json`。开放数值区间不被强行闭合。','',
        '## 历史校准的半合成机制','',
        '两个固定历史族、每真模型10次独立重复；同一次模拟数据用于两个等标量预算视图。先验行只评价一次决策，不算六次独立实验。下表是 n=512 的已知真模型结果；样本很小，不能外推全部风险族。','',
        '|历史族|阈值|视图|方法/预算|成功率|最坏真模型成功率|平均 regret|','|---|---|---|---:|---:|---:|---:|']
    for r in bridge:
        if r['n']==512 and r['method'] in ['MA','A1B','Greedy'] and r['beta'] in ['0.0','0.1','0.25']:
            lines.append(f'|{r["domain"]}|{r["threshold"]}|{r["view"]}|{r["method"]}/{r["beta"]}|{r["success_lower"]:.3%}|{r["worst_truth_success_lower"]:.3%}|{r["mean_regret"]:.7g}|')
    lines+=['','![半合成信息实验](output/bridge_information.png)','',
        '其余样本量、尾部 regret 和预算资格见 `output/bridge_detailed.json`。等价类极限后验的全部集合价格见 `output/bridge_platform_prices.json`；该枚举平台与有限样本实际达标率含义不同。','',
        '## R3、R4与敏感性','',
        '自然缺失面板和规模对照均使用同组内共同输入。所有阈值、预算、CVaR水平、N/M、2026延伸结果保存在 `output/all_threshold_budget_scale_tables.json`；不能从其中挑选最优一列代表预设主比较。信息实验逐日期的标量成本见 `output/information_summary.json`，Full使用双倍信息成本。',
        'H=126与H=63在公共日期子集的比较见 `output/common_H63_H126.json`；未来交叉拟合参照见 `output/crossfit_signed_excess.json`，有符号超额风险可为负。','',
        '## 完整性与解释','',
        f'方法输入哈希一致性：{audit["common_inputs"]}；已评价持仓约束不合格记录：{audit["failures"]}。模型内MA重合、回退与开放状态以及共享前端成本见 `output/postrun_integrity.json`。',
        '本轮冻结后新增的汇总代码只读取结果，不参与模型、阈值或持仓生成。完整指数扩展在测试前因开发计时预算排除；该缺口不能用N=100结果代替。',
        '','## 三项最终判断的使用原则','',
        '半合成只支持这两个历史校准族中的机制判断；真实结果只有固定月份配对区间及成本敏感性共同支持时，才可说有稳健改善。若算法与MA或贪心重合，记为真实风险库中的前沿退化或没有额外增益，不能转用几何阈值包装成主要实证成功。']
    with (ROOT/'REPORT_ZH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
