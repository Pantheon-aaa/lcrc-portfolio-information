"""Read-only additional summaries; this module never feeds experiment decisions."""
from collections import Counter,defaultdict
import numpy as np
from .common import *

def finite(x):
    try:return np.isfinite(float(x))
    except (TypeError,ValueError):return False

def mean(x):return float(np.mean(x)) if len(x) else None

def r1_audit():
    rows=[];frontiers=[];jumps=[]
    for path in (PRIVATE/'r1').glob('*.json'):
        r=read(path)
        if r['status']!='complete':continue
        for z in r['prices']:
            cv=z['cvx'];v=cv.get('value');lo=z['lower'];hi=z['upper']
            # CVXPY is an independent formulation, not a certified dual bound.
            distance=max(float(lo)-float(v),float(v)-float(hi),0.) if all(finite(x) for x in [lo,hi,v]) else None
            rows.append(dict(id=r['id'],threshold=r['threshold'],G=z['G'],subset_size=len(z['G']),mass=z['mass'],lower=lo,upper=hi,
                             feasible=z['feasible'],status=z['status'],cvx_status=cv['status'],cvx_interval_distance=distance))
        reference=next((d['raw_result'] for d in r['results'] if d['method']=='A1B' and d.get('raw_result')),None)
        if reference:
            Llo=reference['posterior_mean_regret_MA_lower'];Lhi=reference['posterior_mean_regret_MA_upper']
            for z in r['prices']:
                if not z['feasible'] or not finite(z['upper']):continue
                jumps.append(dict(id=r['id'],threshold=r['threshold'],G=z['G'],mass=z['mass'],
                    beta_lower=max(0,float(z['lower']))/Lhi if Lhi>0 else None,
                    beta_upper=max(0,float(z['upper']))/Llo if Llo>0 else None))
        for d in r['results']:
            if d['method']!='A1B':continue
            blo=d['budget_lower']/r['u'] if finite(d['budget_lower']) else np.inf
            bhi=d['budget_upper']/r['u'] if finite(d['budget_upper']) else np.inf
            feasible=[z for z in r['prices'] if z['feasible'] and float(z['upper'])<=blo]
            possible=[z for z in r['prices'] if z['status']!='infeasible' and (z['lower'] is None or float(z['lower'])<=bhi)]
            lower=max([z['mass'] for z in feasible]+[0.]);upper=max([z['mass'] for z in possible]+[0.])
            raw=d.get('raw_result') or {}
            frontiers.append(dict(id=r['id'],domain=r['domain'],threshold=r['threshold'],beta=d['beta'],
                actual_lower=d['success_mass_lower'],actual_upper=d['success_mass_upper'],enumeration_lower=lower,enumeration_upper=upper,
                increment_over_u=d['mean_increment_upper']/r['u'],budget_certified=d['budget_certified'],status=d['raw_status'],
                primary_gap=raw.get('primary_mass_gap'),secondary_gap=raw.get('secondary_price_gap'),
                interval_conflict=bool(d['success_mass_lower']>upper+1e-8),
                primary_certificate_conflict=bool(raw.get('optimal_mass_upper',1.)<lower-1e-8)))
    write(OUT/'r1_price_audit.json',rows);write(OUT/'r1_frontiers.json',frontiers);write(OUT/'r1_budget_jump_intervals.json',jumps)
    return rows,frontiers

def detailed_bridge():
    grouped=defaultdict(list);platform=[];limits=[];families={}
    from research_v4a.mean_budget import Family,MeanBudget
    for path in (PRIVATE/'semisynthetic').glob('*/*.json'):
        r=read(path)
        if path.name.startswith('platform'):
            platform.append(dict(domain=path.parent.name,h=r['h'],threshold=r['threshold'],prices=[{k:z[k] for k in ['G','mass','lower','upper','feasible','status']} for z in r['prices']]))
            domain=path.parent.name
            if domain not in families:
                q=np.load(path.parent/'family.npz');families[domain]=Family(q['Q']/float(q['u']),.2)
            p=np.zeros(6);p[r['h']*3:r['h']*3+3]=1/3;mb=MeanBudget(families[domain],p)
            for beta in [0,.1,.25]:
                blo,bhi=mb.bounds(beta)
                lower=max([z['mass'] for z in r['prices'] if z['feasible'] and float(z['upper'])<=blo]+[0.])
                upper=max([z['mass'] for z in r['prices'] if z['lower'] is None or float(z['lower'])<=bhi]+[0.])
                limits.append(dict(domain=domain,h=r['h'],threshold=r['threshold'],beta=beta,success_lower=lower,success_upper=upper,
                    mean_regret_MA_lower=mb.Llo,mean_regret_MA_upper=mb.Lhi,source='independent subset enumeration under limiting class posterior'))
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
    write(OUT/'bridge_detailed.json',result);write(OUT/'bridge_platform_prices.json',platform);write(OUT/'bridge_class_platforms.json',limits)
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
    groups=defaultdict(list)
    for r in records:
        if r['stage']=='R2' and r['view']=='H63_stale63':groups[(r['phase'],r['domain'],r['date'][:4],r['tau'],r['method'])].append(r)
    write(OUT/'domain_year_tables.json',[dict(phase=k[0],domain=k[1],year=k[2],tau=k[3],method=k[4],recorded=len(rr),
        evaluable=sum(r['regret_upper'] is not None for r in rr),success_lower_recorded=mean([r['success_lower'] for r in rr]),success_upper_recorded=mean([r['success_upper'] for r in rr]),
        mean_loss_over_u_evaluable=mean([r['loss']/r['u'] for r in rr if r['loss'] is not None])) for k,rr in groups.items()])
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

def information_pairs():
    grouped=defaultdict(list);missing=[]
    for path in (PRIVATE/'information').glob('*.json'):
        r=read(path)
        if r['status']!='ready':missing.append(dict(id=r['id'],n=r['n'],view=r['view'],status=r['status']));continue
        for c in r['cells']:
            k=(r['date'],r['domain'],r['phase'],r['n'],c['method'],c['beta'],r['view'].rstrip('01'))
            e=c['evaluation'];p=np.array(r['posterior'])
            grouped[k].append(dict(lower=e['success_lower'],upper=e['success_upper'],loss=e.get('loss'),u=r['u'],scalar_count=r['scalar_count'],ESS=float(1/(p@p)),internal_success=c['decision']['success_mass_lower']))
    averaged=[];pairs=[]
    for k,rr in grouped.items():
        averaged.append(dict(date=k[0],domain=k[1],phase=k[2],n=k[3],method=k[4],beta=k[5],view=k[6],directions=len(rr),
            success_lower=mean([r['lower'] for r in rr]),success_upper=mean([r['upper'] for r in rr]),ESS=mean([r['ESS'] for r in rr]),
            internal_success=mean([r['internal_success'] for r in rr]),scalar_count=mean([r['scalar_count'] for r in rr]),
            loss_over_u=mean([r['loss']/r['u'] for r in rr]) if all(r['loss'] is not None for r in rr) else None))
    lookup={(r['date'],r['domain'],r['n'],r['method'],r['beta'],r['view']):r for r in averaged}
    for r in averaged:
        if r['view']!='Joint':continue
        s=lookup.get((r['date'],r['domain'],r['n'],r['method'],r['beta'],'Separated'))
        if s is None:continue
        assert s['scalar_count']==r['scalar_count']
        pairs.append({k:r[k] for k in ['date','domain','phase','n','method','beta','scalar_count']}|dict(success_difference_lower=r['success_lower']-s['success_upper'],success_difference_upper=r['success_upper']-s['success_lower'],
            loss_difference_over_u=r['loss_over_u']-s['loss_over_u'] if r['loss_over_u'] is not None and s['loss_over_u'] is not None else None))
    write(OUT/'information_orientation_average.json',averaged);write(OUT/'information_paired_responses.json',pairs);write(OUT/'information_unresolved.json',missing)

def library_diagnostic():
    groups=defaultdict(dict)
    for j in read(PRIVATE/'manifest.json'):
        if j['stage']!='F2-library':continue
        path=PRIVATE/'decisions'/f'{j["id"]}.json'
        if not path.exists():continue
        r=read(path)
        if r['status']=='complete':groups[(j['panel']['date'],j['panel']['domain'],j['N'])][j['offset']]=r
    output=[]
    for k,rr in groups.items():
        if len(rr)!=3:continue
        for beta in [.1,.25]:
            get=lambda r,m:np.array(next(c['decision']['evaluated_target'] for c in r['cells'] if c['status']=='complete' and c['method']==m and c['beta']==(0 if m=='MA' else beta)))
            ma=[get(rr[o],'MA') for o in [0,10000,20000]];aa=[get(rr[o],'A1B') for o in [0,10000,20000]]
            within=mean([float(np.linalg.norm(a-m,1)) for a,m in zip(aa,ma)])
            random=mean([float(np.linalg.norm(aa[i]-aa[j],1)) for i,j in [(0,1),(0,2),(1,2)]])
            output.append(dict(date=k[0],domain=k[1],N=k[2],beta=beta,method_L1_difference=within,library_L1_difference=random,library_over_method_ratio=random/within if within>1e-12 else None))
    write(OUT/'F2_library_variation.json',output)

def book_tables():
    path=OUT/'book_summary.json'
    if not path.exists():return []
    grouped=defaultdict(list);endpoint_records=[]
    for bp in (PRIVATE/'books').glob('*.json'):
        book=read(bp);r=book['summary'];daily=book['daily'];last=daily[-1]['nav']
        endpoints=last is not None
        r=dict(r,endpoint_annual_return=(last/1e6)**(252/len(daily))-1 if endpoints else None)
        endpoint_records.append({k:r[k] for k in ['domain','seed','method_beta','cost_bps','complete','unresolved_days','endpoint_annual_return']}|dict(final_nav=last))
        grouped[(r['method_beta'],r['cost_bps'])].append(r)
    out=[]
    for k,rr in grouped.items():
        good=[r for r in rr if r['complete']]
        out.append(dict(method=k[0],cost_bps=k[1],books=len(rr),complete=len(good),unresolved_days=sum(r['unresolved_days'] for r in rr),
            endpoint_evaluable=sum(r['endpoint_annual_return'] is not None for r in rr),annual_return_endpoints=mean([r['endpoint_annual_return'] for r in rr if r['endpoint_annual_return'] is not None]),
            annual_return_complete=mean([r['annual_return'] for r in good]),annual_vol_complete=mean([r['annual_vol'] for r in good]),
            max_drawdown_complete=mean([r['max_drawdown'] for r in good]),sharpe_complete=mean([r['sharpe_zero_rate'] for r in good if r.get('sharpe_zero_rate') is not None]),
            turnover_observed=mean([r['turnover_sum'] for r in rr]),fees_observed=mean([r['total_fees'] for r in rr]),exposure_observed=mean([r['mean_exposure'] for r in rr])))
    write(OUT/'book_cost_tables.json',out)
    write(OUT/'book_endpoint_returns.json',endpoint_records)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));curves={}
    for method in ['MA_b0.0','A1B_b0.1','A1B_b0.25','Greedy_b0.1','EMGMV_b0.0','EW_b0.0']:
        paths=list((PRIVATE/'books').glob(f'*_{method}_20.json'))
        if len(paths)!=6:continue
        books=[read(p)['daily'] for p in paths];days=[r['date'] for r in books[0]]
        x=np.array([[np.nan if r['nav'] is None else r['nav']/1e6 for r in b] for b in books]);y=x.mean(0)
        curves[method]=dict(dates=days,nav=y,rule='six books equally weighted; any unknown component makes aggregate unknown')
        ax.plot(np.arange(len(days)),y,label=method)
    if curves:
        step=max(1,len(days)//6);ax.set_xticks(np.arange(0,len(days),step),[days[i][:7] for i in range(0,len(days),step)])
        ax.legend(fontsize=7);ax.set_ylabel('Total-return research NAV / initial capital');ax.set_title('20 bp each side; gaps remain unknown');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'research_nav.png',dpi=160)
    plt.close(fig);write(OUT/'research_nav_aggregate.json',curves)
    return out

def private_audit():
    hashes=[];overlap=[];costs={};status=Counter();fail=[];end_to_end=[]
    for job in read(PRIVATE/'manifest.json'):
        path=PRIVATE/'decisions'/f'{job["id"]}.json'
        if not path.exists():continue
        r=read(path)
        if r['status']!='complete':continue
        costs.setdefault(r['shared_cost_id'],r['frontend_cpu'])
        end_to_end.append(dict(id=job['id'],stage=job['stage'],N=job['N'],M=job['M'],shared_cost_id=r['shared_cost_id'],
            measured_job_cpu=r['cpu_seconds'],frontend_cold_reference_cpu=r['frontend_cpu'],method_calls_cpu=sum(c.get('measured_cpu',0) for c in r['cells']),
            accounting='job CPU includes actual cache loads, shared QP preparation and future evaluation; cold frontend reference must not be added again to job CPU'))
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
    write(OUT/'computational_costs.json',end_to_end)
    return dict(common_inputs=all(hashes),failures=len(fail),statuses=dict(status))

def frontend_diagnostics():
    current=Counter();scoring=Counter();rows=[]
    for path in (PRIVATE/'frontends').glob('*.json'):
        r=read(path)
        for m in r['current_models']:current[m['status']]+=1
        scoring.update(r['score_fit_status'])
        rows.append(dict(shared_cost_id=r['shared_cost_id'],date=r['date'],N=r['N'],M=r['M'],cpu_seconds=r['cpu_seconds'],wall_seconds=r['wall_seconds'],
            eigen_min=min(r['current_eigen_min']),condition_max=max(r['current_condition']),
            current_unconverged=sum(m['status']=='max_iterations' for m in r['current_models'])))
    write(OUT/'frontend_diagnostics.json',dict(current_model_status=dict(current),prequential_model_status=dict(scoring),unique_cache_records=len(rows),
        counting_rule='model records, including shrinkage models sharing one EM fit; not independent repetitions',records=rows))
    old=read(ROOT.parent/'research_v5_ashare/output/data_audit.json')
    write(OUT/'data_inventory.json',dict(inherited_read_only_audit=old,valuation_audit=read(OUT/'valuation_audit.json'),
        source_absolute_path=str(SOURCE),membership_root=str(MEMBERSHIP),training_cache_read_only=str(CACHE),
        source_identity='same source SHA verified again at v5b valuation construction and final delivery'))

def extend_report(records,table):
    prices,front=r1_audit();bridge=detailed_bridge();grid=additional_tables(records);audit=private_audit()
    information_pairs();library_diagnostic();books=book_tables();frontend_diagnostics()
    import matplotlib.pyplot as plt
    for threshold in ['tau','geo0.5','geo0.75','geo1.05']:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,domain in zip(axes,['csi300','csi500']):
            for method,beta in [('MA','0.0'),('A1B','0.1'),('Greedy','0.1')]:
                for view,style in [('Separated0','--'),('Joint0','-')]:
                    rr=sorted([r for r in bridge if r['domain']==domain and r['threshold']==threshold and r['method']==method and r['beta']==beta and r['view']==view],key=lambda r:r['n'])
                    if rr:ax.plot([r['n'] for r in rr],[r['success_lower'] for r in rr],style,marker='o',label=f'{method} {view}')
            ax.set_title(f'{domain}, threshold={threshold}');ax.set_xlabel('Two-day blocks');ax.set_ylabel('Known-model success');ax.set_ylim(-.03,1.03);ax.grid(alpha=.2)
        axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/f'bridge_{threshold}.png',dpi=160)
        if threshold=='geo0.75':fig.savefig(OUT/'bridge_information.png',dpi=160)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for threshold in ['tau0','tau1','tau2','geo0.5','geo0.75','geo1.05']:
        series=[mean([r['actual_lower'] for r in front if r['threshold']==threshold and float(r['beta'])==float(b)]) for b in [0,.1,.25,.5,'inf']]
        if any(v is not None for v in series):ax.plot(range(5),[np.nan if v is None else v for v in series],marker='o',label=threshold)
    ax.set_xticks(range(5),['0','.10','.25','.50','inf']);ax.set_xlabel('Mean-regret budget beta');ax.set_ylabel('Mean posterior success mass');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'r1_frontier.png',dpi=160);plt.close(fig)
    lines=['','## R1：历史风险库的价格台阶与数值核验','',
        f'独立价格记录 {len(prices)} 个，预算前沿记录 {len(front)} 个；A1-B 持仓成功质量下界超出枚举可能质量上界的冲突 {sum(r["interval_conflict"] for r in front)} 个。CVXPY 的值只是独立建模参考，不当作严格对偶证书。',
        '逐集合价格、独立模型状态及区间距离见 `output/r1_price_audit.json`，前沿见 `output/r1_frontiers.json`。开放数值区间不被强行闭合。','',
        '![历史库内部前沿](output/r1_frontier.png)','所有可行子集的预算跳点区间见 `output/r1_budget_jump_intervals.json`；不是仅以五个绘图点替代完整枚举。','',
        '## 历史校准的半合成机制','',
        '两个固定历史族、每真模型10次独立重复；同一次模拟数据用于两个等标量预算视图。先验行只评价一次决策，不算六次独立实验。下表是 n=512 的已知真模型结果；样本很小，不能外推全部风险族。','',
        '|历史族|阈值|视图|方法/预算|成功率|最坏真模型成功率|平均 regret|','|---|---|---|---:|---:|---:|---:|']
    for r in bridge:
        if r['n']==512 and r['method'] in ['MA','A1B','Greedy'] and r['beta'] in ['0.0','0.1','0.25']:
            lines.append(f'|{r["domain"]}|{r["threshold"]}|{r["view"]}|{r["method"]}/{r["beta"]}|{r["success_lower"]:.3%}|{r["worst_truth_success_lower"]:.3%}|{r["mean_regret"]:.7g}|')
    lines+=['','![半合成0.5Gamma](output/bridge_geo0.5.png)','![半合成0.75Gamma](output/bridge_geo0.75.png)',
        '![半合成1.05Gamma](output/bridge_geo1.05.png)','![半合成经验阈值](output/bridge_tau.png)','',
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
    lines=['','## 总收益指数研究账本与成本','',
        '下表仅对净值全程可评价的账本计算年化风险结果；完整数少于6时，不得解释为六个预设面板的总体表现。若只是中途估值陈旧、最终可靠累计估值已恢复，可识别的首末端点年化收益另见 `output/book_endpoint_returns.json`，仍不填造中间日收益或补算Sharpe/回撤。费用、换手和暴露为实际可观察期间的诊断。各成本场景完整结果见 `output/book_cost_tables.json`。','',
        '|方法|每边成本bp|完整账本/6|完整账本平均年化收益|平均年化波动|观察期平均暴露|','|---|---:|---:|---:|---:|---:|']
    for r in books:
        if r['cost_bps']!=20:continue
        a='未决' if r['annual_return_complete'] is None else f'{r["annual_return_complete"]:.2%}'
        v='未决' if r['annual_vol_complete'] is None else f'{r["annual_vol_complete"]:.2%}'
        lines.append(f'|{r["method"]}|20|{r["complete"]}/6|{a}|{v}|{r["exposure_observed"]:.2%}|')
    if (OUT/'research_nav.png').exists():lines+=['','![研究账本净值](output/research_nav.png)']
    lines+=['','净值图严格按六个面板等权；任一个组成账本未知时，相应总体曲线留空，不改用剩余面板重新归一化。']
    with (ROOT/'REPORT_ZH.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    main={r['method']:r for r in table if r['stage']=='R2'}
    integrity=read(OUT/'postrun_integrity.json')
    overlap=[r for r in integrity['overlap'] if r['stage']=='R2' and r['phase']=='test' and r['tau']==1 and r['beta'] in [.1,.25]]
    identical=bool(overlap) and all(r['max_weight_difference']==0 for r in overlap)
    a=main.get('A1B:0.1');m=main.get('MA:0.0');conclusions=['## 本轮结果判断','']
    if a and m:
        conclusions.append(f'**近似真实主实验没有显示A1-B相对MA的额外达标收益。** 主阈值下，两者在各自{a["evaluable"]}个可评价记录上的达标率分别为{a["success_common"]:.3%}和{m["success_common"]:.3%}；平均regret/u分别为{a["regret_over_u_common"]:.5f}和{m["regret_over_u_common"]:.5f}。')
        if identical:conclusions.append(f'更直接的证据是：{len(overlap)}个主预算查询（同一实例的两档预算不算独立样本）的持仓均逐元素等于MA。当前库下MA已覆盖全部模型ε集合，主预算前沿平坦，不能将其相对等权的优势归因于A1-B。')
    selected=[r for r in bridge if r['n']==512 and r['view']=='Separated0' and r['threshold']=='geo0.5' and r['method'] in ['MA','A1B','Greedy'] and r['beta'] in ['0.0','0.1']]
    conclusions+=['','**半合成支持局部机制，但不支持精确求解器普遍胜过简单Greedy。** 在两个固定历史族、0.5Γ阈值、512块Separated观测下：']
    for domain in ['csi300','csi500']:
        rr={r['method']:r for r in selected if r['domain']==domain}
        if len(rr)==3:conclusions.append(f'- {domain}：MA {rr["MA"]["success_lower"]:.2%}，A1-B(β=0.10) {rr["A1B"]["success_lower"]:.2%}，Greedy {rr["Greedy"]["success_lower"]:.2%}；A1-B最坏真模型成功率{rr["A1B"]["worst_truth_success_lower"]:.2%}。')
    conclusions+=['','这个分离针对平均模型成功率，不能写成所有市场均受保护；Joint充分识别后各方法均可达标，也不能归因于A1-B独有的信息利用优势。',
        '','**论文层面的结论：保留机制证据和真实前沿退化证据，不主张本轮已验证真实市场增益。** 真实目标与MA相同时，没有独立的平均风险或交易成本收益。部分账本路径仍未决，不能以少数完整账本的收益排行补强结论。更丰富风险库是否必要属于后续新研究问题；本轮没有按测试表现更换模型或阈值。','']
    report=(ROOT/'REPORT_ZH.md').read_text(encoding='utf-8');first,rest=report.split('\n',1)
    (ROOT/'REPORT_ZH.md').write_text(first+'\n\n'+'\n'.join(conclusions)+rest,encoding='utf-8')
