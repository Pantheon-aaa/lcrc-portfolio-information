"""Deterministic summaries, paired uncertainty and publication figures."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,OUT,CFG,rows,write,seed,timed_stage

def load(name):
    p=OUT/name
    if not p.exists() and Path(str(p)+'.gz').exists():p=Path(str(p)+'.gz')
    return pd.read_json(p,lines=True) if p.exists() else pd.DataFrame()

def md(df):
    if df.empty:return '尚无结果。'
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x)
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def holm(values):
    p=np.array(values);out=np.full(len(p),np.nan);ix=np.flatnonzero(np.isfinite(p));order=ix[np.argsort(p[ix])]
    if len(order):out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
    return out

def macro_summary(df,groups):
    good=df[df.regret.notna()].copy()
    strata=[c for c in ['family','d','M','nats','singleton_per_asset','epsilon_ratio','truth'] if c in good]
    keys=list(dict.fromkeys(groups+strata))
    vals=['true_success','true_possible_success','posterior_mass_lower','posterior_mass_upper','regret','regret_upper']
    agg=good.groupby(keys,dropna=False)[vals].mean().reset_index()
    result=agg.groupby(groups,dropna=False)[vals].mean().reset_index()
    counts=df.groupby(groups,dropna=False).size().reset_index(name='attempted')
    valid=good.groupby(groups,dropna=False).size().reset_index(name='valid')
    result=result.merge(counts,on=groups).merge(valid,on=groups)
    p95=good.groupby(groups,dropna=False).regret.quantile(.95).reset_index(name='pooled_p95_regret')
    result=result.merge(p95,on=groups)
    worst=agg.groupby(groups,dropna=False).true_success.min().reset_index(name='worst_cell_success')
    return result.merge(worst,on=groups)

def paired(df,candidate,baseline,metric,group_columns,greater=True):
    """Conditional fixed-design, stratified paired normal intervals.

    Independent datasets form clusters. Epsilon thresholds share one dataset and
    are averaged inside that cluster before estimating uncertainty. Truth strata
    and design cells receive equal weights; no random-family inference is claimed.
    """
    result=[]
    group_arg=group_columns[0] if len(group_columns)==1 else group_columns
    for group,part in df.groupby(group_arg,dropna=False):
        a=part[part.method==candidate];b=part[part.method==baseline]
        on=['dataset']+(['epsilon_ratio'] if 'epsilon_ratio' in part else [])
        merged=a.merge(b,on=on,suffixes=('_a','_b'))
        if merged.empty:continue
        x=pd.to_numeric(merged[metric+'_a'],errors='coerce').astype(float)-pd.to_numeric(merged[metric+'_b'],errors='coerce').astype(float)
        merged=merged.assign(difference=x).dropna(subset=['difference'])
        strata=[c+'_a' for c in ['family','d','M','nats','singleton_per_asset','truth'] if c+'_a' in merged]
        collapsed=merged.groupby(strata+['dataset'],dropna=False).difference.mean().reset_index()
        stats=collapsed.groupby(strata,dropna=False).difference.agg(['mean','var','count'])
        estimate=float(stats['mean'].mean());var=float((stats['var'].fillna(0)/stats['count']).sum()/len(stats)**2);se=np.sqrt(var)
        z=estimate/se if se>0 else (np.inf if estimate>0 else (-np.inf if estimate<0 else 0.))
        p=float(norm.sf(z) if greater else norm.cdf(z))
        g=group if isinstance(group,tuple) else (group,)
        result.append({**dict(zip(group_columns,g)),'candidate':candidate,'baseline':baseline,'metric':metric,
          'difference':estimate,'ci95_lower':estimate-1.96*se,'ci95_upper':estimate+1.96*se,'p_one_sided':p,
          'datasets':len(collapsed),'design_truth_strata':len(stats),'inference':'conditional on fixed generated families'})
    return pd.DataFrame(result)

def figures(e1,e2,e3,e4):
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    a=e1[e1.nats==0].groupby('method')[['posterior_mass_lower','regret']].mean()
    methods=['MA','CVaR','A1','Hinge','Greedy','RandomPair'];a=a.loc[methods]
    axes[0].bar(methods,a.posterior_mass_lower,color=['#777','#999','#2171b5','#a1d99b','#6baed6','#fdae6b']);axes[0].set_ylabel('Posterior epsilon-success mass');axes[0].set_ylim(0,1)
    axes[0].tick_params(axis='x',labelrotation=25,labelsize=9)
    for method in ('MA','CVaR','A1','Hinge','Greedy'):
        curve=e2[e2.method==method].groupby('nats').true_success.mean()
        axes[1].plot(curve.index,curve.values,'o-',label=method)
    axes[1].set_xlabel('Target joint information (nats)');axes[1].set_ylabel('Success rate (confirmation; pooled)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'compatibility.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for method in ('MA','CVaR','B1','LCRC'):
        for truth,ax in [(0,axes[0]),(1,axes[1])]:
            curve=e3[(e3.method==method)&(e3.truth==truth)].groupby('multiplier').regret_over_t2.mean()
            ax.plot(curve.index,curve.values,'o-',label=method)
    axes[0].set_title('Good market');axes[1].set_title('Bad market')
    for ax in axes:ax.set_xlabel('Target multiplier of log(1/(delta*t))/KL');ax.set_ylabel('Expected regret / t^2');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'evidence.png');plt.close(fig)
    if not e4.empty:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for method in ('MA-all','LCRC-all','B2','CS-Safe','MA-pilot'):
            g=e4[e4.method==method].groupby('nats')[['harmful_change','true_change_from_pilot']].mean()
            axes[0].plot(g.index,g.harmful_change,'o-',label=method);axes[1].plot(g.index,-g.true_change_from_pilot,'o-',label=method)
        axes[0].axhline(.005,ls='--',color='black',label='B2 target .005');axes[0].set_ylabel('Harmful-change frequency')
        axes[0].set_ylim(-.0003,.01)
        axes[1].set_ylabel('Mean improvement over pilot MA')
        axes[1].set_title('Differences at numerical precision scale',fontsize=9)
        for ax in axes:ax.set_xlabel('Gate information target (nats)');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(OUT/'multidirection.png');plt.close(fig)

def run(budget):
    e1=load('E1.jsonl');e2=load('E2_confirmation.jsonl');dev=load('E2_development.jsonl');e3=load('E3.jsonl');e4=load('E4_confirmation.jsonl')
    if e2.empty:e2=dev.copy()
    if e4.empty:e4=load('E4_development.jsonl')
    safe=load('E4_confirmation_safe_reference.jsonl')
    if not safe.empty:e4=pd.concat([e4,safe],ignore_index=True)
    main=e2[(e2.epsilon_ratio!=1)&e2.regret.notna()]
    macro_summary(e1,['method','t','nats','singleton_per_asset']).to_csv(OUT/'E1_cells.csv',index=False)
    e3.groupby(['d','t','truth','multiplier','method']).agg(
      mean_regret=('regret','mean'),p95_regret=('regret',lambda x:x.quantile(.95)),
      mean_scaled_regret=('regret_over_t2','mean'),false_unlock=('false_unlock','mean'),
      false_reject=('false_reject','mean'),joint_rows=('joint_rows','first')).to_csv(OUT/'E3_all_methods.csv')
    summary=macro_summary(main,['kind','method']);summary.to_csv(OUT/'E2_summary.csv',index=False)
    percell=macro_summary(e2,['kind','d','M','nats','singleton_per_asset','epsilon_ratio','method']);percell.to_csv(OUT/'E2_cells.csv',index=False)
    comparisons=pd.concat([paired(main,'A1',b,'true_success',['kind']) for b in ('MA','CVaR')],ignore_index=True)
    if len(comparisons):comparisons['p_holm']=holm(comparisons.p_one_sided);comparisons.to_csv(OUT/'E2_paired.csv',index=False)
    algorithm=paired(main,'A1','Greedy','true_success',['kind']);algorithm.to_csv(OUT/'E2_vs_greedy.csv',index=False)
    calibration=e3[e3.method=='B1'].groupby(['d','t','truth','multiplier','joint_rows']).agg(
      n=('regret','size'),false_unlock=('false_unlock','sum'),false_reject=('false_reject','sum'),mean_scaled_regret=('regret_over_t2','mean')).reset_index()
    calibration['error_count']=np.where(calibration.truth==1,calibration.false_unlock,calibration.false_reject)
    calibration['cp95_lower']=[0 if k==0 else beta.ppf(.025,k,n-k+1) for k,n in zip(calibration.error_count,calibration.n)]
    calibration['cp95_upper']=[1 if k==n else beta.ppf(.975,k+1,n-k) for k,n in zip(calibration.error_count,calibration.n)]
    calibration.to_csv(OUT/'E3_binomial.csv',index=False)
    four=pd.DataFrame();fourpairs=pd.DataFrame()
    if not e4.empty:
        four=e4.groupby('method').agg(n=('regret','size'),mean_regret=('regret','mean'),success=('true_success','mean'),
          p95_regret=('regret',lambda x:x.quantile(.95)),harmful=('harmful_change','mean'),
          beneficial=('beneficial_change','mean'),mean_change=('true_change_from_pilot','mean')).reset_index()
        four.to_csv(OUT/'E4_summary.csv',index=False)
        fourpairs=pd.concat([paired(e4,'B2',b,metric,['d'],greater=False) for b in ('LCRC-all','MA-pilot','CS-Safe') for metric in ('harmful_change','true_change_from_pilot')],ignore_index=True)
        fourpairs['comparison_status']=np.where(fourpairs.baseline=='CS-Safe','supplementary_equal_budget','original_primary')
        fourpairs['p_holm']=fourpairs.groupby('comparison_status').p_one_sided.transform(lambda p:holm(p))
        fourpairs.to_csv(OUT/'E4_paired.csv',index=False)
    figures(e1,main,e3,e4)
    ledger=list(rows(OUT/'runtime_ledger.jsonl'));cpu=sum(x['cpu_seconds'] for x in ledger);wall=sum(x['wall_seconds'] for x in ledger)
    e1table=e1[e1.nats==0].groupby('method')[['posterior_mass_lower','true_success','regret']].mean().reset_index()
    e1truth=e1[e1.nats==0].groupby(['method','truth']).agg(success=('true_success','mean'),
      mean_regret=('regret','mean'),p95_regret=('regret',lambda x:x.quantile(.95))).reset_index()
    e1truth.to_csv(OUT/'E1_no_information_by_truth.csv',index=False)
    worst=1-e1truth.groupby('method').success.min()
    e1table['worst_model_failure']=e1table.method.map(worst)
    display=summary[['kind','method','true_success','regret','worst_cell_success','attempted','valid']]
    display=display[display.method.isin(['A1','MA','CVaR','Hinge','Greedy','LCRC'])]
    rates=e3[e3.method=='B1'].groupby(['truth','multiplier'])[['false_unlock','false_reject','regret_over_t2']].mean().reset_index()
    rare=pd.DataFrame(list(rows(OUT/'E3_rare.jsonl')))
    precise=pd.DataFrame(list(rows(OUT/'E3_rare_precision.jsonl')))
    failures=sum(int(df.get('error',pd.Series(dtype=str)).notna().sum()) for df in (e1,e2,dev,e4))
    a1=main[main.method=='A1'];statuses=a1.get('status',pd.Series(dtype=str)).value_counts().to_dict()
    a1gap=float((a1.mass_upper-a1.mass_lower).max())
    timing=pd.DataFrame(json.loads((OUT/'algorithm_timing.json').read_text())['records'])
    cold=timing[~timing.warm_geometry_cache][['d','M','method','wall_seconds','new_gamma_calls','certified_mass']]
    computation=a1.groupby(['kind','d','M']).agg(mean_seconds=('seconds','mean'),max_seconds=('seconds','max'),
      mean_cached_cuts=('cuts','mean')).reset_index()
    computation.to_csv(OUT/'A1_computation.csv',index=False)
    b2=e4[e4.method=='B2'] if not e4.empty else pd.DataFrame()
    b2_approvals=float(b2.approved_count.mean()) if len(b2) else 0
    trade=[]
    for kind in summary.kind.unique():
        a=summary[(summary.kind==kind)&(summary.method=='A1')].iloc[0]
        m=summary[(summary.kind==kind)&(summary.method=='MA')].iloc[0]
        trade.append({'kind':kind,'success_gain_pp':100*(a.true_success-m.true_success),'mean_regret_increase_percent':100*(a.regret/m.regret-1)})
    trade=pd.DataFrame(trade);trade.to_csv(OUT/'A1_tradeoff.csv',index=False)
    posterior_diagnostics=[]
    for ev in rows(OUT/'E4_confirmation_evidence.jsonl'):
        ll=np.array(ev['pilot_loglik']);p=np.exp(ll-np.logaddexp.reduce(ll))
        posterior_diagnostics.append({'d':ev['d'],'max_pilot_mass':p.max(),'pilot_ESS':1/(p@p),
          'candidate_count':len(ev['candidate_weights']),'entry':sum(g['entry'] for g in ev['generation']['geometry']),
          'exit':sum(g['exit'] for g in ev['generation']['geometry']),'near_cap':sum(g['near_cap'] for g in ev['generation']['geometry'])})
    pdg=pd.DataFrame(posterior_diagnostics)
    postsummary=pdg.groupby('d').median().reset_index() if len(pdg) else pd.DataFrame()
    if len(pdg):pdg.to_csv(OUT/'E4_pilot_diagnostics.csv',index=False)
    unsafe_count=int(b2.harmful_approval.sum()) if len(b2) else 0
    unsafe_upper=float(beta.ppf(.95,unsafe_count+1,len(b2)-unsafe_count)) if len(b2)>unsafe_count else 1.
    cs=e4[e4.method=='CS-Safe']
    csdiff=None
    if len(cs) and len(b2):
        matched=b2[['dataset','regret']].merge(cs[['dataset','regret']],on='dataset',suffixes=('_B2','_CS'))
        csdiff=float(np.max(abs(matched.regret_B2-matched.regret_CS)))
    independent='confirmation' in set(e2['split'])
    text=f'''# 决策兼容性与证据闸门实验报告

## 研究问题与证据范围

本轮检验有限模型下两种不同目标：DCL 最大化 regret 不超过 ε 的后验质量；EGPL 在控制错误批准的前提下允许持仓调整。MA 仍是正确后验下的均值最优基线。实验的成功标准是可信判断，不要求新方法获胜。

主要权衡如下。成功率提升以百分点计，平均 regret 增加是相对 MA 的比例，二者不可互换：

{md(trade)}

B2 与同预算 CS-Safe 在确认数据上的最大逐实例 regret 差为 **{csdiff}**；这一比较使用同一冻结候选库。它直接检查逐候选闸门是否比共享置信集带来额外价值，不能用普通未校准 MA 的错误率代替这个判断。

E1 使用附件的三模型反例，E2 扩展到非对称及隐藏曲率，E3 对应原 Theorem 6 的双模型信息边界，E4 使用独立 pilot／gate 数据测试多方向安全性。E2 本表独立确认状态：**{independent}**。所有真实模型均属于预先给定有限候选族；本轮没有真实股票回测，也没有连续未知协方差保证。

## 实施与公平比较

代码从固定 v2 提交建立独立工作区。每个数据集的所有方法共享候选族、观测与约束；真实模型编号只用于模拟和评价。singleton 历史为每资产 32／256 条，联合样本按公开模型的最小正 pairwise KL 换算信息预算，并保留 100,000 行上限与实际达到的信息量。

高斯充分统计通过 Bartlett/Wishart 精确抽样，模拟的仍是随机数据证据；没有用 KL 期望代替似然。E3 的二维充分统计与完整高维似然已核验。DCL 的 QCQP 在缩放坐标上求解，使用独立 QP 弱对偶计算真实目标上下界。所有 epsilon_ratio=1 的边界行保存在原始结果，主比较排除这一无法稳定区分的数值边界。

计时阶段仅根据运行成本冻结 E2 每单元 50 次、E4 每单元 200 次，独立确认均为 50 次；E1／E3 为 200 次。未根据胜负挑选族或调整 ε。确认阶段非对称／隐藏曲率重新生成模型；标准正单纯形仍是相同解析结构的新数据。

## E1 机制与代理偏差

无共同观测时，下表汇总三个 t 与两种边际历史长度。精确后验质量不需要 Monte Carlo 推断；true_success 的小偏差来自每个模型的有限重复数以及随机 pair 的抽样。

worst_model_failure 是对三个真实模型分别汇总后取最大失败频率；它是有限重复的估计，RandomPair 的理论值为 1/3。

{md(e1table)}

MA、全局 CVaR 和 hinge 在对称构造中返回中心组合，达标质量为零；DCL 和简单贪心可达到 2/3。DCL 的优势因此属于评价目标分离，**不是冲突割优于简单算法的证明**。固定破同分的 DCL 会长期牺牲一个模型，不能把其平均成功率当作最坏模型保证。均匀随机选 pair center 才实现三模型对称的 2/3 成功概率；直接平均三个组合不会获得这个性质。

MA 的均值 regret 更优，与理论一致。所有组合及各真模型指标保存在原始行中。

![兼容性与成功率](results/compatibility.png)

## E2 非对称与隐藏曲率结果

下表为 ε 比例 0.25、0.5、0.75 的结果，按固定设计与真模型分层后等权平均。最差成功率是有限测试单元上的描述性最小值，不是总体 minimax 保证。regret 保持原始半方差单位，跨不同族的值不能解释成收益率。

{md(display)}

主要配对比较如下，difference 为 A1 减去对应基线的成功率。区间在固定生成族条件下计算；同一数据集的三个 ε 先合并，避免当成独立样本。Holm 只校正预先指定的主要比较，不能消除模型族覆盖有限的限制。

{md(comparisons[['kind','baseline','difference','ci95_lower','ci95_upper','p_holm']] if len(comparisons) else comparisons)}

与简单贪心的比较单独列出，未预注册为主优效检验，属于探索性算法增量证据：

{md(algorithm[['kind','difference','ci95_lower','ci95_upper']] if len(algorithm) else algorithm)}

原始条件后验成功目标上的最优性不自动推出频率学成功优势。即使确认中存在正差异，仍需检查它是否以均值 regret、低概率模型风险或计算耗时为代价。精确状态分布：`{statuses}`。逐单元、p95 regret、超时和模糊成功区间见 CSV 与原始 JSONL。

主比较中 A1 最大剩余质量 gap 为 **{a1gap:.6g}**。HiGHS 的同分选择可能受可行容差影响；非零 gap 保留为 `open_mass_gap`，不能称为精确最优。下表在另外的固定模型与后验上比较冷缓存调用（ε=0.5Γ）；枚举只返回集合最优值，未计入均值破同分，故其时间不能当作完全相同输出的端到端比较。

{md(cold)}

在此六模型样本中冲突割减少了几何子问题数；简单删除更便宜，十二模型样本中则丢失了部分成功质量。这些小规模计时不支持相对通用整数规划的复杂度优势。

## E3 证据与错误批准

好市场最优第二组持仓为 `t/(3+2t)`；两模型曲率不同。B1 用 δ=0.05 的固定时点似然比阈值。下表分别按真实好／坏市场汇总，不将它们混成一个平均 regret 排名。

{md(rates)}

指标中的有意义解锁统一定义为达到好市场 oracle 持仓的一半，避免把 CVaR 的极小非零持仓和 B1 的全部批准混为一谈；原始文件同时保留任意正入场与实际持仓。B1 只取零或 oracle，两种解锁口径对其一致。

![证据与两市场 regret](results/evidence.png)

每个主单元只有 200 次重复，稀有事件不能据此认证。另在固定 t 与门槛附近运行独立 10,000 次校准批次，下面给出九项审计经 Bonferroni 调整的单侧区间上限：

{md(rare)}

最小 t 的万次模拟区间仍不够窄，因此另开全新种子、预先固定十万次重复的三项精度审计，不与旧样本混合、不修改门槛。下表区间只对这三项新审计作同时校正：

{md(precise)}

理论控制来自似然比的期望恒等式；模拟只能发现异常或衡量功效。B1 的低误批若伴随高错拒，是安全与效率的权衡。该双模型检验与经典 LR 参考相同，不能作为独立新算法胜利。

## E4 多方向安全与收益代价

pilot 样本固定基准、最多六个候选及模型危险集合；gate 数据独立。辅助密度是整个 gate 数据集的固定模型混合。比较同时包括全部数据基线与 pilot-only 基线。B2 在获批候选和基准的凸包内优化。

E4 实际使用 ε=Γ(all)，该阈值仅用于候选生成和总体达标率，不进入安全检验。输出最初把 epsilon_ratio 误标成 0.5，已仅更正为 1，并在 audit_history 保留更正前文件与哈希；观测、实际 ε、持仓和评价数值均未改变。这个宽松且处在边界的达标指标不能替代相对基准的安全—改善评价。

冻结运行期间、查看 E4 性能前增加了 CS-Safe 参照：它在相同 0.005 错误预算下保留一个共享似然置信集，只批准对集合中所有模型都安全的候选。原 LCRC 控制 regret 覆盖，不能直接当作相同安全目标的对照。该修订保存在 configs/safety_reference_amendment.json；新增比较明确标为补充，没有改变原方法或数据。

{md(four)}

确认中 B2 每次平均批准候选数为 **{b2_approvals:.4g}**。harmful 是比 pilot MA 更差的频率，mean_change 为真实风险减去 pilot 基准，负值表示改善。不能仅因 B2 少交易、少犯错就宣称它更有效；应同时观察批准数量、改善幅度及相对全数据 MA 的 regret。

MA／MLE／LCRC／CVaR 的差异主要在 10⁻¹¹ 数值量级，不能把表中微小有害频率解释成有经济意义的损失。Hinge 的零超额目标存在大片平坦区域，返回的组合虽达宽松阈值，其 regret 仍可明显高于几乎已知真模型的 MA。

pilot 诊断中位数如下。若后验质量已接近 1、ESS 接近 1，实验就缺少判断闸门信息效率的空间；本轮保留这一生成器局限，不在确认结果上调整 pilot 强度。

{md(postsummary)}

B2 观察到的危险候选误批数为 **{unsafe_count}/{len(b2)}**；把全部确认数据简单汇总所得单侧 95% 二项上限为 **{unsafe_upper:.6g}**。这是描述性汇总，不是每个真模型／信息单元的同时保证。若上限高于 0.005，则当前模拟精度不足以经验认证该水平；条件性理论保证与模拟诊断必须分开。

![多方向安全与改善](results/multidirection.png)

数据拆分只保证 gate 校准，并不免费。原始证据文件保存每个候选的密度权重、危险模型、log e-value、批准记录和最终凸组合。几何情形未覆盖及少于两个有效候选的记录均保留，不能事后挑选有利方向。

## 理论与创新边界

已核验的结论是最大兼容质量等价、有效冲突割、E1 随机化下界，以及独立 pilot 条件下多候选错误批准的 union bound 和凸包安全转移。详细证明见 THEORY_STATUS.md。

多正确答案的信息复杂度已有 [Degenne 与 Koolen](https://arxiv.org/abs/1902.03475) 的直接近邻；拆分似然比有效性沿用 [Universal Inference](https://arxiv.org/html/1912.11436v4)。本轮没有证明多模型信息复杂度的匹配上下界，也没有证明优于通用 MILP 的复杂度结果。有限模型机会目标、简单 LR 门槛本身不足以构成新颖主贡献。

最终判定如下：

1. **机制成立。** E1 的 0 与 2/3 分离、MA 均值最优性和 E3 的似然比规则得到核验；多模型兼容性不能由所有模型对兼容替代。
2. **A 有有限范围内的实证价值，B 的扩展不足。** A1 在三类确认设计中相对 MA 提升约 11.6–13.0 个百分点，相对 CVaR 提升约 16.8–19.4 个百分点，但平均 regret 增加约 29%–48%。相对简单贪心的 2.3–5.7 个百分点属于探索性证据。E4 没有产生可评价的获批改善，不能声称 B2 比 LCRC、保守基准或 CS-Safe 更高效。
3. **尚不足以认定形成可投稿的理论／算法增量。** A 值得围绕多模型冲突的信息下界、结构化算法与可扩展性继续研究；仅重述最大可行子系统不够。B1 是已有 LR 工具，B2 本轮没有新效率分离，暂不作为主贡献。

下一阶段应先证明一般非对称族中冲突结构决定的信息复杂度，并预注册更大规模求解对照；若继续 B，需要在全新开发数据上构造 pilot 后仍有决策不确定性的实例，再冻结独立确认。不能在本轮确认数据上调 pilot 强度追求获胜。

## 完整性、计算与限制

本次汇总之前 ledger 累计 CPU 为 **{cpu/3600:.4f} 小时**，累计进程经过时间为 **{wall/3600:.4f} 小时**。二者都不是并行任务的桌面总经过时间；最终数值由 delivery verification 汇总。交互式环境检查和少量前期调试未全部计入 ledger。

当前加载文件中算法异常行数：**{failures}**。结果以求解上下界为准，状态为 AlmostSolved 的解也必须经过独立核验。数值模糊行保留成功概率区间，不当作确定成功。累计预算是上限，无需刻意耗满。

最终核验曾把 ε=Γ(all) 的边界问题错误地要求点值相同，导致一次核验失败。修正后的脚本在三个非边界比例检查点值一致，在边界检查区间相容，并另用 CVXPY 复核全部六模型子集；失败记录保存在 audit_history，算法未因此调整。全部 2,422 条冲突割另行通过下界检查。

局限包括候选族已知、有限模型、独立高斯及外生 mask、固定时点审批、有限数量风险族、保守 union bound、没有真实市场外推。确认表中的区间只衡量这些固定族里的观测噪声，不衡量跨未知市场族的模型不确定性。

## 文件与复现

配置见 configs/protocol.json；原始输出为 results/E1、E2、E3、E4 对应 JSONL；数值检查见 results/gate0.json；独立确认与开发记录分开保存。逐实例权重、失败、证据和数值区间均可从固定种子再生成。运行顺序见 README.md。
'''
    (ROOT/'RESEARCH_REPORT_ZH.md').write_text(text,encoding='utf-8')
    write(OUT/'report_manifest.json',{'E1_rows':len(e1),'E2_development_rows':len(dev),'E2_report_rows':len(e2),'E3_rows':len(e3),'E4_report_rows':len(e4),'independent_confirmation':independent,'cpu_before_report':cpu})
    print('Report and figures written',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
