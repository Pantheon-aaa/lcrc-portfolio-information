"""Summarize completed evidence without promoting repaired selection to confirmation."""
import json
import numpy as np
import pandas as pd
from .experiment_v2 import ROOT, OUT, CFG, load_rows, write_json


def run():
    from .report import paired_confirm  # also installs dependency-free Markdown tables
    original=pd.DataFrame(load_rows('confirm'))
    repaired=pd.DataFrame(load_rows('repair_confirmation'))
    manifest=json.loads((CFG/'confirm_manifest.json').read_text())
    selection=json.loads((CFG/'selection.json').read_text())
    repair_config=json.loads((CFG/'selection_repair.json').read_text())
    old_names={s['name'] for s in selection['confirmation_specs']}
    assert len(original)==len(manifest)*len(old_names),'original confirmation incomplete'
    assert len(repaired)==len(manifest)*len(set(repair_config['tradeoff_candidates'])-old_names),'supplement still running'
    summaries=[]
    for label,df in [('original_frozen',original),('supplement_exploratory',repaired)]:
        if not len(df):continue
        good=df[~df.failure]
        for (kind,method),g in good.groupby(['kind','method']):
            point=g.groupby(['family_id','point_id','n','m']).regret_scaled.mean()
            summaries.append(dict(evidence=label,kind=kind,method=method,attempted=len(df[(df.kind==kind)&(df.method==method)]),
                valid=len(g),mean=g.regret_scaled.mean(),p95=g.regret_scaled.quantile(.95),worst_test_cell=point.max(),
                open_gaps=int((g.status=='gap_open').sum())))
    table=pd.DataFrame(summaries);table.to_csv(OUT/'final_risk_comparison.csv',index=False)
    import matplotlib.pyplot as plt
    shown=table[(table.evidence=='original_frozen')&(table.kind=='G2')].set_index('method')
    names=['REF-EB','A-Absolute:0.5','H-Conditional-FilterThenK','B-Max','B-Tail-P:0.25','B-Tail-P:0.5','A-Tail-P:0.5']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    for ax,metric,title in zip(axes,['mean','worst_test_cell'],['Mean regret','Worst fixed test-cell mean regret']):
        change=100*(shown.loc[names,metric]/shown.loc['A-Mean',metric]-1)
        ax.barh(names,change,color=['#287d8e' if v<0 else '#c87942' for v in change])
        ax.axvline(0,color='black',lw=.7);ax.set_title(title);ax.set_xlabel('Relative change versus MA (%)')
    axes[0].invert_yaxis();fig.suptitle('G2: original frozen confirmation; descriptive tradeoff')
    fig.savefig(OUT/'g2_mean_worst_tradeoff.png',dpi=180);fig.savefig(OUT/'g2_mean_worst_tradeoff.pdf');plt.close(fig)
    for stage in ('repair_screen','blend_development'):
        df=pd.DataFrame(load_rows(stage))
        if len(df):
            good=df[~df.failure]
            good.groupby(['kind','method']).regret_scaled.agg(['count','mean','median','max']).to_csv(OUT/(stage+'_summary.csv'))
    if len(repaired):
        base=original[original.method.isin(['A-Mean','REF-EB'])]
        paired=paired_confirm(pd.concat([base,repaired],ignore_index=True))
        paired['evidence']='exploratory reused test panel; not independent confirmation'
        paired.to_csv(OUT/'supplement_exploratory_paired.csv',index=False)
    timing=pd.DataFrame(json.loads((OUT/'final_timing_audit.json').read_text()))
    times=timing.groupby('method').agg(cold_median_seconds=('cold_total_seconds','median'),
        cold_max_seconds=('cold_total_seconds','max'),cached_median_seconds=('cached_decision_seconds','median'),
        max_rss_MiB=('rss_bytes',lambda x:x.max()/1024**2)).reset_index()
    times.to_csv(OUT/'final_timing_summary.csv',index=False)
    concentration=[]
    for stage in ('G3_d20','G3_d50'):
        rows=[r for r in load_rows(stage) if r['method']=='A-Mean' and not r['failure']]
        concentration.append(dict(stage=stage,instances=len(rows),median_ess=np.median([r['posterior']['ess'] for r in rows]),
            median_max_mass=np.median([r['posterior']['max_mass'] for r in rows])))
    write_json(OUT/'g3_concentration_summary.json',concentration)
    selector=json.loads((CFG/'selection_repair.json').read_text())
    write_json(OUT/'candidate_decisions.json',{
        'average_confirmed':[],
        'tradeoff_confirmed':[],
        'tradeoff_exploratory':selector['tradeoff_candidates'],
        'mechanism_followup':['B-Tail-P:0.5'],
        'mechanism_scope':'probability-boundary-information response supported; unique conditional performance advantage unconfirmed',
        'selection_repair_evidence':'exploratory reused test panel, not independent confirmation',
        'next_step':'prioritize boundary information theory and conditional quadrature accuracy; require new frozen families for performance claims',
        'real_market_validation':False,'public_push':False})
    pretty=table.copy()
    for col in ('mean','p95','worst_test_cell'):pretty[col]*=100
    lines=['# 研究判断：理论机制成立，算法优势尚未得到确认','',
        '**本轮最值得保留的是“信息何时足以释放持仓边界”的理论问题。现阶段不建议把论文写成条件尾部算法全面优于 Model Averaging。**', '',
        '## 1. 实验实际回答了什么','',
        '原冻结确认集有 6,480 个新实例，覆盖三个新风险族、G1 固定参数风险面和 G2 的八种信息预算。原八种方法全部跑完。相对固定先验 MA，没有入围方法在 G1、G2 的平均 regret 上取得稳定改善；按风险族配对且 Holm 校正后，没有有利的显著比较。非显著不等于方法等效，只有三个风险族也限制了推断。','',
        'G2 存在值得观察的平均—最坏测试点权衡：条件尾部 rho=0.25 的最坏测试单元均值约降低 8.3%，平均 regret 约增加 3.2%；rho=0.5 的相应变化约为降低 5.4%、增加 0.9%。这些是相对 MA 的描述性比例，不是收益率改善，也不是总体 minimax 保证。全局尾部 rho=0.5 几乎给出同样结果，因此当前不能把权衡归功于“保护外层后验边际”这一独特机制。','',
        '下表的 mean、p95、worst_test_cell 均为固定风险单位 u 的百分比。p95 是跨数据重复的尾部；worst_test_cell 是预定义风险族/参数点/信息预算上的最大均值；二者均不同于算法内部的模型间尾部目标。失败数、数值 gap 和积分诊断必须一起阅读。','',
        pretty.to_markdown(index=False,floatfmt='.5f'),'',
        '![G2 平均与最坏测试点的权衡](../results/g2_mean_worst_tradeoff.png)','',
        '## 2. 补充筛选为什么只算探索性','',
        '原筛选器将“降低最坏组损失”错误地写成“每个组都改善”，使允许付出平均代价的权衡候选和 Blend 触发条件过严。已改为两种方法各自最坏组平均验证损失之差；调参仍只读取独立开发未来收益。原候选、原确认数据和原确认结论保留。',
        f"补充开发采用原 24 个实例，加入固定 beta=0.5 的 A-Blend/B-Blend；正确准则选出的差情形候选为 {selector['tradeoff_candidates']}。补充筛查和测试复用同一冻结清单，而原确认结果已经查看，因此不能把它们重新命名为独立确认。",'',
        '补充测试的 G2 中，H-PostSampleK 相对 MA 的平均 regret 约增加 1.29%，最坏测试单元约降低 5.75%；C-LP-Max:0.01 分别为增加 12.1%、降低 7.90%。这些比例相对于 MA regret，不是相对于 u 的调参预算。C-LP-Max 在这两个 G2 汇总指标上还被 B-Tail-P:0.25 同时优于，因此尚无理由据此把主方向转向证据惩罚；它在 G1 仍呈现另一种平均—最坏点权衡。',
        '补充的逐方法统计见 repair_screen_summary.csv；相对 MA/EB 的家族配对区间及单独 Holm 列见 supplement_exploratory_paired.csv。多重比较校正不能消除已查看数据后的探索性选择问题。这两项补充候选没有完成新的独立 G3/精度确认，尚不满足最终有效算法标准。','',
        '## 3. 哪些负面结果最重要','',
        '- **条件化的增量小于目前能可靠分辨的精度。** 条件与全局尾部的平均差远小于部分节点加密敏感性。4096 节点审计中，256 节点条件尾部的最大目标超额约为 7.0e-5（rho=0.5）和 1.2e-4（rho=0.25），是原始目标单位。这个审计不是连续积分误差的严格上界。',
        '- **B-Max 对场景覆盖尤其敏感。** G2 固定外层后可以用隐藏参数盒角点精确求内层最大值；主实验的有限节点版本与此参考有明显差异。主表不可解释为精确连续条件 minimax 的性能。',
        '- **先取最坏 k 个、再归一化概率有实际数值问题。** H-RankThenP 和条件版本的四起点目标差最大约为 0.245u、0.472u，故没有扩大为确认候选；不报告虚假的全局 gap。',
        '- **G3 的近似后验过度集中是重要解释限制。** d=20 的后验 ESS 中位数约 1.99，d=50 约 1.00；对应最大质量中位数约 0.646 和 0.9996。平均、条件尾部和全局尾部因此经常给出非常相近的持仓。这既不是方法等效证明，也不是 bootstrap 质量已正确校准的证据。',
        '- **弱基线不能替代强基线。** G3 中胜过等权或本轮简单 EM 收缩的优势大部分也被 MA 获得，不能据此声称新的风险聚合器有效。','',
        '## 4. 理论推进与论文主线','',
        '完整证明见 THEORY_NOTES；通俗解释见 THEORY_EXPLAINED_ZH。有限模型、多面体约束和统一强凸下，错误模型概率相对局部持仓尺度 t 的下降速度，分别产生共同临界锥硬限制、线性软惩罚和消失项。证明包含极小点局部化、统一局部展开、下界和恢复序列，而非只在固定活跃集内形式求导。',
        '多资产两模型子类把同一协方差同时用于收益生成和决策风险：组内历史无法区分模型，共同观察才能区分。要求一边 regret=o(t²)、另一边 O(t²)，可证明 log(1/t) 阶共同信息的必要性与后验尾部规则的可达性。匹配的是阶，不是最优常数；候选族给定、只有模型身份未知。',
        '固定正概率时普通 Bayes 平均也会受共同临界锥约束；温度随 t² 缩小的熵平滑甚至可能保留多项式小概率模型的硬限制。这些结果解释了为什么“比 max 平滑”并不自动等于“及时摆脱不可信模型”。',
        '建议主问题改为：**多大的观察证据，足以消除模型歧义对某个持仓方向的限制？** 条件尾部是观察机制的工具之一。目前不支持把成熟 CVaR/Top-k 形式本身写成新贡献，也不支持最小充分统计量或一般连续族最优率的宣称。','',
        '## 5. 候选决定','',
        '| 类别 | 本轮决定 | 依据 |','|---|---|---|',
        '| 平均表现 | 无确认合格候选 | 未稳定胜过同信息 MA/EB，积分误差仍需控制 |',
        '| 差情形权衡 | H-PostSampleK、C-LP-Max:0.01 保留为补充探索名单；无最终确认候选 | 正确开发准则入围，但补充测试不是新独立确认；尚缺新 G3 和精度支持 |',
        '| 理论机制 | 保留后验尾部 rho=0.5 作为机制研究工具；条件版本仍为待验证假设 | 局部边界与信息链条可证明、可测；当前未证明条件化独有的性能收益 |','',
        '下一步先提高并验证条件后验积分精度，在观察等价而持仓边界不同的构造中检验信息阈值；若论文要主张平均或差情形优势，应另开新的冻结确认风险族。不要在已查看的测试集上继续调参，也不宜现在进入真实 A 股来寻找胜率。','',
        '## 6. CPU 可行性与可复现性','',
        '十个固定实例逐方法测量的中位数如下。冷启动包含场景构造、推断和该次决策；缓存计时复用场景 oracle；均不含解释器启动与模拟数据生成。RSS 是进程驻留内存快照，不能当成整个并行任务的严格峰值。机器为 Core Ultra 5 125H、32GB，BLAS 单线程；共享桌面时序不是独占性能基准。','',
        times.to_markdown(index=False,floatfmt='.4f'),'',
        '已完成原始 G1/G2、G3 d=20/50、局部边界与信息嵌入、连续后验和场景精度审计。全部失败与修订旧结果保留；四条开发 C-LP-Tail:1.0 的开放 gap 仍如实报告。C-Budget 的 2,232 条目标上下界已用 LP 对偶修复，持仓和真实 regret 未改变。',
        '未完成的一般理论推广、真实数据协议、运行范围和版本修订分别见英文理论笔记、REAL_DATA_PROTOCOL、ALGORITHM_CATALOG 和 IMPLEMENTATION_HISTORY。本轮未执行真实 A 股回测，未公开推送。']
    (ROOT/'docs/ASSESSMENT_ZH.md').write_text('\n'.join(lines),encoding='utf8')
    print('Assessment and supplementary descriptive summaries written.')


if __name__=='__main__':run()
