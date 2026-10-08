"""Generate factual summaries; incomplete stages stay visibly incomplete."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp,t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment_v2 import ROOT,OUT,CFG,load_rows,write_json


def _markdown(self,index=False,floatfmt='.4f'):
    """Dependency-free compact Markdown table for the research report."""
    frame=self.reset_index() if index else self
    def cell(v):
        if isinstance(v,(float,np.floating)):return format(v,floatfmt)
        return str(v).replace('|',' / ').replace('\n',' ')
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


pd.DataFrame.to_markdown=_markdown


def paired_confirm(df):
    rows=[]
    for kind,g in df.groupby('kind'):
        pivot=g.pivot(index=['family_id','point_id','n','m','rep'],columns='method',values='regret_scaled')
        for baseline in ('A-Mean','REF-EB'):
            for method in pivot.columns:
                if method in ('A-Mean','REF-EB'):continue
                diff=(pivot[method]-pivot[baseline]).dropna()
                family_means=diff.groupby(level='family_id').mean()
                avg=float(family_means.mean()); se=float(family_means.std(ddof=1)/np.sqrt(len(family_means)))
                half=student_t.ppf(.975,len(family_means)-1)*se
                p=float(ttest_1samp(family_means,0).pvalue) if se>0 else (1. if avg==0 else 0.)
                rows.append(dict(kind=kind,method=method,baseline=baseline,difference=avg,
                    low=avg-half,high=avg+half,p=p,families=len(family_means),paired_instances=len(diff),
                    interval='t interval on family means; only three independent families; exploratory approximation'))
    if not rows:return pd.DataFrame()
    out=pd.DataFrame(rows);order=np.argsort(out.p.to_numpy());adjust=np.empty(len(out));running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.,(len(out)-rank)*out.p.iloc[idx]));adjust[idx]=running
    out['p_holm']=adjust
    return out


def run():
    from .extended_summaries import run as extended
    extended()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    expected={s:len(json.loads((CFG/f'{s}_manifest.json').read_text())) for s in ('pilot','development','G1','G2','confirm')}
    expected.update(blend_development=expected['development'],repair_screen=expected['G1']+expected['G2'],repair_confirmation=expected['confirm'])
    inventory=[]; frames=[]
    for stage in ('pilot','development','G1','G2','confirm','G3_d20','G3_d50','baseline_audit','resolution','blend_development','repair_screen','repair_confirmation'):
        raw=load_rows(stage)
        if not raw: inventory.append({'stage':stage,'rows':0,'instances':0,'state':'not run'});continue
        frame=pd.DataFrame(raw);fail=int(frame.failure.sum());instances=frame.instance_id.nunique()
        inventory.append({'stage':stage,'rows':len(frame),'instances':instances,'expected_instances':expected.get(stage),
            'failures':fail,'state':'completed manifest instances' if instances==expected.get(stage,216 if stage.startswith('G3') else instances) else 'partial'})
        good=frame[~frame.failure].copy()
        if stage=='resolution':continue
        if 'regret_scaled' in good:
            columns=[c for c in ['stage','kind','variant','d','n','m','method'] if c in good]
            summary=good.groupby(columns,dropna=False).regret_scaled.agg(['count','mean','median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).reset_index()
            summary.columns=columns+['count','mean','median','p90','p95'];summary.to_csv(OUT/f'{stage}_summary.csv',index=False)
        if stage in ('G1','G2','confirm'):frames.append(good)
        if stage=='confirm' and len(good):paired_confirm(good).to_csv(OUT/'confirmation_paired.csv',index=False)
    write_json(OUT/'inventory.json',inventory)
    report=['# 条件尾部与似然重加权：v2 研究报告','',
            '本报告只汇总本目录实际存在的记录。方法筛选是探索性研究，不是外部预注册；旧数据和数值修正均保留。',
            '本轮未进行真实 A 股回测，也未推送公开仓库。','',
            '**主要判断：原独立确认没有产生稳定胜过 Model Averaging 的算法。最有价值的推进是概率衰减—持仓边界—共同信息量之间的理论链条。**',
            '先读 [中文结论与候选决定](docs/ASSESSMENT_ZH.md)，再按本报告查阅逐项证据；[通俗理论解释](docs/THEORY_EXPLAINED_ZH.md) 与 [完整英文证明 PDF](output/pdf/THEORY_NOTES.pdf) 可分别阅读。',
            '补充筛选修正了原最坏组准则错误，加入 Blend 和两项探索候选。由于原确认集已经查看，repair_confirmation 的文件名仅指所复用的数据清单，其证据身份是探索性，不计入原确认比较。详见 IMPLEMENTATION_HISTORY 与 supplement_exploratory_paired.csv。','',
            '## 实际执行状态','',pd.DataFrame(inventory).to_markdown(index=False),'',
            '## 理论结果与边界','',
            '英文完整推导见 [THEORY_NOTES.md](docs/THEORY_NOTES.md)。主要结论在有限模型、共同基准最优解、强凸和固定多面体条件下成立；尚未经过外部审稿或全面优先性确认。','',
            '- 概率相对决策尺度的三种速度分别留下硬临界锥、线性软惩罚和消失项；给出局部化、下界、恢复序列及最优方向收敛证明。',
            '- 外层组概率也下降时，条件方法的软项通常是组内一阶惩罚的尾部，而非直接套用全局加权公式。',
            '- 固定温度熵目标具有平均型局部极限；温度为 t² 时，多项式衰减的坏模型仍可能保留硬边界。仅观察温度/t² 不足以判定行为。',
            '- 完整 mask 高斯 affinity 给出带先验赔率的概率上界；Theorem 6 将检验下界嵌入同一收益/损失协方差下的多资产块模型，证明 log(1/t) 量级的共同观察必要性与后验 CVaR 可达性。候选族给定，仍不包括一般连续风险族或最优常数。',
            '- 固定分组下，条件尾部对联合概率质量的总变差扰动有稳定性界，不需要人为给极小外层组加正概率 floor。','',
            '## 数值修正与信息权限','',
            '固定网格在 G2 中出现低 ESS 与跨积分节点不稳定，触发计划内的重要性积分。使用数据拟合的提议分布，并显式乘先验/提议比；没有再次重复乘似然。G1 使用 128 场景，G2 使用 256 场景。',
            '更换积分后重跑独立开发数据并重新冻结候选；测试清单和数据种子未改变。归档结果不能与当前记录混合汇总。',
            '另外修正了过滤后奇数场景数的整数 Top-k 取整，重新运行开发筛选并重算受影响方法；两项平均候选名称未改变。大场景改用直接 Clarabel 路径后仍以相同 QP 上下界验收，失败则回退；计时需按源代码版本解释。',
            'C-Budget 的内层支持函数改用经修复的 LP 对偶上界；2,232 条历史目标界已经逐条复核，持仓和真实 regret 保持不变。修正后全部满足目标 gap，原界另行归档。',
            'QP 数值 gap、后验积分误差和频率学统计误差分别报告。tail objective 不被解释为 95% 置信证书。','']
    if (CFG/'selection.json').exists():
        sel=json.loads((CFG/'selection.json').read_text())
        report+=['## 原冻结开发筛选','',f"平均表现初选：{sel['mean_candidates']}；原准则下差情形初选：{sel['tradeoff_candidates']}；机制假设：{sel['mechanism_hypotheses']}。",
            '这些名单只使用开发集独立未来实现损失。它们不是已经通过确认的赢家；固定边际的理论机制候选可以保留而不宣称平均表现领先。','']
    for stage in ('G1','G2','confirm'):
        raw=load_rows(stage)
        if not raw:continue
        df=pd.DataFrame(raw);df=df[~df.failure]
        if not len(df):continue
        keys=['kind','method'] if stage=='confirm' else ['method']
        table=df.groupby(keys).agg(instances=('regret_scaled','size'),mean_regret=('regret_scaled','mean'),p95=('regret_scaled',lambda x:x.quantile(.95)),
            median_seconds=('seconds','median'),open_gaps=('status',lambda x:int((x=='gap_open').sum()))).reset_index()
        table.mean_regret*=100;table.p95*=100
        report += [f'## {stage} 实测结果','', 'regret 以固定开发风险单位的百分比表示，不是投资收益率。不同准则的权衡需分开解释。','',table.to_markdown(index=False,floatfmt='.4f'),'']
        if stage=='G2':
            fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
            for ax,n in zip(axes,(64,256)):
                for name in ['A-Mean','REF-EB','A-Tail-P:0.5','B-Tail-P:0.5','B-Max']:
                    g=df[(df.n==n)&(df.method==name)].groupby('m').regret_scaled.mean()
                    if len(g):ax.plot(g.index,g.values*100,'o-',label=name,ms=3)
                ax.set_title(f'G2: {n} blocks');ax.set_xlabel('Joint blocks');ax.set_ylabel('Regret (% of fixed risk unit)')
            axes[-1].legend(fontsize=7);fig.savefig(OUT/'g2_information.png',dpi=180);fig.savefig(OUT/'g2_information.pdf');plt.close(fig)
            report+=['![共同信息与组合损失](results/g2_information.png)','']
    if (OUT/'confirmation_paired.csv').exists():
        paired=pd.read_csv(OUT/'confirmation_paired.csv');wins=paired[(paired.high<0)&(paired.p_holm<.05)]
        report+=['## 独立确认的配对比较','',
            '区间按风险族均值计算，只有三个独立风险族，跨风险族推断仍然有限。Holm 校正只涉及冻结候选与 MA/EB 两项主要基线；不能从非显著结果推断等效。','',
            paired.to_markdown(index=False,floatfmt='.5f'),'']
        if len(wins):report+=['满足该近似区间与多重比较标准的条目：','',wins[['kind','method','baseline','difference']].to_markdown(index=False),'']
        else:report+=['当前没有条目同时满足负向配对区间和 Holm 显著性标准；不能将均值排名写成稳定领先。','']
    if (OUT/'resolution.jsonl').exists():
        df=pd.DataFrame(load_rows('resolution'));df=df[~df.failure]
        summary=df.groupby(['method','size']).agg(mean_objective=('reference_objective','mean'),mean_ess=('posterior',lambda x:np.mean([z['ess'] for z in x]))).reset_index()
        summary.to_csv(OUT/'resolution_summary.csv',index=False)
        spread=df.groupby(['instance_id','size','method']).reference_objective.agg(lambda x:x.max()-x.min()).groupby(['size','method']).max().reset_index(name='max_scramble_objective_spread')
        spread.to_csv(OUT/'resolution_spread.csv',index=False)
        report+=['## 场景精度审计','', '下表为同一 1024 场景参考目标上的跨 scramble 最大差异。结果差异不超过数值/积分敏感性的方法不被解释为可靠优势。',
                 '',spread.to_markdown(index=False,floatfmt='.7f'),'']
    for stage in ('G1','confirm'):
        path=OUT/f'{stage}_four_prior_grid.csv'
        if path.exists():
            grid=pd.read_csv(path);grid=grid[~grid.method.str.startswith('OracleGridPrior')]
            table=grid.groupby(['distribution','method']).weighted_risk.mean().unstack(0)*100
            report += [f'## {stage}：同一固定真值网格的四种权重','',
                '每个风险族的八个固定真值分别按均匀、中心、边界双峰和偏斜权重汇总，再对三个风险族平均。这是有限网格风险，不能解释为连续生成先验的积分。',
                '',table.reset_index().to_markdown(index=False,floatfmt='.4f'),'']
    for stage in ('G3_d20','G3_d50'):
        path=OUT/f'{stage}_path_paired.csv'
        if path.exists():
            paired=pd.read_csv(path);summary=paired.groupby(['method','metric']).difference.mean().unstack(-1)
            report += [f'## {stage}：未知结构和时间变化','',
                '下表是六种环境等权汇总后，相对 A-Mean 的配对损失差；负数较好。regret_scaled 使用固定风险单位，realized_risk 是未来 21 日实现的半二阶矩。详细的环境分组和整条路径区间见对应 path_paired.csv。',
                '每种环境仅三条独立路径；12 个滚动窗口不能算作 12 次独立重复。这部分属于压力检验，不给出正式市场优势结论。',
                '',summary.reset_index().to_markdown(index=False,floatfmt='.6f'),'']
    if (OUT/'posterior_audit.json').exists():
        audits=json.loads((OUT/'posterior_audit.json').read_text());a=[]
        for r in audits:
            c=[x for x in r['comparison'] if x['size']==256]
            a.append({'family':r['family_id'],'max_Rhat':max(r['rhat']),'min_ESS':min(r['ess']),
                'max_MA_weight_distance':max(x['weight_distance'] for x in c),
                'max_MA_reference_excess':max(x['mc_reference_excess'] for x in c)})
        report += ['## 连续后验独立核验','',
            '另在三个风险族的高共同信息、边界真值实例运行两条连续参数 Metropolis 链。该抽样使用完整似然与均匀连续先验，不读取真参数；与 256/512/1024 节点积分比较。以下为 256 节点相对 MCMC 均值决策的诊断。',
            'Rhat/ESS 只是有限链诊断，参考本身有 Monte Carlo 误差；此处没有证明条件尾部积分的全局精度。','',pd.DataFrame(a).to_markdown(index=False,floatfmt='.8f'),'']
    if (OUT/'high_resolution_audit.json').exists():
        raw=json.loads((OUT/'high_resolution_audit.json').read_text());lines=[]
        for row in raw:
            for c in row['comparison']:lines.append({'method':row['method'],'size':c['size'],'excess':c['high_reference_excess'],'distance':c['weight_distance']})
        frame=pd.DataFrame(lines)
        table=frame.groupby(['method','size']).agg(max_excess=('excess','max'),max_weight_distance=('distance','max')).reset_index()
        table.to_csv(OUT/'high_resolution_summary.csv',index=False)
        report += ['## 加密后的参考与 B-Max 的特殊问题','',
            '初始审计发现部分尾部方法对节点较敏感，因而增加 4096 节点参考（32 个外层、128 个内层）。这些审计未用于重新选择确认候选。',
            'G2 在固定外层参数后对隐藏参数仿射，regret 对隐藏参数凸，因此条件最大值可在隐藏参数盒子的角点上精确取得。B-Max 的本表参考使用全部角点与加密后的外层边际；主实验中的 B-Max 是有限场景版本，不能冒称精确连续 minimax。',
            '其余方法仍是数值积分参考，不是连续最优性证明。下表最大目标差使用原始半风险单位；应除以固定 u 后与性能差比较。','',table.to_markdown(index=False,floatfmt='.7f'),'']
    if (OUT/'embedded_boundary.jsonl').exists():
        df=pd.DataFrame(load_rows('embedded_boundary'))
        table=df.groupby(['truth','log_multiplier','rho']).regret_over_t2.mean().reset_index()
        table.to_csv(OUT/'embedded_boundary_summary.csv',index=False)
        report += ['## 信息下界的协方差嵌入核验','',f'保存 {len(df)} 条两模型、多资产可实现的充分投影实验记录。协方差正定性、仅组内观测的似然一致、完整 QP 与解析持仓一致、完整维度与投影实验的 KL 一致均已检查。',
            '共同观察数设为固定倍数乘 log(1/t)；这里只验证定理子类的机制，不混入 G1/G2 算法排名，也不声称有限尺度模拟证明渐近最优常数。','']
    if (OUT/'boundary.jsonl').exists():
        b=pd.DataFrame(load_rows('boundary'));report+=['## 多资产边界机制','',f'已保存 {len(b)} 条解析面上优化记录。粗尺度另与完整维度 QP/尾部问题交叉核对；细尺度用显式面上公式避免 oracle 目标相减的精度平台。','']
        g=b[(b.d==6)&(b.seed==0)&(b.mechanism=='boundary')]
        fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
        for probability in ['fixed','sqrt','linear','quadratic']:
            x=g[(g.probability==probability)&(g.method=='Tail:.5')].sort_values('t')
            axes[0].plot(x.t,x.x_over_t,'o-',label=probability,ms=3)
        for name in ['Mean','Tail:.5','Soft-fixed','Soft-t2','Soft-t1.5']:
            x=g[(g.probability=='quadratic')&(g.method==name)].sort_values('t')
            axes[1].plot(x.t,x.x_over_t,'o-',label=name,ms=3)
        for ax in axes:ax.set_xscale('log');ax.set_xlabel('t');ax.set_ylabel('Position response x/t');ax.legend(fontsize=7)
        axes[0].set_title('Probability decay and CVaR response');axes[1].set_title('Quadratically rare adverse model')
        fig.savefig(OUT/'boundary_response.png',dpi=180);fig.savefig(OUT/'boundary_response.pdf');plt.close(fig)
        report+=['![概率与边界响应](results/boundary_response.png)','']
    ledger=load_rows('runtime_ledger')
    report+=['## 资源与解释限制','',f"已记录实验进程累计墙钟时间 {sum(x['wall_seconds'] for x in ledger)/3600:.3f} 小时；各项详见 runtime_ledger.jsonl。短暂的交互核验与首次异常终止没有全部纳入该计时，所以它不是整个研究过程的精确总耗时。",
        '部分独立结构审计与性能实验在不同进程并行；均限制单线程 BLAS，时间来自共享桌面，不属于独占机器基准。',
        'G3 使用近似 bootstrap 参考且包含模型失配，所有统计性能结论均为经验结果。开发集 EB 先验未跨越模型空间强行用于 G3。',
        '一般连续后验的统计最优性、全资产信息下界常数与外部市场优势仍未证明。真实数据后续协议见 docs/REAL_DATA_PROTOCOL.md。','']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(report),encoding='utf-8')
    # Source/result manifest is generated last; archive files are preserved, not pooled.
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    write_json(ROOT/'manifest.json',files)
    print('Report written; inventory explicitly distinguishes completed/partial stages.')


if __name__=='__main__':run()
