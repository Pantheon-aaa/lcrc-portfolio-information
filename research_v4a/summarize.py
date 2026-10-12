"""Prespecified family-level analysis. Uncertain successes remain intervals."""
import json,itertools,collections,time,math
import numpy as np
from .common import ROOT,OUT,rows,write,append,run,seed,decision_path
from .posterior_precision import qualify_bound
from .certificate_qualification import qualify as strict_budget

KINDS=['asymmetric','hidden_curvature','block']
METHODS=['MA','A1-v3','A1-B','Budget-Greedy','Budget-Hinge','Budget-CVaR-0.25','Budget-CVaR-0.5','Budget-CVaR-1.0']
FIELDS=['success_lower','success_upper','posterior_lower','posterior_upper','regret','regret_scaled','regret_ma','posterior_regret','posterior_ma','budget_certified','uncertain','seconds','primary_gap','secondary_gap_scaled','residual']

def finite(x,default=np.nan):
    try:v=float(x);return v if math.isfinite(v) else default
    except (ValueError,TypeError):return default
def beta_str(b):return str(float(b))
def number(r):
    ok='weights' in r;u=r['risk_unit'];L=finite(r.get('posterior_mean_regret_MA_lower'))
    return [float(r.get('true_success_lower',False)),float(r.get('true_success_upper',True)),
      finite(r.get('success_mass_lower'),0.),finite(r.get('success_mass_upper'),1.),
      finite(r.get('true_regret')),finite(r.get('true_regret'))/u,finite(r.get('true_regret_MA')),
      finite(r.get('posterior_mean_regret_method_upper')),finite(r.get('posterior_mean_regret_MA_upper')),
      float(r.get('budget_certified',False)),float(not ok or r.get('true_success_lower')!=r.get('true_success_upper')),
      finite(r.get('cold_seconds')),finite(r.get('primary_mass_gap'),1.),
      finite(r.get('secondary_price_gap'))/L if L>0 else np.nan,finite(r.get('primal_residual'))]

class Cell:
    def __init__(self):self.n=0;self.total=[0.]*len(FIELDS);self.valid=[0]*len(FIELDS);self.tail=[];self.failtail=[]
    def add(self,x):
        self.n+=1
        for i,v in enumerate(x):
            if math.isfinite(v):self.total[i]+=v;self.valid[i]+=1
        if math.isfinite(x[4]):self.tail.append(x[4])
        if x[1]==0 and math.isfinite(x[4]):self.failtail.append(x[4])
    def value(self):return np.array([v/n if n else np.nan for v,n in zip(self.total,self.valid)])

def summary(budget):
    cfg=json.loads((ROOT/'config_frozen.json').read_text());cells={};statuses=collections.Counter();counts=collections.Counter();pairs={};current=None;group={};rawtails=collections.defaultdict(list);failtails=collections.defaultdict(list)
    precision={r['dataset']:r for r in json.loads((OUT/'posterior_precision.json').read_text())['datasets']} if (OUT/'posterior_precision.json').exists() else {}
    def flush():
        if not group:return
        first=next(iter(group.values()));r=first[0]
        if not r['n_blocks'] or r['epsilon_scale']!='common':return
        for beta in (.1,.25):
            a=group.get(('A1-B',beta_str(beta)))
            for other in ('Budget-Greedy','Budget-Hinge'):
                b=group.get((other,beta_str(beta)))
                # Missing outcomes are unknown, never discarded from a paired comparison.
                al,au=(a[1][0:2] if a else (0.,1.));bl,bu=(b[1][0:2] if b else (0.,1.))
                key=(r['family'],r['kind'],beta,other,r['true_model'],r['n_blocks'],r['mask_design'],r['epsilon_ratio'])
                if key not in pairs:pairs[key]=[0.,0.,0]
                pairs[key][0]+=al-bu;pairs[key][1]+=au-bl;pairs[key][2]+=1
    for i,r in enumerate(rows(decision_path('confirmation'),fast=True)):
        if i%20000==0:budget.check()
        if r['dataset'] in precision and 'weights' in r and 'posterior_precision' not in r:
            guard=precision[r['dataset']];r=qualify_bound(r,guard['Rmax'],guard)
        r=strict_budget(r)
        key=(r['dataset'],r['epsilon_scale'],r['epsilon_ratio'],r['true_model'])
        # Zero-sample portfolios are evaluated in truth order per method; exclude from tests.
        if current!=key:flush();group={};current=key
        x=number(r);group[(r['method'],beta_str(r['beta']))]=(r,x)
        k=(r['family'],r['kind'],r['method'],beta_str(r['beta']),r['epsilon_scale'],r['epsilon_ratio'],r['n_blocks'],r['mask_design'],r['true_model'])
        if k not in cells:cells[k]=Cell()
        cells[k].add(x);statuses[(r['method'],beta_str(r['beta']),r['status'])]+=1;counts[r['method']]+=1
        if r['n_blocks'] and r['epsilon_scale']=='common':
            tk=(r['method'],beta_str(r['beta']))
            if np.isfinite(x[4]):rawtails[tk].append(x[4])
            if x[1]==0 and np.isfinite(x[4]):failtails[tk].append(x[4])
    flush();budget.check()
    # Entirely missing datasets also count as unknown; stopping the CPU budget
    # must not silently turn a balanced frozen design into a completed-case ranking.
    from .experiments import specs
    unknown=np.full(len(FIELDS),np.nan);unknown[:4]=[0.,1.,0.,1.];unknown[9:11]=[0.,1.];unknown[12]=1.
    method_betas=[('MA','0.0'),('A1-v3','inf')]+[(m,beta_str(b)) for m in METHODS if m not in ('MA','A1-v3') for b in cfg['betas']]
    padded=0
    for kind,d,M,split,index in specs('confirmation'):
        family=f'{kind}-{d}-{M}-{split}-{index}'
        for n,design,truth in itertools.product(cfg['n_blocks'],('Separated','Joint'),range(6)):
            repetitions=cfg['confirmation_repetitions'] if n else 1
            for scale,ratios in [('common',cfg['common_epsilon_scales']),('geometry',cfg['geometry_ratios'])]:
                for ratio,mb in itertools.product(ratios,method_betas):
                    method,beta=mb;k=(family,kind,method,beta,scale,ratio,n,design,truth);c=cells.setdefault(k,Cell())
                    for _ in range(max(0,repetitions-c.n)):c.add(unknown);padded+=1
            if n:
                for ratio,beta,baseline in itertools.product(cfg['common_epsilon_scales'],(.1,.25),('Budget-Greedy','Budget-Hinge')):
                    k=(family,kind,beta,baseline,truth,n,design,ratio);v=pairs.setdefault(k,[0.,0.,0]);missing=max(0,repetitions-v[2]);v[0]-=missing;v[1]+=missing;v[2]+=missing
    write(OUT/'summary_missing_qualification.json',dict(padded_unknown_decision_rows=padded,scope='Planned but unobserved decisions enter success and paired effects as intervals, not deleted observations. Unknown regrets remain unavailable.'))
    # Each truth/information/mask/threshold stratum is averaged first, then families.
    family_data=collections.defaultdict(list);curves=collections.defaultdict(list);truths=collections.defaultdict(list);outputs=[]
    for k,c in cells.items():
        f,kind,method,beta,scale,ratio,n,design,truth=k;v=c.value()
        outputs.append(dict(family=f,kind=kind,method=method,beta=beta,scale=scale,ratio=ratio,n=n,design=design,truth=truth,repetitions=c.n,**dict(zip(FIELDS,v))))
        curves[(f,kind,method,beta,scale,ratio,n,design)].append(v)
        if n and scale=='common':family_data[(f,kind,method,beta)].append(v);truths[(f,kind,method,beta,truth)].append(v)
    strata_path=OUT/'stratum_summary.jsonl';strata_path.write_text('',encoding='utf-8')
    for i,output in enumerate(outputs):
        if i%1000==0:budget.check()
        append(strata_path,output)
    fm=[]
    for k,vs in family_data.items():
        v=np.nanmean(vs,axis=0);f,kind,method,beta=k
        worst=min(np.mean([x[0] for x in t]) for tk,t in truths.items() if tk[:4]==k)
        fm.append(dict(family=f,kind=kind,method=method,beta=beta,worst_model_success=worst,**dict(zip(FIELDS,v))))
    write(OUT/'family_summary.json',fm)
    overall=[]
    for method,beta in sorted({(x['method'],x['beta']) for x in fm}):
        sub=[x for x in fm if x['method']==method and x['beta']==beta];r={'method':method,'beta':beta,'families':len(sub)}
        for field in FIELDS+['worst_model_success']:r[field]=float(np.nanmean([x[field] for x in sub]))
        tt=rawtails[(method,beta)];ft=failtails[(method,beta)];r['p95_regret']=float(np.quantile(tt,.95)) if tt else None;r['failure_conditional_regret']=float(np.mean(ft)) if ft else None
        r['actual_mean_ratio_to_MA']=r['regret']/r['regret_ma'];r['posterior_mean_ratio_to_MA']=r['posterior_regret']/r['posterior_ma'];overall.append(r)
    write(OUT/'overall_summary.json',overall)
    # Independent family is the resampling unit; never treat repeated datasets as new families.
    family_pairs=collections.defaultdict(list)
    for k,(lo,hi,n) in pairs.items():family_pairs[k[:4]].append((lo/n,hi/n))
    fp=[dict(family=k[0],kind=k[1],beta=k[2],baseline=k[3],lower=float(np.mean(np.array(v)[:,0])),upper=float(np.mean(np.array(v)[:,1]))) for k,v in family_pairs.items()]
    write(OUT/'family_paired_effects.json',fp);tests=[];rng=np.random.default_rng(seed('v4a-stratified-bootstrap-10000'));B=cfg['bootstrap_repetitions']
    for beta in (.1,.25):
        for baseline in ('Budget-Greedy','Budget-Hinge'):
            sub=[r for r in fp if r['beta']==beta and r['baseline']==baseline];boots=np.zeros((B,2));kindstats=[]
            for kind in KINDS:
                ss=sorted([r for r in sub if r['kind']==kind],key=lambda x:x['family']);arr=np.array([[r['lower'],r['upper']] for r in ss]);idx=rng.integers(0,len(ss),(B,len(ss)));boot=arr[idx].mean(1);boots+=boot/3
                kindstats.append(dict(kind=kind,families=len(ss),effect_lower=arr[:,0].mean(),effect_upper=arr[:,1].mean(),interval=[np.quantile(boot[:,0],.025),np.quantile(boot[:,1],.975)]))
            a=np.array([[r['lower'],r['upper']] for r in sub]);center=a.mean(1);radius=(a[:,1]-a[:,0])/2
            signs=np.array(list(itertools.product([-1.,1.],repeat=len(sub))));null=signs@center/len(sub)
            # Conservative sign-randomization p interval: any true paired vector in its box.
            noise=radius.mean();observed=abs(center.mean());p_upper=float(np.mean(abs(null)+noise>=max(0.,observed-noise)-1e-14));p_lower=float(np.mean(np.maximum(0.,abs(null)-noise)>=observed+noise-1e-14))
            tests.append(dict(beta=beta,baseline=baseline,families=len(sub),effect_lower=a[:,0].mean(),effect_upper=a[:,1].mean(),bootstrap_interval=[np.quantile(boots[:,0],.025),np.quantile(boots[:,1],.975)],p_lower=p_lower,p_upper=p_upper,kind_statistics=kindstats))
    order=np.argsort([r['p_upper'] for r in tests]);running=0
    for rank,i in enumerate(order):running=max(running,min(1.,(len(tests)-rank)*tests[i]['p_upper']));tests[i]['holm_p_upper']=running
    write(OUT/'primary_tests.json',dict(estimand='Equal family, truth, positive n, mask and common-threshold paired actual success difference',bootstrap=B,seed=seed('v4a-stratified-bootstrap-10000'),test='two-sided family sign randomization; symmetry assumption; Holm across four comparisons',tests=tests))
    write(OUT/'status_summary.json',[dict(method=k[0],beta=k[1],status=k[2],rows=v) for k,v in sorted(statuses.items())])
    cv=[dict(family=k[0],kind=k[1],method=k[2],beta=k[3],scale=k[4],ratio=k[5],n=k[6],design=k[7],worst_model_success_lower=min(x[0] for x in v),worst_model_success_upper=min(x[1] for x in v),**dict(zip(FIELDS,np.nanmean(v,axis=0)))) for k,v in curves.items()]
    write(OUT/'curves.json',cv);budget.check();figures(overall,fm,cv,tests);budget.check();write_report(overall,fm,tests)

def figures(overall,fm,curves,tests):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=ROOT/'figures';dest.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in ['A1-B','Budget-Greedy','Budget-Hinge','Budget-CVaR-0.25','Budget-CVaR-0.5','Budget-CVaR-1.0']:
        ss=sorted([r for r in overall if r['method']==method and r['beta']!='inf'],key=lambda r:float(r['beta']));xs=[float(r['beta']) for r in ss]
        axs[0].plot(xs,[r['success_lower'] for r in ss],'.-',label=method);axs[1].plot(xs,[r['actual_mean_ratio_to_MA'] for r in ss],'.-',label=method)
    axs[0].set(xlabel='Posterior mean-regret allowance beta',ylabel='Actual success (common thresholds)');axs[1].set(xlabel='Posterior mean-regret allowance beta',ylabel='Prior-weighted actual mean regret / MA');axs[0].legend(fontsize=8);axs[1].axhline(1,color='gray',lw=.8);fig.tight_layout();fig.savefig(dest/'S2_budget_frontier.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,kind in zip(axs,KINDS):
        for beta in (.1,.25):
            ss=[r for r in fm if r['kind']==kind and r['method']=='A1-B' and r['beta']==beta_str(beta)];d=[]
            for r in ss:
                ma=next(x for x in fm if x['family']==r['family'] and x['method']=='MA');d.append(r['success_lower']-ma['success_lower'])
            ax.plot(range(4),d,'o-',label=f'beta={beta}')
        ax.set(title=kind,xlabel='Independent confirmation family');ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4))
    axs[0].set_ylabel('Actual success gain over MA');axs[0].legend();fig.tight_layout();fig.savefig(dest/'S2_family_effects.png');plt.close(fig)
    platforms=list(rows(OUT/'S2_platforms.jsonl'));fig,axs=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,kind in zip(axs,KINDS):
        for design,linestyle in [('Separated','-'),('Joint','--')]:
            ns=sorted({r['n'] for r in curves});vals=[]
            for n in ns:
                ss=[r['success_lower'] for r in curves if r['kind']==kind and r['method']=='A1-B' and r['beta']=='0.1' and r['design']==design and r['n']==n and r['scale']=='common'];vals.append(np.mean(ss))
            line=ax.plot(ns,vals,'o'+linestyle,label=design)[0]
            pp=[r['platform_lower'] for r in platforms if r['kind']==kind and r['mask_design']==design and r['beta']==.1 and r['epsilon_scale']=='common'];ax.axhline(np.mean(pp),color=line.get_color(),ls=':',lw=1)
        ax.set(title=kind,xlabel='Observation blocks (equal scalar cost)');ax.set_xscale('symlog',linthresh=32)
    axs[0].set_ylabel('Actual success; dotted = independent platform');axs[0].legend();fig.tight_layout();fig.savefig(dest/'S2_information_platform.png');plt.close(fig)
    s1=list(rows(OUT/'S1.jsonl'))
    fig,axs=plt.subplots(2,4,figsize=(14,6),sharex=True,sharey=True)
    for i,fraction in enumerate((.25,.75)):
        for j,d in enumerate((3,6,10,20)):
            mat=np.zeros((5,7))
            for ri,ratio in enumerate((.5,.75,.875,.95,1.05)):
                for bi,beta in enumerate((0,.01,.025,.05,.1,.25,'inf')):
                    r=next(r for r in s1 if r['method']=='A1-B-symmetry' and r['d']==d and r['epsilon_scale']==ratio and r['true_model']==0 and r['family']==f'analytic-{d}-{fraction}' and r['beta']==beta);mat[ri,bi]=r['success_mass_lower']
            ax=axs[i,j];im=ax.imshow(mat,vmin=0,vmax=1,aspect='auto',origin='lower');ax.set_title(f'd={d}; t fraction={fraction}');ax.set_xticks(range(7),['0','.01','.025','.05','.10','.25','inf']);ax.set_yticks(range(5),['.50','.75','.875','.95','1.05'])
            if j==0:ax.set_ylabel('epsilon / L0')
            if i==1:ax.set_xlabel('beta')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Certified success mass',shrink=.8);fig.savefig(dest/'S1_full_grid.png',bbox_inches='tight');plt.close(fig)
    # Every reachable analytic price, not an interpolation of the seven budget queries.
    from .families import analytic,analytic_price
    prices=[];fig,axs=plt.subplots(2,4,figsize=(14,6),sharex=True,sharey=True)
    for i,fraction in enumerate((.25,.75)):
        for j,d in enumerate((3,6,10,20)):
            f,L0=analytic(d,fraction);ax=axs[i,j]
            for ratio in (.5,.75,.875,.95,1.05):
                formula=analytic_price(d,f.metadata['t'],ratio*L0)
                for a in formula:prices.append(dict(d=d,fraction=fraction,epsilon_ratio=ratio,L0=L0,**a))
                reachable=[a for a in formula if np.isfinite(a['cost'])]
                xs=sorted(set([0.,1.]+[a['beta'] for a in reachable if 0<=a['beta']<=1]))
                ys=[1. if ratio>=1 else max([0.]+[a['m']/d for a in reachable if a['beta']<=b]) for b in xs]
                ax.step(xs,ys,where='post',label=f'epsilon/L0={ratio}')
            ax.set(title=f'd={d}; t fraction={fraction}',xlim=(0,1),ylim=(0,1.02));ax.set_xscale('symlog',linthresh=.01)
            if i==1:ax.set_xlabel('beta (exact-real price jumps)')
            if j==0:ax.set_ylabel('Optimal success mass')
    axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(dest/'S1_exact_staircases.png');plt.close(fig)
    write(OUT/'S1_analytic_prices.json',dict(scope='Analytic reference only; independent numerical checks in gate0 and S1_boundaries. Empty group price zero; full group Gamma=L0 and price zero iff epsilon>=L0, else infeasible.',prices=prices))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for d in (3,6,10,20):
        ss=[r for r in s1 if r['method']=='A1-B-symmetry' and r['d']==d and r['epsilon_scale']==.875 and r['true_model']==0 and '0.25' in r['family'] and r['beta']!='inf'];ss.sort(key=lambda r:r['beta']);axs[0].step([r['beta'] for r in ss],[r['success_mass_lower'] for r in ss],where='post',label=f'd={d}')
    axs[0].set(xlabel='beta',ylabel='S1 success mass (grid samples)',title='epsilon / L0 = 0.875');axs[0].legend()
    s3=list(rows(OUT/'S3_computation.jsonl'))
    for method in ('A1-B-cold','A1-B-cached','SCIP','Enumeration'):
        xs=[];ys=[]
        for M in (6,12,24):
            vv=[finite(r.get('cpu_seconds')) for r in s3 if r['method']==method and r.get('M')==M]
            if vv:xs.append(M);ys.append(np.nanmean(vv))
        axs[1].plot(xs,ys,'o-',label=method)
    axs[1].set(xlabel='Models',ylabel='Mean process CPU seconds / query',title='S3: same objective, 30 s wall deadline');axs[1].legend();fig.tight_layout();fig.savefig(dest/'S1_S3_computation.png');plt.close(fig)

def write_report(overall,fm,tests):
    # Re-run after all experiments finish; no hand-picked method or rho.
    get=lambda method,beta:next(r for r in overall if r['method']==method and r['beta']==beta_str(beta))
    ma=get('MA',0);a10=get('A1-B',.1);a25=get('A1-B',.25);un=get('A1-B',np.inf);old=get('A1-v3',np.inf)
    ledger=list(rows(OUT/'runtime_ledger.jsonl'));cpu=sum(x['cpu_seconds'] for x in ledger);cfg=json.loads((ROOT/'config_frozen.json').read_text());completed=len(list(rows(OUT/'S2_confirmation_validated_completed.jsonl')))
    lines=['# research_v4a：平均代价受控的决策兼容性实验报告','',
      '本轮仅研究 A1-B。原始 v3 结果不参与本轮显著性检验；以下数字来自新种子的确认族。', '',
      f'主实证口径：12 个独立确认风险族，三个冻结公共阈值，两个 mask 和正样本量等权。MA 达标率 {ma["success_lower"]:.2%}；A1-B 在 10% / 25% 后验平均 regret 预算下为 {a10["success_lower"]:.2%} / {a25["success_lower"]:.2%}。区间未决事件按下界计，完整上下界保存在汇总文件。', '',
      f'新版无预算 A1-B 达标率 {un["success_lower"]:.2%}，旧 A1 为 {old["success_lower"]:.2%}；实际均值 regret 相对 MA 分别为 {un["actual_mean_ratio_to_MA"]:.4f} / {old["actual_mean_ratio_to_MA"]:.4f}。旧算法破同分与全局二级优化分列，不能把二者差异全部归因于平均预算。', '',
      '## 固定设计与完成情况','',
      f'- 起点：`{cfg["baseline_commit"]}`；独立分支 `research/decision-compatible-v4a`，只作本地提交。',
      '- 附件原文及 SHA256、冻结配置、全部族矩阵、随机种子和版本均已保存。开发族与确认族独立，不按胜负改族。',
      f'- 开发 affinity 界触发统一 n=1024；n={cfg["n_blocks"]}。公共 ε/u={cfg["common_epsilon_scales"]}；几何诊断网格单列，不混入主要检验。',
      f'- 确认已完成 {completed}/4344 个决策数据集（预期 4320 个非零 + 24 个零样本设计）；零样本组合评价全部真模型。重复阈值、方法和预算不是独立数据集。',
      f'- 本报告生成前台账累计实验 CPU {cpu/3600:.3f} 小时；最终准确总量见 audit.json 和 runtime_ledger.jsonl。墙钟与累计 CPU 分列，安装和 Git 操作不计实验 CPU。',
      '- 预确认修订了 surrogate 可行修正后的目标上界记录。随后审计发现 MILP 对小概率系数的数值尺度问题：用固定 1e6 尺度重算 A1-B 和旧 A1 的全部数据，并重跑完整 S3，所有初始记录均保留；设计、种子、观测、阈值、预算和重复数不变。正式结果仅使用 validated 记录。', '',
      '## 主性能与平均代价','',
      '| 方法 | β | 实际达标率区间 | 后验成功质量 | 实际均值 / MA | p95 regret | 确定失败条件 regret | 最坏模型达标率（族均值） | 预算认证率 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in overall:
        failure='—' if r['failure_conditional_regret'] is None else f'{r["failure_conditional_regret"]:.4g}'
        lines.append(f'| {r["method"]} | {r["beta"]} | [{r["success_lower"]:.4f}, {r["success_upper"]:.4f}] | {r["posterior_lower"]:.4f} | {r["actual_mean_ratio_to_MA"]:.4f} | {r["p95_regret"]:.4g} | {failure} | {r["worst_model_success"]:.4f} | {r["budget_certified"]:.2%} |')
    for endpoint,label in [(un,'新版 β=∞'),(old,'旧 A1（同一批新数据）')]:
        excess=endpoint['success_lower']-ma['success_lower']
        if excess>1e-8:lines+=['',f'相对 {label} 的超过 MA 部分，10% / 25% 预算保留 {(a10["success_lower"]-ma["success_lower"])/excess:.1%} / {(a25["success_lower"]-ma["success_lower"])/excess:.1%}。这是描述性比值，分母接近零时不稳定，不能替代预设配对检验。']
    lines+=['','后验预算约束是逐数据集的条件平均约束，并不保证每个真模型的样本外平均代价比例。实际均值以先验均匀、模型/观测/阈值等权汇总；p95 是该平衡设计中的经验统计。失败条件均值仅针对已确定失败的事件，不把边界未决事件当作确定失败，也不对缺失持仓插补有限 regret。成功比较始终保留未决及缺失事件的上下界；失效与未完成数另见审计。','',
      '![预算前沿](figures/S2_budget_frontier.png)','', '![逐族效应](figures/S2_family_effects.png)','',
      '## 预设四项检验','',
      '固定种子的 10,000 次分层族级 bootstrap；每类只重采样其 4 个独立确认族。双侧族级符号随机化以效应对称为检验假设；对 4 项总体检验 Holm 校正。成功边界不确定性向效应两端传播。', '',
      '| β | 对照 | 配对达标率差区间 | Bootstrap 95% 区间 | Holm p（保守上界） |', '| ---: | --- | ---: | ---: | ---: |']
    for t in tests:lines.append(f'| {t["beta"]} | {t["baseline"]} | [{t["effect_lower"]:.5f}, {t["effect_upper"]:.5f}] | [{t["bootstrap_interval"][0]:.5f}, {t["bootstrap_interval"][1]:.5f}] | {t["holm_p_upper"]:.5f} |')
    lines+=['','每类 n=4，逐族效应及区间比单独的类内显著性更有解释力；不能把 4320 个观测数据集称为 4320 个独立风险族。所有 CVaR ρ 均列出，没有事后选择最优 ρ。','',
      '## 兼容价格、平台与计算','',
      'S1 的 280 个预算查询通过解析质量核验；独立 CVXPY 价格核验与通用求解分列。置换对称约简只适用于该解析族。S1 确定性选择的最坏模型成功率与均匀随机兼容组的覆盖率不同，详见 S1 附加检查。', '',
      '![完整解析网格](figures/S1_full_grid.png)','',
      '全部可达基数的 Γ_m、β_m、代价和解析见证另存 S1_analytic_prices.json；下图画出完整解析台阶，独立数值边界证书见 S1_boundaries.json。解析曲线本身不作为算法性能成绩。','',
      '![完整价格台阶](figures/S1_exact_staircases.png)','',
      '平台通过枚举六模型集合计算，并对可行集合用独立 CVXPY 复核价格。Joint 和 Separated 每块同为 d 个标量、来自两个独立时点；空时点不提供信息。预算间隔为零时保留左右界，不宣称集合最终稳定。','',
      '![信息与平台](figures/S2_information_platform.png)','',
      'S3 的冷启动、同后验缓存、枚举及独立安装 SCIP 使用同一主成功目标和全局二级代价。公共预计算与缓存构建必须计入完整成本；图中逐查询 CPU 不包括已另存的公共族生成成本。超时的 incumbent 和开放 gap 都保留，不能将其与已证最优的速度直接排名。','',
      '![解析与计算](figures/S1_S3_computation.png)','',
      '## 数值与理论边界','',
      'QP/QCQP 原始对偶值、可行残差和近零 L* 状态逐实例保存。成功区间与主最优区间分开；主目标闭合而二级仍开放时保留 success_optimal_secondary_open。浮点舍入余量不是严格有向区间证明；高精度 KKT 检查是数值审计，不是新统计方法。', '',
      f'主实证 A1-B 的逐实例预算认证率为 {a10["budget_certified"]:.2%} / {a25["budget_certified"]:.2%}。返回持仓的实际成功表现与预算认证是两个字段；未认证实例仍保留在比较中，不能因此宣称每个实例都满足严格相对限额。高度集中的后验会使 L* 小于浮点 oracle 误差，尤其应检查 budget_certified，而不是只读主优化 status。β=0 仍按协议返回 MA，但正的浮点 QP 误差界不能认证字面上的零增量；其实际持仓成功率照常报告，严格预算与主/二级证书保留开放。所有未经预算认证的数值 incumbent 都经过 certificate_qualification.py 收紧资格，原始名义字段与持仓保留。', '',
      '有限高斯似然和均匀先验下出现浮点零后验时，统一按保存的 log density 作 80 位精度误差核查，并用保守误差界扩大数值区间；未证明预算间隔的主目标和全局二级目标保留开放。汇总应用这些资格修订，但原始名义证书不覆盖，持仓与实际成功标签不变。主质量 ≤1e-8 的容差也不等于每个极小后验权重在严格二级排序中都可以忽略；新版无预算与旧 A1 的平均代价差不能全解读为精确同主目标的破同分收益。', '',
      '理论状态表见 THEORY.md。兼容集合、Helly、最大可行子系统、似然 affinity 与 MA 的 Bayes 均值最优性均基于经典工具。[Park–Lee](https://arxiv.org/html/2606.08895v1) 已直接研究不相容集合与错误指数，本轮不以此为原创。连续持仓的价格公式和有预算的部分识别结论是候选增量，优先性尚不能由本轮实验确认。正确后验下的优化优势不能直接称为泛化优势；新风险族确认用于测量前沿大小和相对强基线差距，不提供模型失配或真实回测的证据。', '',
      '## 最终判断','']
    lines.append(f'平均代价约束下相对 MA 的总体达标率变化为 {(a10["success_lower"]-ma["success_lower"])*100:.2f} / {(a25["success_lower"]-ma["success_lower"])*100:.2f} 个百分点。是否保留跨族收益，应结合逐族结果及预设强基线比较判断，不能只看优化目标的后验提升。')
    sig=[t for t in tests if t['effect_lower']>0 and t['holm_p_upper']<.05]
    lines.append(f'四项预设比较中，{len(sig)} 项取得正效应且保守 Holm p<0.05。精确前沿机制已获数值支持；统计增量与同目标求解增量分别解释。失败、开放证书和未完成单元都保留，不能据赢家补跑。')
    lines+=['','## 同目标求解附表','', '| M | 方法 | 查询数 | 主 gap≤1e-8 | 全局二级闭合 | 平均 CPU 秒 | 中位经过秒 |', '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    s3=[strict_budget(r) for r in rows(OUT/'S3_computation.jsonl')];computational=[]
    refinements=json.loads((OUT/'S3_refined_certificates.json').read_text()) if (OUT/'S3_refined_certificates.json').exists() else []
    refmap={(r['case'],str(r['beta'])):r for r in refinements}
    for r in s3:
        if r['method']=='Enumeration' and r.get('budget_certified') and (r['case'],str(r['beta'])) in refmap:r['primary_mass_gap']=refmap[(r['case'],str(r['beta']))]['primary_mass_gap']
    for M in (6,12,24):
        for method in ('A1-B-cold','A1-B-cached','SCIP','Enumeration'):
            ss=[r for r in s3 if r.get('M')==M and r['method']==method]
            if not ss:continue
            closed=np.mean([finite(r.get('primary_mass_gap'),1)<=1e-8 for r in ss]);secondary=np.mean([r.get('primary_exact_at_float_posterior',False) and finite(r.get('primary_mass_gap'),1)<=1e-8 and finite(r.get('secondary_price_gap'),np.inf)<=1e-6*finite(r.get('posterior_mean_regret_MA_lower'),0) for r in ss]);cpu_mean=np.nanmean([finite(r.get('cpu_seconds')) for r in ss]);wall=np.nanmedian([finite(r.get('cold_seconds')) for r in ss])
            prep={r['family']:r['preparation_seconds'] for r in s3 if r.get('M')==M and 'preparation_seconds' in r};setup={r['case']:r['cache_setup_seconds'] for r in s3 if r.get('M')==M and 'cache_setup_seconds' in r}
            overhead=sum(prep.values())+(sum(setup.values()) if method=='A1-B-cached' else 0.)
            def calls(field):
                if method!='A1-B-cached':return sum(r.get(field,0) for r in ss)
                return sum(max(r.get(field,0) for r in ss if r['case']==c) for c in {r['case'] for r in ss})
            computational.append(dict(M=M,method=method,queries=len(ss),primary_closed=closed,secondary_closed=secondary,mean_cpu_seconds=cpu_mean,median_wall_seconds=wall,solver_wall_seconds=sum(finite(r.get('cold_seconds'),0.) for r in ss),public_preparation_wall_seconds=sum(prep.values()),cache_setup_wall_seconds=sum(setup.values()) if method=='A1-B-cached' else 0.,end_to_end_wall_seconds=sum(finite(r.get('cold_seconds'),0.) for r in ss)+overhead,geometry_calls=calls('geometry_calls'),price_calls=calls('price_calls'),geometry_cuts=calls('geometry_cuts'),price_cuts=calls('price_cuts'),status_counts=dict(collections.Counter(r['status'] for r in ss)),deadline_reached=sum(finite(r.get('cold_seconds'),0)>=29.9 for r in ss)));lines.append(f'| {M} | {method} | {len(ss)} | {closed:.1%} | {secondary:.1%} | {cpu_mean:.4f} | {wall:.4f} |')
    write(OUT/'computation_summary.json',computational)
    perturb=list(rows(OUT/'S3_perturbation.jsonl'));pt=[]
    lines+=['','## 后验扰动与跳点资格','', '| δ | 检查数 | 收紧预算非负 | 成功下界为正 | 预算上界认证 | 成功下界认证 |', '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for delta in (0,1e-4,1e-3,1e-2):
        rr=[r for r in perturb if r['delta']==delta and 'bound_applicable' in r]
        z=dict(delta=delta,checks=len(rr),applicable=sum(r['bound_applicable'] for r in rr),positive_success_bound=sum(r['success_bound']>0 for r in rr),budget_certified=sum(r['budget_bound_certified'] for r in rr),success_certified=sum(r['bound_applicable'] and r['success_bound_certified'] for r in rr));pt.append(z)
        lines.append(f'| {delta:g} | {len(rr)} | {z["applicable"]} | {z["positive_success_bound"]} | {z["budget_certified"]} | {z["success_certified"]} |')
    jump_cases={r['index']:r.get('success_frontier_jump_certified',False) for r in perturb if 'bound_applicable' in r}
    write(OUT/'perturbation_summary.json',dict(by_delta=pt,instances=len({r['index'] for r in perturb}),certified_success_frontier_jumps=sum(jump_cases.values()),nominal_price_locations=len(jump_cases)))
    lines+=['',f'预定 20 个实例，已记录 {len({r["index"] for r in perturb})} 个实例；是否每个位置都完成，以 missing_cells.json 为准。其中 {sum(jump_cases.values())}/{len(jump_cases)} 个名义价格位置的两侧成功平台能被区间严格分开；其余只称为价格上界附近的数值压力位置，不声称已定位一个真实成功台阶。收紧预算为负或成功下界非正时，式（26）对成功的保证是不可用或空泛的，不能将这类通过计为有力的稳定性证据。']
    lines+=['','S3 全部 600 个预定查询是否完成，以 missing_cells.json 为准。上述均值包含超时，不能据均值单独宣称速度优势；应同时看主/二级闭合率。每一族的公共生成/likelihood/oracle 与每一 ε 的缓存初始化耗时在原始行保存；相同元数据是重复引用，汇总时只加一次。冷/缓存方法的族 oracle 预计算相同，不计作缓存独有收益。']
    lines+=['','参考证书修订：原始枚举和平台曾在 β=∞ 时因 `inf <= inf` 将几何已证不兼容的集合计入候选上界，导致保守但不必要的开放 gap；S1 的零预算全成功点也未将实际持仓质量提升为主下界。已独立复核并单列收紧证书，原始记录完整保留。该参考上界修订不改变持仓；另一次 scaled-mass-v2 修复则重算了受影响方法的全部固定数据。上述枚举主 gap 使用收紧后的证书；原生枚举计时包含全局二级定价，另存全部 63 个子集的 CVXPY 建模审计成本。独立 CVXPY 点值的 44 个精度分歧经过 80 位原始/弱对偶复核，2 个获得高精度认证，42 个近零 L* 实例仍保留价格开放状态，没有放宽 ε 或预算。']
    if (ROOT/'DELIVERY_STATUS_ZH.md').exists():lines+=['',(ROOT/'DELIVERY_STATUS_ZH.md').read_text(encoding='utf-8')]
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':run('summarize',summary)
