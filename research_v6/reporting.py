import json,gzip,time,tarfile,hashlib,platform,io
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import zstandard as zstd
from .common import *
from .pipeline import readtraj
def binomial(k,n):
    return [0 if not k else float(beta_dist.ppf(.025,k,n-k+1)),1 if k==n else float(beta_dist.ppf(.975,k+1,n-k))] if n else [None,None]
def aggregate():
    records=[];counts={};failures=[];allocation_tables={};geometries={}
    for split in ('development','confirmation','calibration'):
        count=0
        for path in (RUN/split).glob('*.json.gz'):
            r=readtraj(path);stop=r.get('stop');T=r['T_candidate'];T=float(T) if isinstance(T,(str,int,float)) else np.nan
            rec={k:r[k] for k in ('family','kind','truth','rep','ratio','delta','premium','policy','status','certified','false_certification','certification_truth_indeterminate','stop_t','stop_cost','restricted_cost','censor_cost','cpu_seconds')}
            # Every trajectory in these three stages has the frozen 100,000-action cap.
            d=int(round(r['censor_cost']/100000))
            actions=int(sum(r['counts']));times=int(sum(n*k for n,k in zip(r['counts'],(d,2,(d+1)//2,1))))
            rec.update(split=split,design='Separated' if r['allowed']==[1] else 'All',actions=actions,scalars=d*actions,independent_times=times,joint_fraction=r['counts'][3]/max(1,actions),T_candidate=T,normalized_stop_cost=(r['stop_cost']/(T*np.log(1/r['delta']))) if stop and np.isfinite(T) and T>0 else None)
            C=r['confidence_mask'];rec.update(confidence_mask=C,confidence_size=int(C).bit_count())
            if split!='development':
                if r['family'] not in geometries:geometries[r['family']]=read(RUN/'geometry'/f"{r['family']}.json")
                g=geometries[r['family']].get(str(C),{})
                rec.update(gamma_lower=g.get('lower'),gamma_upper=g.get('upper'))
            if stop:rec.update(actual_regret_lower=stop['certificate']['regret_lower'][r['truth']],actual_regret_upper=stop['certificate']['regret_upper'][r['truth']])
            key=f"{r['family']}-{r['ratio']}-{r['premium']}-"+''.join(map(str,r['allowed']))
            path_table=RUN/'tables_klv2'/f'{key}.npz'
            if split=='confirmation' and actions and path_table.exists():
                if key not in allocation_tables:
                    with np.load(path_table) as z:allocation_tables[key]=(z['rates'],z['full'])
                rates,full=allocation_tables[key];target=full[r['truth'],np.argmax(rates[r['truth']])]
                rec['allocation_distance_oracle_L1']=float(np.abs(np.array(r['counts'])/actions-target).sum())
            else:rec['allocation_distance_oracle_L1']=None
            records.append(rec);count+=1
            if r['false_certification'] or r['certification_truth_indeterminate'] or r['status']=='empty_confidence_set':failures.append(dict(path=str(path.relative_to(RUN)),**rec))
        counts[split]=count
    frame=pd.DataFrame(records);OUT.mkdir(parents=True,exist_ok=True);frame.to_json(OUT/'trajectory_aggregate.json.gz',orient='records',compression='gzip',force_ascii=False)
    summary=[]
    keys=['split','kind','ratio','delta','premium','design','policy']
    for key,g in frame.groupby(keys,dropna=False):
        k=int(g.false_certification.sum());n=len(g);st=g.loc[g.certified,'stop_cost']
        summary.append(dict(zip(keys,key),n=n,certification_rate=float(g.certified.mean()),false_certification=k,error_interval=binomial(k,n),numerical_indeterminate=int(g.certification_truth_indeterminate.sum()),restricted_mean_cost=float(g.restricted_cost.mean()),stop_median_conditional=float(st.median()) if len(st) else None,stop_p95_conditional=float(st.quantile(.95)) if len(st) else None,joint_fraction=float(g.joint_fraction.mean()),mean_scalars=float(g.scalars.mean()),mean_independent_times=float(g.independent_times.mean()),allocation_distance_oracle_L1=float(g.allocation_distance_oracle_L1.mean()) if g.allocation_distance_oracle_L1.notna().any() else None,normalized_stop_mean_conditional=float(g.normalized_stop_cost.mean()) if g.normalized_stop_cost.notna().any() else None))
    write(OUT/'sequential_summary.json',summary)
    calibration=[]
    for (family,truth),g in frame[frame.split=='calibration'].groupby(['family','truth']):
        calibration.append(dict(family=family,truth=int(truth),n=len(g),errors=int(g.false_certification.sum()),interval=binomial(int(g.false_certification.sum()),len(g)),certified=int(g.certified.sum())))
    write(OUT/'calibration_errors.json',calibration)
    cfg=read(ROOT/'config_frozen.json');planned=len(read(ROOT/'manifest.json'))
    write(OUT/'completion.json',dict(planned_confirmation=planned,processed_confirmation=counts.get('confirmation',0),unrun_confirmation=planned-counts.get('confirmation',0),counts=counts,terminal=read(OUT/'terminal_completion.json') if (OUT/'terminal_completion.json').exists() else None))
    write(OUT/'exceptions.json',dict(trajectory_exceptions=failures,job_failures=[read(p) for p in (RUN/'failures').glob('*.json')]))
    return frame,summary
def tests(df):
    d=df[(df.split=='confirmation')&(df.design=='All')&(df.delta==.05)&(df.ratio==.5)]
    out=[];rng=np.random.default_rng(20261013)
    for cost in (1,4):
      for comparator in ('equal_cost','joint_only'):
        key=['family','kind','truth','rep'];a=d[(d.premium==cost)&(d.policy=='conflict_kl')];b=d[(d.premium==cost)&(d.policy==comparator)]
        pair=a.merge(b,on=key,suffixes=('_new','_base'));family=[]
        for (f,kind),g in pair.groupby(['family','kind']):
            # Equal-weight truth means, then family means.
            delta=(g.restricted_cost_base-g.restricted_cost_new).groupby(g.truth).mean()
            family.append(dict(family=f,kind=kind,cost_reduction=float(delta.mean()),pairs=len(g),truths=len(delta),certification_difference=float((g.certified_new.astype(float)-g.certified_base.astype(float)).mean())))
        a=pd.DataFrame(family);boot=[]
        if len(a):
            for _ in range(10000):boot.append(np.mean([rng.choice(g.cost_reduction.to_numpy(),len(g),replace=True).mean() for _,g in a.groupby('kind')]))
            values=a.cost_reduction.to_numpy();observed=float(values.mean());signs=rng.choice([-1,1],size=(10000,len(values)));p=(1+int(((signs*values).mean(1)>=observed-1e-12).sum()))/10001
            result=dict(effect=observed,interval=np.quantile(boot,[.025,.975]),p=p)
        else:result=dict(effect=None,interval=None,p=None)
        out.append(dict(premium=cost,comparator=comparator,families=family,paired_records=len(pair),**result))
    valid=sorted([i for i,r in enumerate(out) if r['p'] is not None],key=lambda i:out[i]['p']);prev=0
    for rank,i in enumerate(valid):prev=max(prev,min(1,out[i]['p']*(4-rank)));out[i]['holm_p']=prev
    for r in out:r['qualification']='family-level paired available records; incomplete frozen coverage remains explicit' if read(OUT/'completion.json')['unrun_confirmation'] else 'frozen full confirmation'
    write(OUT/'four_tests.json',out);return out
def terminal_summary():
    result=[];errors=[];stream_costs={}
    for path in (RUN/'terminal').glob('*.json.gz'):
        r=readtraj(path);j=r['job'];p=np.array(r['posterior']);
        ds=(j['family'],j['truth'],j['rep'],j['premium'],tuple(j['allowed']))
        if j['n']:
            if ds not in stream_costs:
                stream=readtraj(RUN/'terminal_streams'/f'{seed(*ds)}.json.gz')
                stream_costs[ds]={int(k):v['cost'] for k,v in stream['snapshots'].items()}
            observation_cost=stream_costs[ds][j['n']]
        else:observation_cost=0
        for a in r['answers']:
            if 'error' in a:errors.append(dict(job=j,**a));continue
            truths=r['zero_prior_truths'] if j['n']==0 else [j['truth']]
            for truth in truths:
                lo=a['regret_lower'][truth];hi=a['regret_upper'][truth]
                certified=a['beta']=='inf' or a['mean_increment_upper']<=a['budget_lower']
                result.append(dict(**{k:v for k,v in j.items() if k not in ('allowed','truth')},truth=truth,observation_cost=observation_cost,design='Separated' if j['allowed']==[1] else 'All',method=a['method'],beta=str(a['beta']),success_low=hi<=r['epsilon'],success_high=lo<=r['epsilon'],regret=(lo+hi)/2,posterior_mass=a['success_mass_lower'],posterior_mass_upper=a['success_mass_upper'],optimal_mass_lower=a['optimal_mass_lower'],optimal_mass_upper=a['optimal_mass_upper'],budget_certified=certified,primary_gap=a['primary_mass_gap'],status=a['status'],cpu=a['cpu']))
    df=pd.DataFrame(result);summary=[]
    if len(df):
      for key,g in df.groupby(['n','ratio','design','method','beta']):
        summary.append(dict(zip(['n','ratio','design','method','beta'],key),n_records=len(g),success_low=float(g.success_low.mean()),success_high=float(g.success_high.mean()),mean_regret=float(g.regret.mean()),p95=float(g.regret.quantile(.95)),failure_regret=float(g.loc[~g.success_high,'regret'].mean()) if (~g.success_high).any() else None,budget_certified=float(g.budget_certified.mean()),mass=float(g.posterior_mass.mean()),mean_primary_gap=float(g.primary_gap.mean())))
    write(OUT/'terminal_summary.json',summary);write(OUT/'terminal_errors.json',errors)
    if len(df):df.to_json(OUT/'terminal_aggregate.json.gz',orient='records',compression='gzip')
    paired=[];worst=[]
    if len(df):
      keys=['family','rep','truth','premium','n','ratio','design','beta']
      aa=df[df.method=='A1-B']
      for method in ('Greedy','Hinge','CVaR'):
       pp=aa.merge(df[df.method==method],on=keys,suffixes=('_a','_b'))
       for key,g in pp.groupby(['n','ratio','design','beta']):
        paired.append(dict(zip(['n','ratio','design','beta'],key),comparator=method,pairs=len(g),success_difference_lower=float((g.success_low_a.astype(float)-g.success_high_b.astype(float)).mean()),success_difference_upper=float((g.success_high_a.astype(float)-g.success_low_b.astype(float)).mean()),mean_regret_difference=float((g.regret_a-g.regret_b).mean()),attained_mass_difference_lower=float((g.posterior_mass_a-g.posterior_mass_upper_b).mean()),attained_mass_difference_upper=float((g.posterior_mass_upper_a-g.posterior_mass_b).mean()),optimality_gap_lower=float(np.maximum(0,g.optimal_mass_lower_a-g.posterior_mass_upper_b).mean()),optimality_gap_upper=float(np.maximum(0,g.optimal_mass_upper_a-g.posterior_mass_b).mean()),both_budget_certified=float((g.budget_certified_a&g.budget_certified_b).mean())))
      for key,g in df.groupby(['family','n','ratio','design','method','beta']):
       rates=g.groupby('truth')[['success_low','success_high']].mean()
       worst.append(dict(zip(['family','n','ratio','design','method','beta'],key),truths_evaluated=len(rates),worst_failure_lower=float(1-rates.success_high.min()),worst_failure_upper=float(1-rates.success_low.min()),qualification='complete truth coverage' if len(rates)==6 else 'partial truth coverage; not full worst-model estimate'))
    write(OUT/'terminal_paired.json',paired);write(OUT/'terminal_worst_model.json',worst)
    return summary

def geometry_summary():
    from .pipeline import load_family
    from .models import specs,equivalence
    from .information import randomized_platform
    out=[]
    for sp in specs('confirmation'):
        name=f'{sp[0]}-{sp[1]}-6-confirmation-{sp[4]}';f=load_family(name)
        geo={int(k):v for k,v in read(RUN/'geometry'/f'{name}.json').items()}
        gamma=(geo[63]['lower']+geo[63]['upper'])/2
        for ratio in (.5,.75,1.25):
          eps=ratio*gamma;ok=[True]+[geo[s]['upper']<=eps for s in range(1,64)]
          # Bridge/WithoutJoint are structural diagnostics only, not added trajectory comparisons.
          for design,allowed in [('Singleton',[0]),('Separated',[1]),('Bridge',[2]),('WithoutJoint',[0,1,2]),('All',[0,1,2,3])]:
            for group in equivalence(f,allowed):
              g=geo[sum(1<<j for j in group)]
              out.append(dict(family=name,kind=sp[0],ratio=ratio,epsilon=eps,design=design,group=group,gamma_all_lower=geo[63]['lower'],gamma_all_upper=geo[63]['upper'],gamma_class_lower=g['lower'],gamma_class_upper=g['upper'],uniformly_coverable=g['upper']<=eps,uniformly_impossible=g['lower']>eps,randomized_platform=randomized_platform(6,ok,group)))
    write(OUT/'confirmation_geometry.json',out)

def inventory():
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for package in ('numpy','scipy','clarabel','cvxpy','highspy','mpmath','numba','psutil','zstandard','matplotlib','pandas'):
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    write(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=versions,blas_threads=1,maximum_numerical_processes=2,base=BASE))
    done={p.name[:-8] for p in (RUN/'confirmation').glob('*.json.gz')}
    missing=[]
    for j in read(ROOT/'manifest.json'):
        key=j['family']+f"_t{j['truth']}_r{j['rep']}_e{j['ratio']}_d{j['delta']}_c{j['premium']}_a"+''.join(map(str,j['allowed']))+'_'+j['policy']
        if key not in done:missing.append(j)
    with gzip.open(OUT/'unrun_confirmation.json.gz','wt',encoding='utf8') as z:json.dump(missing,z)
    done={p.name[:-8] for p in (RUN/'terminal').glob('*.json.gz')};missing=[]
    for j in read(ROOT/'terminal_manifest.json'):
        key=f"{j['family']}_r{j['rep']}_t{j['truth']}_c{j['premium']}_a{''.join(map(str,j['allowed']))}_n{j['n']}_e{j['ratio']}"
        if key not in done:missing.append(j)
    with gzip.open(OUT/'unrun_terminal.json.gz','wt',encoding='utf8') as z:json.dump(missing,z)
def plots(df):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=df[(df.split=='confirmation')&(df.design=='All')&(df.delta==.05)&(df.ratio==.5)]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,c in zip(axes,(1,4)):
        g=d[d.premium==c].groupby('policy').restricted_cost.mean();ax.barh(g.index,g.values);ax.set_title(f'Joint cost multiplier {c}');ax.set_xlabel('Restricted mean cost (observed scalar cost units)')
    fig.tight_layout();fig.savefig(OUT/'allocation_cost.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,c in zip(axes,(1,4)):
        for policy,g in d[d.premium==c].groupby('policy'):
            v=np.sort(g.loc[g.certified,'stop_cost'].to_numpy());ax.step(v,np.arange(1,len(v)+1)/len(g),where='post',label=policy)
        ax.set_title(f'Joint multiplier {c}');ax.set_xlabel('Certification cost');ax.set_ylabel('Fraction of ALL trajectories certified');ax.set_xscale('symlog',linthresh=1);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'certification_cdf.png',dpi=160);plt.close(fig)
    h=read(OUT/'S0_hidden_ablation.json');fig,ax=plt.subplots(figsize=(7,4))
    for name in sorted({r['family'] for r in h}):
        z=[r for r in h if r['family']==name];ax.plot([r['strength'] for r in z],[r['gamma']['upper'] for r in z],'o-',label=name)
    ax.set_xlabel('Common diagonal perturbation');ax.set_ylabel('Certified upper bound on Gamma');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'hidden_curvature.png',dpi=160);plt.close(fig)
    ge=read(OUT/'confirmation_geometry.json');names=list(dict.fromkeys(x['family'] for x in ge))
    designs=['Singleton','Separated','Bridge','WithoutJoint','All']
    vals=np.array([[max(x['gamma_class_upper']/x['gamma_all_upper'] for x in ge if x['family']==name and x['design']==des and x['ratio']==.5) for des in designs] for name in names])
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(vals,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(designs)),designs,fontsize=8);ax.set_yticks(range(len(names)),names,fontsize=7)
    for i in range(len(names)):
        for j in range(len(designs)):ax.text(j,i,f'{vals[i,j]:.3g}',ha='center',va='center',color='white' if vals[i,j]<.5 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='Worst equivalence-class Gamma / whole-family Gamma');fig.tight_layout();fig.savefig(OUT/'equivalence_geometry.png',dpi=160);plt.close(fig)
def report():
    df,summary=aggregate();tt=tests(df);term=terminal_summary();geometry_summary();inventory();plots(df);comp=read(OUT/'completion.json');c=df[df.split=='confirmation'];main=c[(c.design=='All')&(c.delta==.05)&(c.ratio==.5)]
    text=['# research_v6：决策充分信息与自适应观测合成报告','',
      '本轮仅纯合成。新族/新种子与历史复现分栏；没有真实行情或历史校准半合成。理论和统计保证的条件见[理论状态](THEORY_STATUS_ZH.md)，实现调整见[开发日志](DEVELOPMENT_LOG_ZH.md)。','',
      '## 完成与研究边界','',f"确认计划{comp['planned_confirmation']:,}条轨迹，已处理{comp['processed_confirmation']:,}，未运行{comp['unrun_confirmation']:,}。预算耗尽未运行与轨迹内未认证分别报告；未按胜负筛选。",'',
      '可用动作成本为观测d个标量，Joint溢价另计。所有方法共用固定先验整路径混合置信序列。oracle_allocation使用真模型的最优LP分配并保留强制探索，只作参照，不是理论下界。Separate-only只有一个合法动作，因此五种分配法是同一过程，仅运行一个代表。','',
      '## 主参数的采样成本与认证','',
      '|Joint成本倍数|方法|轨迹数|认证率|错误认证|截断平均成本|','|---|---|---:|---:|---:|---:|']
    for (cost,policy),g in main.groupby(['premium','policy']):text.append(f'|{cost}|{policy}|{len(g)}|{g.certified.mean():.3%}|{int(g.false_certification.sum())}|{g.restricted_cost.mean():.3f}|')
    text+=['','未认证成本按共同上限100,000d计入截断平均，不取成功停止子样本均值。标量、成本、独立时点及CPU不是同一量。全部δ/ε/设计见`results/sequential_summary.json`。','',
      '![采样成本](results/allocation_cost.png)','![停止分布](results/certification_cdf.png)','',
      '## 四项主要比较','', '|Joint倍数|冲突方法相对基线|成本减少|族级95%区间|Holm p|配对记录|','|---|---|---:|---|---:|---:|']
    for r in tt:text.append(f"|{r['premium']}|{r['comparator']}|{r['effect']}|{r['interval']}|{r.get('holm_p')}|{r['paired_records']}|")
    text+=['','正的成本减少表示冲突方法成本较低，但必须结合认证率与错误核验。每类仅四个独立确认族；重复采样不能冒充新风险族。若清单未完成，这些是公共已完成配对的探索性统计，不称完整冻结确认。','',
      '## 理论诊断','',
      '旧隐藏曲率族包含非均匀对角扰动，H1≈0并不推出最终模型具有共同最优组合。消融的Γ、最优持仓差异及一阶响应误差在`S0_hidden_ablation.json`；移除扰动后数值近零与精确共同最优控制分开。','',
      '![隐藏曲率消融](results/hidden_curvature.png)','',
      '![等价类几何](results/equivalence_geometry.png)','',
      '热图为最坏观测等价类Γ与全族Γ之比；共同组合的可认证性由阈值与类内Γ比较，不能仅由模型是否可辨识判断。新确认族的随机化平台及上下界见`results/confirmation_geometry.json`。','',
      '三模型两两兼容而整体冲突，随机选择pair center的最坏失败率1/3。它否定“Γ类>ε意味着任意固定δ随机化任务均不可能”的过强解释。有限答案的约简及Helly见证是经典工具；有限高斯族渐近成本下界已补成独立核查草稿，尚未经外部审阅，本轮启发式最优性仍未证明。T候选归一化结果不作有限样本下界检验。见[下界草稿](FINITE_GAUSSIAN_LOWER_BOUND_ZH.md)。','',
      '## 错误控制与数值','',f"确认中观察到错误认证{int(c.false_certification.sum())}次、真模型评价区间未决{int(c.certification_truth_indeterminate.sum())}次。额外独立校准逐真模型二项区间见`calibration_errors.json`；零错误不能当作零风险。",'',
      '理论错误控制依赖自适应动作的可预测性、正确有限候选族以及有效数值界。停止输出由有理持仓/弱对偶证书定义；浮点展示权重不是额外的精确保证。似然包络依赖IEEE误差模型，未做全软件形式化验证。','',
      '## 终端规则与求解','',f"终端完成情况：{comp.get('terminal')}。模型内成功、实际成功和预算资格见`terminal_summary.json`；冷启动/缓存/贪心结果见`S3_stress.json`。",'',
      'A1-B作为信息耗尽的终端规则，不继承δ认证。旧平均代价证书开放状态保留。M=6枚举价格并保留跳点区间；M=12只报告查询结果和开放界，不声称枚举了所有跳点或证明了多项式加速。','',
      'β=0各方法均为MA端点；零观测只计算一次先验持仓。逐数据集的配对效应区间在`results/terminal_paired.json`，逐族最坏模型结果与真模型覆盖数在`results/terminal_worst_model.json`。未覆盖全部真模型的行不冒称完整最坏情形估计。未运行清单为`results/unrun_terminal.json.gz`和`results/unrun_confirmation.json.gz`。','',
      '## 最终判断的依据','',
      '1. Γ等价类判据刻画共同组合的存在性；随机化平台与统一认证必须分开。隐藏曲率负结果不能仅凭名称解释。',
      '2. 分配价值以主表和四项配对比较为准，任何优势限定本轮有限高斯族和动作成本。',
      '3. 终端前沿用于量化妥协与贪心间隙；未完成项和开放证书不删去。',
      '4. 论文转向是否值得继续，应看结构性解释和观测成本证据；不能把现有多答案识别、Ville或Helly工具重命名为原创。','',
      'CPU最终台账见`results/runtime.json`。完整轨迹按动作流种子、动作序列、排除证据和有理停止证书保存于独立运行目录，并无损归档。未自动发布GitHub。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf8')
def audit():
    cfg=read(ROOT/'config_frozen.json');checks=[]
    approved=read(ROOT/'implementation_revision.json')['approved_core_hashes'] if (ROOT/'implementation_revision.json').exists() else cfg['core_hashes']
    checks.append(dict(name='core_hashes',passed=all(sha(ROOT/k)==v for k,v in approved.items())))
    checks.append(dict(name='gate',passed=read(OUT/'gate0.json')['passed']))
    failures=[];n=0
    for path in (RUN/'confirmation').glob('*.json.gz'):
        r=readtraj(path);n+=1
        if r['stop']:
            cert=r['stop']['certificate'];ids=[j for j in range(len(cert['weights']) if False else len(cert['regret_upper'])) if r['stop']['confidence_mask']>>j&1]
            if not ids or max(cert['regret_upper'][j] for j in ids)>r['epsilon']:failures.append(str(path))
        if sum(length for _,length in r['action_rle'])!=sum(r['counts']):failures.append(str(path))
    checks.append(dict(name='stop_certificates_action_counts',passed=not failures,records=n,failures=failures))
    from .pipeline import load_family,setup,jobkey
    from .models import specs
    from .likelihood import constants
    from .sequential import simulate
    replay=[]
    for sp in specs('confirmation'):
        name=f'{sp[0]}-{sp[1]}-6-confirmation-{sp[4]}';f=load_family(name);con=constants(f)
        for policy in ('equal_cost','conflict_kl'):
            key=jobkey(f,0,0,.5,.05,1,[0,1,2,3],policy);p=RUN/'confirmation'/f'{key}.json.gz'
            if not p.exists():replay.append(dict(key=key,status='not_run'));continue
            saved=readtraj(p);geo,tab=setup(f,.5,1,[0,1,2,3]);r=simulate(f,geo,tab,con,0,0,.5,.05,1,[0,1,2,3],policy)
            acts=np.concatenate([np.repeat(a,length) for a,length in saved['action_rle']]) if saved['action_rle'] else np.array([],dtype=int)
            ok=np.array_equal(acts,r['actions']) and np.array_equal(saved['log_likelihood'],r['log_likelihood']) and saved['stop_t']==r['stop_t']
            replay.append(dict(key=key,passed=ok))
    checks.append(dict(name='independent_uncached_replay',passed=all(x.get('passed',False) for x in replay),records=replay))
    frozen=read(OUT/'freeze_summary.json')
    checks.append(dict(name='frozen_config_unchanged',passed=sha(ROOT/'config_frozen.json')==frozen['config_sha256']))
    with gzip.open(ROOT/'manifest.json.gz','rb') as z:manifest_digest=hashlib.file_digest(z,'sha256').hexdigest()
    checks.append(dict(name='manifest_unchanged',passed=manifest_digest==frozen['manifest_sha256']))
    checks.append(dict(name='cpu_below_limit',passed=used()<43200))
    write(OUT/'audit.json',dict(passed=all(x['passed'] for x in checks),checks=checks,source_hashes=sources()))
def archive():
    destination=RUN/'archives';destination.mkdir(exist_ok=True);archive=destination/'synthetic_v6.tar.zst';files=[p for p in RUN.rglob('*') if p.is_file() and 'archives' not in p.relative_to(RUN).parts and 'metrics' not in p.relative_to(RUN).parts]
    manifest={str(p.relative_to(RUN)).replace('\\','/'):sha(p) for p in files};write(destination/'manifest.json',manifest)
    with archive.open('wb') as raw:
      with zstd.ZstdCompressor(level=3).stream_writer(raw) as zs:
       with tarfile.open(fileobj=zs,mode='w|') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(RUN)).replace('\\','/'),recursive=False)
    count=0
    with archive.open('rb') as raw:
      with zstd.ZstdDecompressor().stream_reader(raw) as zs:
       with tarfile.open(fileobj=zs,mode='r|') as tar:
        for m in tar:
            if m.isfile():
                if hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()!=manifest[m.name]:raise RuntimeError('archive mismatch '+m.name)
                count+=1
    write(OUT/'archive_verification.json',dict(files=count,sha256=sha(archive),bytes=archive.stat().st_size,verified=count==len(files),excluded='archives itself and final supervisor metrics'))
def dispatch(stage,*args):{'report':report,'audit':audit,'archive':archive}[stage](*args)
