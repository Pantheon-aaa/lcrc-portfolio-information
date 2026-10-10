import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,PRIVATE

def run(actions=False):
    suffix='_actions' if actions else ''
    out=ROOT/'output'; folder=out/f'figures{suffix}';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':140,'font.size':10})
    s=pd.read_csv(out/f'summary_test{suffix}.csv');s=s[(s.period=='test')&(s.cost_bps==0)]
    methods=['EqualWeight','InverseVol','EM-Shrink','MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25','B-Max','A-Max','C-LP-Max','C-Soft']
    fig,ax=plt.subplots(1,2,figsize=(13,5),sharex=True)
    for a,c in zip(ax,['long','limited']):
        t=s[s.cohort==c].set_index('method').reindex(methods)
        a.barh(np.arange(len(t)),100*t.annual_vol,color=['#b1b8c2' if m in methods[:3] else '#387fb5' if m=='MA' else '#e59843' for m in methods])
        a.set_yticks(np.arange(len(t)),methods);a.invert_yaxis();a.set_xlabel('平均面板年化波动率 (%)');a.set_title('长历史组' if c=='long' else '不完整历史组');a.grid(axis='x',alpha=.2)
    fig.suptitle('2020—2025 测试期，同一执行账本，无成本');fig.tight_layout();fig.savefig(folder/'risk_comparison.png');plt.close(fig)
    p=pd.read_csv(out/f'paired{suffix}.csv');fig,ax=plt.subplots(figsize=(10,5))
    y=np.arange(len(p));mid=100*p.relative_difference.to_numpy();lo=100*p.ci_low.to_numpy();hi=100*p.ci_high.to_numpy()
    ax.errorbar(mid,y,xerr=np.array([mid-lo,hi-mid]),fmt='o',color='#387fb5',capsize=3)
    ax.set_yticks(y,[f"{'长历史' if r.cohort=='long' else '不完整历史'} | {r.method} 对 {r.baseline}" for r in p.itertuples()]);ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('二阶损失相对差异 (%)，负值代表条件尾部风险较低');ax.set_title('配对差异与 95% 区间：63 日区块同步重采样');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(folder/'paired_intervals.png');plt.close(fig)
    df=pd.read_parquet(PRIVATE/f'evaluation/daily_test{suffix}.parquet');df=df[(df.cost_bps==0)&(df.period=='test')]
    df['year']=pd.to_datetime(df.date).dt.year; fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,c in zip(ax,['long','limited']):
        for m in ['EM-Shrink','MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25']:
            z=df[(df.cohort==c)&(df.method==m)].groupby('year').ret.apply(lambda r:100*np.sqrt(252*np.mean(r**2)))
            a.plot(z.index,z.values,marker='o',label=m)
        a.set_title('长历史组' if c=='long' else '不完整历史组');a.set_ylabel('年度实现均方根风险 (%)');a.grid(alpha=.2)
    ax[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'annual_risk.png');plt.close(fig)
    d=pd.read_csv(PRIVATE/'evaluation/diagnostics_test.csv');d=d[d.method=='MA']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for c in ['long','limited']:
        x=d[d.cohort==c];ax[0].hist(x.ess,bins=25,alpha=.5,label=c);ax[1].hist(x.max_mass,bins=25,alpha=.5,label=c)
    ax[0].set_xlabel('有效场景数 ESS');ax[1].set_xlabel('最大场景质量');ax[0].set_ylabel('窗口数');ax[0].legend();fig.suptitle('测试期广义后验集中程度');fig.tight_layout();fig.savefig(folder/'posterior_concentration.png');plt.close(fig)

if __name__=='__main__':
    import sys
    run('--actions' in sys.argv)
