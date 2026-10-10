"""Generate report-ready tables from completed evaluations, without hand transcription."""
import pandas as pd
import numpy as np
from .common import ROOT,read

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'

def run(actions=False):
    out=ROOT/'output';suffix='_actions' if actions else ''
    summary=pd.read_csv(out/f'summary_test{suffix}.csv');s=summary[(summary.period=='test')&(summary.cost_bps==0)]
    methods=['EqualWeight','InverseVol','EM-Shrink','LedoitWolf-CompleteRows','MA','A-Tail-0.25','A-Tail-0.5','B-Tail-0.25','B-Tail-0.5','B-Max','A-Max','C-LP-Max','C-Soft','MA-power1','B-Tail-0.5-power1']
    pieces={}
    for cohort in ('long','limited'):
        t=s[s.cohort==cohort].set_index('method');rows=[];base=t.loc['MA','second_moment']
        for m in methods:
            if m not in t.index:continue
            r=t.loc[m];rows.append([m,f'{100*r.annual_vol:.3f}%',f'{1e4*r.second_moment:.5f}',f'{100*(r.second_moment/base-1):+.2f}%',f'{100*r.worst_quarter_rms:.2f}%'])
        pieces[cohort]=table(['方法','年化波动率','二阶损失 ×10⁴','损失相对 MA','最差季度 RMS 风险'],rows)
    p=pd.read_csv(out/f'paired{suffix}.csv');rows=[]
    for r in p.itertuples():rows.append(['长历史' if r.cohort=='long' else '不完整历史',r.method,r.baseline,f'{100*r.relative_difference:+.2f}%',f'[{100*r.ci_low:+.2f}%, {100*r.ci_high:+.2f}%]',f'{r.holm_p:.4f}'])
    pieces['paired']=table(['股票组','方法','基线','损失相对差异','95% 区间','Holm p'],rows)
    t=summary[(summary.period=='test')&summary.method.isin(['EM-Shrink','MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25'])];rows=[]
    for (c,m),g in t.groupby(['cohort','method']):
        g=g.set_index('cost_bps');r=g.loc[0]
        rows.append(['长历史' if c=='long' else '不完整历史',m,f'{r.turnover:.2f}',f'{100*r.cash:.3f}%',f'{100*r.stale:.3f}%',*[f'{100*g.loc[x,"annual_return"]:.2f}%' for x in [0,10,20]]])
    pieces['cost']=table(['股票组','方法','年化单边换手','平均现金','平均陈旧估值','0 bps 年化收益','10 bps','20 bps'],rows)
    ext=summary[(summary.period=='extension')&(summary.cost_bps==0)];rows=[]
    for c,g in ext.groupby('cohort'):
        g=g.set_index('method');base=g.loc['MA','second_moment']
        for m in ['EM-Shrink','MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25','B-Max']:
            rows.append(['长历史' if c=='long' else '不完整历史',m,f'{100*g.loc[m,"annual_vol"]:.3f}%',f'{100*(g.loc[m,"second_moment"]/base-1):+.2f}%'])
    pieces['extension']=table(['股票组','方法','年化波动率','损失相对 MA'],rows)
    f=pd.read_csv(out/'summary_fifty.csv');f=f[f.period=='test'];rows=[]
    for c,g in f.groupby('cohort'):
        g=g.set_index('method');base=g.loc['MA','second_moment']
        for m in ['EqualWeight','EM-Shrink','MA','A-Tail-0.5','B-Tail-0.5','B-Tail-0.25']:
            rows.append(['长历史' if c=='long' else '不完整历史',m,int(g.loc[m,'windows']),f'{1e4*g.loc[m,"second_moment"]:.5f}',f'{100*(g.loc[m,"second_moment"]/base-1):+.2f}%'])
    pieces['fifty']=table(['股票组','方法','窗口数','二阶损失 ×10⁴','损失相对 MA'],rows)
    for name,content in pieces.items():(out/f'table{suffix}_{name}.md').write_text(content,encoding='utf-8')
    print('\n'.join('\n'+name+'\n'+content for name,content in pieces.items()))

if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    run('--actions' in sys.argv)
