"""Recreate scientific figures from the committed synthetic records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; PAPER=ROOT/'results'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def run():
    s1=pd.read_csv(OUT/'s1.csv'); summary=pd.read_csv(OUT/'summary.csv')
    s2=pd.read_parquet(OUT/'S2_instances.parquet'); s3=pd.read_parquet(OUT/'S3_instances.parquet')
    timing=json.loads((OUT/'pilot_timing.json').read_text()); tuning=json.loads((OUT/'tuning.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7,2.35),layout='constrained')
    for ax,mech,title in zip(axes,['compatibility','hidden_curvature','boundary'],['Multiple-model compatibility','Hidden curvature','Different critical cones']):
        for d,color in [(6,'#2563aa'),(20,'#d26822')]:
            g=s1[(s1.mechanism==mech)&(s1.d==d)&(s1.seed==0)].sort_values('t')
            ax.plot(g.t,g.ratio/g.psi,'o-',ms=3,color=color,label=f'd={d}')
        ax.axhline(1,color='black',lw=.8,ls='--'); ax.set_xscale('log'); ax.set_title(title,fontsize=9)
        ax.set_xticks([g.t.min(),g.t.max()],[f'{g.t.min():.2g}',f'{g.t.max():.2g}'])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Perturbation t'); ax.set_ylabel(r'$\Gamma_t/(t^2\Psi)$'); ax.legend(fontsize=7)
    fig.savefig(PAPER/'structure.pdf'); fig.savefig(OUT/'structure.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.55),layout='constrained')
    methods=['LCRC','Affine-MLE','Model-average','EM-shrink','Likelihood-absolute']
    colors=['#bb3e35','#e1a128','#176d9c','#727b76','#9256a1']
    for ax,n in zip(axes,[64,256]):
        for method,color in zip(methods,colors):
            g=summary[(summary.stage=='S2')&(summary.n_blocks==n)&(summary.method==method)].sort_values('joint_blocks')
            ax.errorbar(np.arange(4),g['mean']*100,yerr=1.96*g.se*100,label=method,color=color,marker='o',ms=3,capsize=2,lw=1)
        ax.set_xticks(range(4),[0,4,16,64]); ax.set_xlabel('Joint blocks m'); ax.set_title(f'{n} blocks; fixed scalar budget')
        ax.set_ylabel('Regret (% of fixed risk unit)')
    axes[1].legend(fontsize=6.5,loc='upper right');fig.savefig(PAPER/'comparison.pdf');fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.35),layout='constrained')
    cover=summary[(summary.stage=='S3')&(summary.variant=='base')&(summary.method=='LCRC')].sort_values('max_cells')
    axes[0].plot(cover.max_cells,cover.volume_fraction,'o-',label='Retained box volume')
    axes[0].plot(cover.max_cells,cover.mean_certificate_scaled*100,'s-',label='Bound (% of risk unit)')
    axes[0].set_xscale('log',base=2);axes[0].set_xlabel('Cell budget');axes[0].legend(fontsize=7)
    axes[1].plot(cover.max_cells,cover.median_seconds,'o-',color='#bb3e35');axes[1].set_xscale('log',base=2)
    axes[1].set_yscale('log');axes[1].set_xlabel('Cell budget');axes[1].set_ylabel('Median LCRC seconds')
    fig.savefig(PAPER/'cover_stress.pdf');fig.savefig(OUT/'cover_stress.png',dpi=180);plt.close(fig)


if __name__ == "__main__":
    run()
