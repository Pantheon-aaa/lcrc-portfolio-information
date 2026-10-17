import itertools,time,tarfile,hashlib,sys,platform,importlib.metadata
import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def binomial(k,n):return [0. if k==0 else float(beta_dist.ppf(.025,k,n-k+1)),1. if k==n else float(beta_dist.ppf(.975,k+1,n-k))]
def bootstrap_family(values,kinds):
    rng=np.random.default_rng(seed('bootstrap'));values=np.asarray(values);groups=[np.flatnonzero(np.array(kinds)==k) for k in sorted(set(kinds))]
    sims=np.mean(np.column_stack([values[rng.choice(g,(10000,len(g)),replace=True)] for g in groups]),axis=1)
    return np.quantile(sims,[.025,.975]).tolist()

def e1_summary():
    rows_=read(OUT/'e1_records.json');complete=[r for r in rows_ if r['status']=='complete'];counts={};paired=[];month={}
    for r in complete:
        for stale in (0,20,63,126):
            e=r['evaluation'][f'H63_stale{stale}'];s=counts.setdefault(str(stale),dict(available=0,success=0,loss=[]))
            if 'loss' in e:s['available']+=1;s['success']+=e['success_lower'];s['loss'].append(e['loss'])
        a=r['evaluation']['H63_stale0'];b=r['evaluation']['H63_stale63']
        if 'loss' in a and 'loss' in b:paired.append(dict(id=r['id'],loss_difference=b['loss']-a['loss'],success_difference=b['success_lower']-a['success_lower']))
        m=month.setdefault(r['date'][:7],[0.,0.,0.]);m[0]+=1;m[1]+=1-b['success_upper'];m[2]+=1-b['success_lower']
    for s in counts.values():s.update(success_rate=s['success']/s['available'] if s['available'] else None,mean_loss=float(np.mean(s.pop('loss'))))
    # Preserve the original 72-month calendar; unobserved targets contribute unknown intervals.
    months=sorted(set(r['date'][:7] for r in rows_));vec=np.array([month.get(m,[0.,0.,0.]) for m in months]);bs={};rng=np.random.default_rng(seed('e1-bootstrap'))
    for block in (3,6,12):
        starts=rng.integers(0,len(months),(10000,int(np.ceil(len(months)/block))));ix=(starts[:,:,None]+np.arange(block))%len(months);ix=ix.reshape(10000,-1)[:,:len(months)];z=vec[ix].sum(1)
        bs[str(block)]=dict(known_internal_conditioned_interval=[float(np.quantile(z[:,1]/z[:,0],.025)),float(np.quantile(z[:,2]/z[:,0],.975))])
    r=dict(planned=len(rows_),valid=len(complete),all_MA_covered=sum(x['internal_all_covered'] for x in complete),A_mean=np.mean([x['A_upper'] for x in complete]),A_max=max(x['A_upper'] for x in complete),gamma_ratio_mean=np.mean([x['gamma_upper_over_epsilon'] for x in complete]),gamma_ratio_max=max(x['gamma_upper_over_epsilon'] for x in complete),labels=counts,strict_approx_paired=len(paired),paired_max_abs_loss_difference=max(abs(x['loss_difference']) for x in paired),paired_success_difference_mean=np.mean([x['success_difference'] for x in paired]),known_internal_failure_interval=[vec[:,1].sum()/vec[:,0].sum(),vec[:,2].sum()/vec[:,0].sum()],calendar_joint_internal_and_future_failure_interval=[vec[:,1].sum()/len(rows_),(vec[:,2].sum()+len(rows_)-len(complete))/len(rows_)],bootstrap=bs,EM_final_nonconverged_instances=sum(x['em_max_iterations']>0 for x in complete))
    write(OUT/'e1_summary.json',r);write(OUT/'e1_strict_approx_pairs.json',paired)
    write(OUT/'e1_label_status_counts.json',{str(stale):pd.Series([x['evaluation'][f'H63_stale{stale}'].get('status','not_recorded') for x in complete]).value_counts().to_dict() for stale in (0,20,63,126)})
    metric_names=('Lstar_upper_over_u','oracle_L1_spread','entropy','ESS','old_MA_recompute_L1')
    write(OUT/'e1_additional_diagnostics.json',dict(metrics={k:dict(mean=float(np.mean([x[k] for x in complete])),min=float(min(x[k] for x in complete)),max=float(max(x[k] for x in complete))) for k in metric_names},gamma_status_counts=pd.Series([x['gamma_status'] for x in complete]).value_counts().to_dict(),A1_MA_same=sum(a['same'] for x in complete for a in x['a1_same_MA']),A1_queries=sum(len(x['a1_same_MA']) for x in complete),source_component_unresolved=len(rows_)-len(complete),future_label_unresolved=len(complete)-counts['63']['available'],saturated=len(complete),internally_active=0))
    bins=[]
    for group in ('year','domain','missingness'):
        for key in sorted(set(x['date'][:4] if group=='year' else x['domain'] if group=='domain' else ('unavailable' if x['evaluation']['H63_stale63'].get('quality') is None else 'approximate' if x['evaluation']['H63_stale63']['quality'].get('approximate_fraction',0)>0 else 'complete_raw') for x in complete)):
            rr=[x for x in complete if (x['date'][:4] if group=='year' else x['domain'] if group=='domain' else ('unavailable' if x['evaluation']['H63_stale63'].get('quality') is None else 'approximate' if x['evaluation']['H63_stale63']['quality'].get('approximate_fraction',0)>0 else 'complete_raw'))==key];ee=[x['evaluation']['H63_stale63'] for x in rr];bins.append(dict(group=group,key=key,n=len(rr),failure_lower=np.mean([1-x['success_upper'] for x in ee]),failure_upper=np.mean([1-x['success_lower'] for x in ee]),A_mean=np.mean([x['A_upper'] for x in rr])))
    write(OUT/'e1_strata.json',bins)
    # Trace extreme raw-return rows to past training and future label windows.
    ext=read(RUN/'e1_source_extremes.json');panels={p['id']:p for p in read(OLD/'panels.json')};idx=read(OLD.parent/'ashare_v5/data/index.json');exposure=[]
    for x in ext:
        di=idx['dates'].index(x['date']);col=idx['symbols'].index(x['symbol']);past=[];future=[]
        for rr in rows_:
            panel=panels[rr['id'].removeprefix('R2_').removesuffix('_N50_M12')]
            if col not in panel['ranking'][:50]:continue
            if panel['t']-566<=di<=panel['t']:past.append(rr['id'])
            if panel['t']+2<=di<=panel['t']+64:future.append(rr['id'])
        exposure.append(dict(date=x['date'],raw_return=x['return_total'],past_training_instances=len(past),future_H63_instances=len(future),past_ids=past,future_ids=future,classification='not independently verified'))
    write(OUT/'e1_extreme_exposure.json',dict(examined_source_rows=len(exposure),past_training_exposed_rows=sum(x['past_training_instances']>0 for x in exposure),future_label_exposed_rows=sum(x['future_H63_instances']>0 for x in exposure),source_max_return=max(x['raw_return'] for x in exposure),row_details='private e1_source_extremes.json; no security-level dates or returns published'))
    exception_exposure=[]
    for x in read(OLD/'valuation/exceptions.json'):
        di=idx['dates'].index(x['date']);col=idx['symbols'].index(x['symbol']);ids=[]
        for rr in rows_:
            panel=panels[rr['id'].removeprefix('R2_').removesuffix('_N50_M12')]
            if col in panel['ranking'][:50] and panel['t']+2<=di<=panel['t']+64:ids.append(rr['id'])
        exception_exposure.append(dict(date=x['date'],price_return_error=x['error'],future_H63_instances=len(ids),ids=ids))
    write(OUT/'e1_valuation_exception_exposure.json',dict(examined_exceptions=len(exception_exposure),future_label_exposed_exceptions=sum(x['future_H63_instances']>0 for x in exception_exposure),source_file_sha256=sha(OLD/'valuation/exceptions.json'),row_details='retained only in the frozen private valuation source'))
    fig,ax=plt.subplots(figsize=(7,4));ax.hist([x['A_upper'] for x in complete],bins=24);ax.axvline(1,color='r',linestyle='--');ax.set(xlabel='max library regret(MA) / epsilon',ylabel='Frozen target instances',title='E1: all 396 valid MA targets already cover the library');fig.tight_layout();fig.savefig(OUT/'e1_saturation.png',dpi=160);plt.close(fig)
    return r

def collect(split):
    data=[];terminal=[];failed=[];cost=[]
    for path in sorted((RUN/split).glob('*.json.gz')):
        raw=read(path);cost.append(raw.get('cpu',0));j=raw['job']
        if raw['status']!='complete':failed.append(dict(job=j,error=raw.get('error')));continue
        base=dict(family=j['family'],rep=j['rep'],design=j['design'],kind=raw['kind'],case=raw['case'],panel=raw['panel'],perturbation=raw['perturbation'],strength=raw['strength'])
        data.extend([dict(base,**{k:v for k,v in row.items() if k not in ('weights','confidence_set')}) for row in raw['rows']])
        for row in raw['terminal']:
            a=dict(base,**{k:v for k,v in row.items() if k not in ('weights','raw')})
            details=row.get('raw',{});a['raw_status']=details.get('status','reference' if row.get('budget_applies') is False else 'missing')
            for key in ('primary_mass_gap','secondary_price_gap','optimal_mass_lower','optimal_mass_upper','geometry_calls','price_calls'):
                if key in details:a[key]=number(details[key])
            a['posterior_precision_open']=details.get('status')=='posterior_precision_open'
            terminal.append(a)
    return pd.DataFrame(data),pd.DataFrame(terminal),failed,cost

def sequential_summary(df):
    out=[];keys=['design','n','ratio','method','panel','perturbation']
    for key,g in df.groupby(keys,dropna=False):
        family=g.groupby('family')[['certified','false_certification','truth_indeterminate']].mean();n=len(g);k=int(g.false_certification.sum())
        out.append(dict(zip(keys,key),records=n,families=len(family),certification_rate=family.certified.mean(),false_certification_rate=family.false_certification.mean(),false_certification_count=k,binomial_interval_descriptive=binomial(k,n),numerically_indeterminate=int(g.truth_indeterminate.sum()),rho_median=float(g.rho.median()) if 'rho' in g and g.rho.notna().any() else None))
    return out

def tests(df):
    result=[]
    for design,panel in itertools.product(('Separated','Joint'),('blind','all_misspecified')):
        g=df[(df.design==design)&(df.n==2048)&(df.ratio==1.25)&(df.method.isin(['data_decision','data_operator']))]
        g=g[g.perturbation=='blind'] if panel=='blind' else g[g.panel.isin(['leave_one','shift'])]
        tab=g.pivot_table(index=['family','kind','case','rep'],columns='method',values='certified',aggfunc='first').dropna()
        tab['diff']=tab['data_decision'].astype(float)-tab['data_operator'].astype(float);vals=tab.groupby(['family','kind'])['diff'].mean();x=vals.to_numpy();kinds=[t[1] for t in vals.index]
        obs=float(np.mean(x));perms=np.array(list(itertools.product((-1,1),repeat=len(x))))@x/len(x);p=float(np.mean(perms>=obs-1e-15))
        result.append(dict(design=design,panel=panel,effect=obs,interval=bootstrap_family(x,kinds),p=p,family_effects=[dict(family=f,kind=k,effect=v) for (f,k),v in vals.items()],paired=len(tab)))
    order=np.argsort([x['p'] for x in result]);prev=0
    for rank,i in enumerate(order):prev=max(prev,min(1,(len(result)-rank)*result[i]['p']));result[i]['holm_p']=prev
    return result

def terminal_summary(df):
    if df.empty:return []
    # Same datum, explicit error intervals. Means of continuous losses use evaluable rows only.
    df['true_success_lower']=df.true_success_lower.astype('boolean').fillna(False).astype(float);df['true_success_upper']=df.true_success_upper.astype('boolean').fillna(True).astype(float)
    out=[]
    for key,g in df.groupby(['design','n','ratio','method','beta'],dropna=False):
        fam=g.groupby('family')[['true_success_lower','true_success_upper']].mean();valid=g[g.truth_regret_upper.notna()]
        out.append(dict(zip(['design','n','ratio','method','beta'],key),records=len(g),families=len(fam),errors=int(g['error'].notna().sum()) if 'error' in g else 0,success_lower=fam.true_success_lower.mean(),success_upper=fam.true_success_upper.mean(),mean_regret=valid.groupby('family').truth_regret_upper.mean().mean(),p95_regret=float(valid.truth_regret_upper.quantile(.95)),failure_regret=float(valid[valid.true_success_upper==0].truth_regret_upper.mean()) if (valid.true_success_upper==0).any() else None,budget_interval_certified_rate=float(g.budget_certified.astype('boolean').fillna(False).mean()) if 'budget_certified' in g else None,MA_identity_rate=float(g.budget_MA_identity.astype('boolean').fillna(False).mean()) if 'budget_MA_identity' in g else None,status_counts=g.raw_status.value_counts().to_dict(),posterior_precision_open=int(g.posterior_precision_open.sum()),max_primary_gap=float(g.primary_mass_gap.max()) if 'primary_mass_gap' in g else None,mean_primary_gap=float(g.primary_mass_gap.mean()) if 'primary_mass_gap' in g else None))
    return out

def review():
    e1=e1_summary();df,term,fail,cost=collect('development')
    write(OUT/'development_review.json',dict(records=len(cost),failed=fail,cpu_p90=float(np.quantile(cost,.9)),certification=sequential_summary(df),terminal_errors=term[term['error'].notna()][['family','case','method','error']].to_dict('records') if 'error' in term else []))
    print('E1',e1,flush=True)
    print('DEVELOPMENT_REVIEW',len(cost),'failed',len(fail),'p90',np.quantile(cost,.9),flush=True)

def smoke():
    from .executor import smoke as run
    run()

def radius_diagnostics(df):
    part=df[(df.method=='data_decision')&(df.ratio==1.25)].copy();part['width_over_slack']=part.rho/(part.epsilon-part.gamma_upper)
    widths=[]
    for key,g in part.groupby(['design','n','panel','perturbation']):
        widths.append(dict(zip(['design','n','panel','perturbation'],key),records=len(g),minimum=float(g.width_over_slack.min()),median=float(g.width_over_slack.median()),p95=float(g.width_over_slack.quantile(.95)),maximum=float(g.width_over_slack.max())))
    event=part.groupby(['family','kind','case','rep','design']).confidence_event.all().reset_index()
    coverage=[]
    coverage_cells=[]
    for key,g in event.groupby(['family','case','design']):
        k=int((~g.confidence_event.astype(bool)).sum());coverage_cells.append(dict(zip(['family','case','design'],key),paths=len(g),misses=k,exact_binomial_interval=binomial(k,len(g))))
    for design,g in event.groupby('design'):
        k=int((~g.confidence_event.astype(bool)).sum());coverage.append(dict(design=design,paths=len(g),simultaneous_misses=k,miss_rate=k/len(g),family_equal_miss_rate=1-g.groupby('family').confidence_event.mean().mean(),binomial_descriptive=binomial(k,len(g)),nominal_delta=.05))
    write(OUT/'radius_widths.json',widths);write(OUT/'radius_coverage.json',coverage);write(OUT/'radius_coverage_cells.json',coverage_cells)

def terminal_diagnostics(term):
    out=[];panels=[]
    for key,g in term.groupby(['design','n','ratio','method','beta','panel','perturbation'],dropna=False):
        ff=g.groupby('family')[['true_success_lower','true_success_upper']].mean();valid=g[g.truth_regret_upper.notna()].copy();valid['normalized_regret']=valid.truth_regret_upper/valid.epsilon
        panels.append(dict(zip(['design','n','ratio','method','beta','panel','perturbation'],key),records=len(g),success_lower=float(ff.true_success_lower.mean()),success_upper=float(ff.true_success_upper.mean()),mean_regret=float(valid.groupby('family').truth_regret_upper.mean().mean()),mean_regret_over_epsilon=float(valid.groupby('family').normalized_regret.mean().mean()),p95_regret=float(valid.truth_regret_upper.quantile(.95)),external_certified_rate=float(g.external_certified.astype('boolean').fillna(False).mean()) if 'external_certified' in g else None))
    write(OUT/'terminal_panels.json',panels)
    for key,g in term.groupby(['family','design','n','ratio','method','beta'],dropna=False):
        a=g.groupby('case')[['true_success_lower','true_success_upper']].mean()
        out.append(dict(zip(['family','design','n','ratio','method','beta'],key),evaluated_cases=len(a),expected_cases=22,worst_case_success_lower=float(a.true_success_lower.min()),worst_case_success_upper=float(a.true_success_upper.min()),complete_case_panel=len(a)==22))
    write(OUT/'terminal_worst_cases.json',out)
    pairs=[]
    for key,g in term[term.method.isin(['A1-B','Greedy'])].groupby(['design','n','ratio','beta'],dropna=False):
        ids=['family','kind','case','rep'];a=g[g.method=='A1-B'].set_index(ids);b=g[g.method=='Greedy'].set_index(ids);ix=a.index.intersection(b.index)
        if not len(ix):continue
        aa=a.loc[ix];bb=b.loc[ix];lo=aa.true_success_lower.astype(float)-bb.true_success_upper.astype(float);hi=aa.true_success_upper.astype(float)-bb.true_success_lower.astype(float)
        delta=(aa.truth_regret_upper-bb.truth_regret_upper).groupby('family').mean()
        pairs.append(dict(zip(['design','n','ratio','beta'],key),paired=len(ix),success_effect_lower=float(lo.groupby('family').mean().mean()),success_effect_upper=float(hi.groupby('family').mean().mean()),mean_regret_difference=float(delta.mean()),interpretation='descriptive paired terminal comparison; not one of the four primary tests'))
    write(OUT/'terminal_paired.json',pairs)

def mechanism_figure(df,term):
    fig,axes=plt.subplots(1,2,figsize=(11,4));kinds=['known','convex','leave_one','blind','rank1','random_psd']
    sel=df[(df.design=='Joint')&(df.n==2048)&(df.ratio==1.25)]
    for method,offset in [('library',-.3),('oracle_decision',-.1),('oracle_operator',.1),('data_decision',.3)]:
        a=sel[sel.method==method].groupby(['family','perturbation']).certified.mean().groupby('perturbation').mean()
        axes[0].bar(np.arange(6)+offset,[a.get(k,0) for k in kinds],width=.2,label=method)
    axes[0].set(xticks=np.arange(6),xticklabels=kinds,ylabel='Certification rate',ylim=(0,1.08),title='Joint: oracle geometry vs statistical calibration');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(fontsize=7)
    for design in ('Joint','Separated'):
        g=df[(df.design==design)&(df.ratio==1.25)&(df.method=='data_decision')].copy();g['width']=g.rho/(g.epsilon-g.gamma_upper)
        a=g.groupby('n').width.median();axes[1].plot(a.index,a.values,marker='o',label=design)
    axes[1].axhline(1,color='r',linestyle='--');axes[1].set(xscale='log',yscale='log',xlabel='Total observation blocks',ylabel='Median radius / available epsilon slack',title='Why data certificates can remain empty');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'e2_geometry_calibration.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,design in zip(axes,('Separated','Joint')):
        g=term[(term.design==design)&(term.ratio==.5)&(term.n>0)]
        for method,beta in [('MA',0),('Greedy',.1),('Greedy',.25),('A1-B',.1),('A1-B',.25)]:
            h=g[(g.method==method)&(g.beta==beta)];a=h.groupby(['family','n']).true_success_lower.mean().groupby('n').mean();ax.plot(a.index,a.values,marker='o',label=f'{method} beta={beta}')
        ax.set(xscale='log',xlabel='Total blocks (half pilot)',ylabel='Actual success rate',title=design);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'terminal_success.png',dpi=160);plt.close(fig)

def certification_figure(df):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,design in zip(axes,('Separated','Joint')):
        g=df[(df.design==design)&(df.ratio==1.25)&(df.panel.isin(['leave_one','shift']))]
        for method in ('library','oracle_decision','data_decision','data_operator','v6_pilot_CS'):
            a=g[g.method==method].groupby(['family','n']).certified.mean().groupby('n').mean();ax.plot(a.index,a.values,marker='o',label=method)
        ax.set(xscale='log',ylim=(-.02,1.02),xlabel='Total observation blocks',ylabel='Certification rate',title=f'{design}: leave-one + shifts');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'e2_certification.png',dpi=160);plt.close(fig)

def formula_floor():
    from .radius import factors,projection
    entries=[];lookup={};names=sorted({j['family'] for j in read(ROOT/'manifest.json.gz')})
    for name in names:
        raw=read(RUN/'families'/f'{name}.json');d=raw['d'];P=projection(d);cap=.2
        for library,a in raw['libraries'].items():
            Q=np.array(a['Q']);scale=min(cap*min(np.trace(q) for q in Q)/d,(cap-1/d)*min(np.trace(P@q@P) for q in Q)/d)
            for n in (32,256,2048):
                low,high,_=factors(n//2,d,.05,3);kappa=(high-low)/(high+low);floor=kappa*scale;slack=.25*a['gamma']['upper']
                row=dict(family=name,library=library,n=n,kappa=kappa,formula_floor=floor,epsilon_slack=slack,floor_over_slack=floor/slack)
                entries.append(row);lookup[(name,library,n)]=floor
    smallest=np.inf;bad=[];comparisons=0
    for path in (RUN/'confirmation').glob('*.json.gz'):
        r=read(path)
        if r['status']!='complete':continue
        library=r['case'] if r['panel']=='leave_one' else 'all'
        for a in r['rows']:
            if a['method'] not in ('data_decision','data_operator') or a['ratio']!=1.25:continue
            floor=lookup[(r['job']['family'],library,a['n'])];smallest=min(smallest,a['rho']-floor);comparisons+=1
            if a['rho']<floor*(1-1e-8):bad.append(dict(job=r['job'],method=a['method'],n=a['n']))
    result=dict(post_hoc_explanatory_diagnostic=True,not_an_information_theoretic_lower_bound=True,entries=entries,compared_records=comparisons,minimum_observed_radius_minus_formula_floor=smallest,violations=bad,all_libraries_at_n2048_floor_exceeds_slack=all(x['floor_over_slack']>1 for x in entries if x['n']==2048),minimum_n2048_floor_over_slack=min(x['floor_over_slack'] for x in entries if x['n']==2048))
    write(OUT/'calibration_formula_floor.json',result)
    widths=read(OUT/'radius_widths.json');grid=(32,256,2048);fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(grid,[min(x['floor_over_slack'] for x in entries if x['n']==n) for n in grid],marker='o',label='Minimum formula floor over all libraries')
    ax.plot(grid,[min(x['minimum'] for x in widths if x['n']==n) for n in grid],marker='o',label='Minimum observed data radius over all paths')
    ax.axhline(1,color='r',linestyle='--',label='Certification threshold');ax.set(xscale='log',yscale='log',xlabel='Total blocks (half calibration)',ylabel='Radius / epsilon slack',title='Post-hoc diagnosis of the chosen calibration formula');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'calibration_floor.png',dpi=160);plt.close(fig)
    return result

def report(mode='full'):
    if mode=='plots':
        certification_figure(pd.read_json(OUT/'certificate_aggregate.json.gz'));print('PLOT REFRESH',flush=True);return
    e1=e1_summary();df,term,fail,cost=collect('confirmation');cfg=read(ROOT/'config_frozen.json');planned=read(ROOT/'manifest.json.gz');done={p.name.removesuffix('.json.gz') for p in (RUN/'confirmation').glob('*.json.gz')}
    from .pipeline import jobkey
    unrun=[j for j in planned if jobkey(j) not in done];write(OUT/'unrun.json.gz',unrun);write(OUT/'failures.json',fail);write(OUT/'completion.json',dict(planned=len(planned),processed=len(done),failed=len(fail),unrun=len(unrun)))
    summary=sequential_summary(df);tt=tests(df);ts=terminal_summary(term);write(OUT/'sequential_summary.json',summary);write(OUT/'four_tests.json',tt);write(OUT/'terminal_summary.json',ts);radius_diagnostics(df);terminal_diagnostics(term);mechanism_figure(df,term);floor=formula_floor()
    df.to_json(OUT/'certificate_aggregate.json.gz',orient='records',compression='gzip',double_precision=15);term.replace({np.inf:'inf',-np.inf:'-inf'}).to_json(OUT/'terminal_aggregate.json.gz',orient='records',compression='gzip',double_precision=15)
    columns=['family','case','rep','design','n','ratio','method','beta']
    write(OUT/'terminal_failures.json',term[term['error'].notna()][columns+['error']].to_dict('records') if 'error' in term else [])
    write(OUT/'terminal_fallbacks.json',term[term.fallback.notna()][columns+['fallback']].to_dict('records') if 'fallback' in term else [])
    stops=[]
    for key,g in df.groupby(['family','kind','case','rep','design','ratio','method']):
        g=g.sort_values('n');yes=g[g.certified];first=yes.iloc[0] if len(yes) else None;stops.append(dict(zip(['family','kind','case','rep','design','ratio','method'],key),certified=first is not None,false_certification=False if first is None else bool(first.false_certification),false_certification_upper=False if first is None else bool(first.false_certification or first.truth_indeterminate),truth_indeterminate=False if first is None else bool(first.truth_indeterminate),restricted_cost=float(g.cost.max() if first is None else first.cost),censored=first is None))
    stop=pd.DataFrame(stops);ss=[];error_cells=[]
    for key,g in stop.groupby(['family','case','design','ratio','method']):
        k=int(g.false_certification.sum());ku=int(g.false_certification_upper.sum());error_cells.append(dict(zip(['family','case','design','ratio','method'],key),paths=len(g),errors_lower=k,errors_upper=ku,exact_binomial_interval_with_numeric_uncertainty=[binomial(k,len(g))[0],binomial(ku,len(g))[1]],scope='independent repeats within this fixed family/truth/view/method; not a simultaneous across-cell interval'))
    write(OUT/'error_cells.json',error_cells)
    for key,g in stop.groupby(['design','ratio','method']):
        ff=g.groupby('family')[['certified','false_certification','false_certification_upper','truth_indeterminate','restricted_cost','censored']].mean();ss.append(dict(zip(['design','ratio','method'],key),**ff.mean().to_dict()))
    write(OUT/'stopping_summary.json',ss)
    certification_figure(df)
    main=[x for x in ss if x['ratio']==1.25]
    lines=['# v7：风险库有效性与决策充分性实验报告','','本轮支持将实际决策收敛为MA饱和检查＋预算Greedy，保留A1-B作为核验工具。库外误差的几何机制成立，但当前数据半径尚不能成为可用认证算法，条件E3不启动。这个判断不把oracle对照当成现实方法，也不将旧实验与新确认合并为一项检验。','','## 执行范围','',f"确认数据流计划{len(planned)}，已处理{len(done)}，失败{len(fail)}，未运行{len(unrun)}。独立风险族为8个；多个检查点、阈值和方法不算新风险族。",'',
        '## E1：真实风险库饱和','',f"{e1['valid']}个有效目标全部满足MA模型内覆盖；最大regret/ε均值{e1['A_mean']:.4f}、最大{e1['A_max']:.4f}。Γ/ε均值{e1['gamma_ratio_mean']:.4f}、最大{e1['gamma_ratio_max']:.4f}。",'',
        f"严格/近似共同可评价{e1['strict_approx_paired']}项，最大损失差{e1['paired_max_abs_loss_difference']:.3g}。不同口径总体达标率不可忽略新增窗口的样本组成。已知内部成功窗口的未来失败识别区间为[{e1['known_internal_failure_interval'][0]:.2%}, {e1['known_internal_failure_interval'][1]:.2%}]；不是总体真风险失败率。",'',
        '旧主792次A1-B与MA同持仓；重新计算MA的L1差异为0。后验ESS均值8.30，候选oracle最大L1分歧的均值0.987，因此饱和不能简单解释为“只剩一个模型”或“所有oracle持仓相同”。12个专家仍有配置差异，只是ε覆盖了其风险regret分歧。','',
        'Γ诊断使用旧凸求解器：393项状态为AlmostSolved、3项为Solved，未升级为严格数值认证。MA覆盖结论直接来自逐模型regret上界和离阈值的明显余量，不依赖该Γ求解器达到严格最优。24个未来标签未决来自cross_phase_label；另有36个成分核验目标未决，均保留在原432项日历中。','',
        '![旧库饱和](results/e1_saturation.png)','',
        '## E2：内部充分与库外半径','',
        '|观测|方法|截至2048认证率|首次认证错误率|计划检查的截断平均成本|','|---|---|---:|---:|---:|']
    for x in main:lines.append(f"|{x['design']}|{x['method']}|{x['certified']:.4%}|[{x['false_certification']:.4%}, {x['false_certification_upper']:.4%}]|{x['restricted_cost']:.2f}|")
    lines+=['','### Joint最终检查点的分场景结果','','|情形|库内规则错误认证|oracle决策界认证|oracle算子界认证|数据决策界认证|旧v6错误认证|','|---|---:|---:|---:|---:|---:|']
    for kind in ('known','convex','leave_one','blind','rank1','random_psd'):
        sub=df[(df.design=='Joint')&(df.n==2048)&(df.ratio==1.25)&(df.perturbation==kind)]
        def rate(method,event):return float(sub[sub.method==method].groupby('family')[event].mean().mean())
        lines.append(f"|{kind}|{rate('library','false_certification'):.3%}|{rate('oracle_decision','certified'):.3%}|{rate('oracle_operator','certified'):.3%}|{rate('data_decision','certified'):.3%}|{rate('v6_pilot_CS','false_certification'):.3%}|")
    lines+=['','每族的22个预设真风险情形等权，不代表市场分布。known属于有限高斯族；convex仅表示协方差在凸库内。预设“all_misspecified”比较特指leave_one和shift面板，不包含convex的似然失配。库内规则和oracle几何在部分情形不依赖重复数据，重复观测不能扩大其独立几何样本数。','','![几何与校准](results/e2_geometry_calibration.png)']
    lines+=['','所有认证成本按32/256/2048三个计划检查点记录。库内或oracle规则有时可在零观测判断，因此该成本不是其理论最小样本复杂度。ratio=.5下，全库Γ+非负ρ必不能认证，这是保留的几何诊断；终端决策和缩小集合的v6对照另评。','','数据方法错误率必须与认证率一起解释：始终拒绝也会得到零错误，零认证不能称为有用的风险控制。oracle半径仅作解释参照。旧v6置信集在失配面板无真模型覆盖保证。','',
        '![库外面板各检查点证书比例](results/e2_certification.png)','','图中仅含leave_one和shift面板，展示每个检查点的当前证书比例，不是累计停止比例。错库时旧v6的历史交集置信集可变空，因此这条曲线可能下降；前面的停止表保留首次认证事件。','','## 四项预设比较','','|观测|面板|认证率差|95%族级区间|Holm p|','|---|---|---:|---|---:|']
    for x in tt:lines.append(f"|{x['design']}|{x['panel']}|{x['effect']:.6f}|{x['interval']}|{x['holm_p']:.6f}|")
    lines+=['','## 冻结后的解释性诊断：公式宽度下限','',f"在n=2048下，所有公开候选库中最小的公式宽度下限/ε余量为{floor['minimum_n2048_floor_over_slack']:.3f}。是否全部超过1：{floor['all_libraries_at_n2048_floor_exceeds_slack']}。逐记录一致性核查{floor['compared_records']}次，违反{len(floor['violations'])}次。",'',
        '该下限针对本轮坐标区间与统一振幅上界的组合公式，不是实际决策距离下界，更不是任何方法的信息论不可能性。它在确认开始后作为解释推导，未用于改动冻结方案或增加检验。详细证明见THEORY_ZH.md。','','![公式下限](results/calibration_floor.png)']
    lines+=['','## 数值与终端资格','','均值/尾部regret、预算资格与终端成功率见`results/terminal_summary.json`。只使用每单元预定前10重复，零样本按库复用。实际持仓的外部认证与几何中心的存在性判据分开。旧预算开放状态不自动修复。','',
        '![终端持仓](results/terminal_success.png)','','分风险情形的结果在`results/terminal_panels.json`，Greedy配对差在`results/terminal_paired.json`。未知数值预算与MA同持仓标记分别记录；身份相同不将近零L*的浮点误差自动改成严格预算证明。每个固定风险情形的错误率精确二项区间在`results/error_cells.json`，多情形合并的二项区间仅作描述，不宣称统一覆盖。','',
        '新半径为卡方区间加向外浮点余量，依赖正确高斯及数值误差假设；不是全软件形式化验证。E1的Γ界是旧凸求解器数值界，保留求解状态。','',
        'E3是否启动见`results/e3_gate.json`。理论推导及局限见[THEORY_ZH.md](THEORY_ZH.md)，最终论文证据整理见[PAPER_SYNTHESIS_ZH.md](PAPER_SYNTHESIS_ZH.md)。CPU、失败和未完成项全部保留。']
    lines+=['','### 终端达标与平均代价摘录（ε=0.5Γ，22种真风险等权）','','|观测|总块数|方法|β|实际达标率|均值regret|p95 regret|区间预算认证率|','|---|---:|---|---:|---:|---:|---:|---:|']
    for x in ts:
        if x['ratio']!=.5 or x['n'] not in (0,32,2048) or not (x['method']=='MA' or x['method'] in ('Greedy','A1-B') and x['beta'] in (.1,.25)):continue
        lines.append(f"|{x['design']}|{x['n']}|{x['method']}|{x['beta']}|{x['success_lower']:.3%}|{x['mean_regret']:.6g}|{x['p95_regret']:.6g}|{x['budget_interval_certified_rate']:.2%}|")
    lines+=['','零样本仅记录一个共同先验视图。上表是描述性终端比较，不属于四项主检验；大量重复不增加独立风险族数。MA零预算返回政策不变，其浮点oracle误差区间仍开放。精确求解器没有稳定超过贪心，信息增加后预算资格也会因后验集中和近零L*而开放。','',
        'Joint下MA在known面板的达标率从n=32的74.58%升至n=2048的100%；在convex面板却从56.25%降至26.25%。后者的协方差在凸库内，但生成密度不属于有限候选族。更多观测可使错误工作模型的后验更集中，不能将合并曲线下降解释为“数据本身有害”。该机制与有限正确模型的Bayes均值最优性不矛盾。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','clarabel','cvxpy','mpmath','pandas','matplotlib','zstandard','psutil')},BLAS_threads=1,max_numerical_workers=2))
    print('REPORT',len(df),len(term),flush=True)

def audit():
    cfg=read(ROOT/'config_frozen.json');checks=[dict(name='gate',passed=read(OUT/'gate0.json')['passed']),dict(name='manifest_hash',passed=sha(ROOT/'manifest.json.gz')==cfg['manifest_sha256'])]
    for key,val in cfg['core_hashes'].items():checks.append(dict(name='source_'+key,passed=sha(ROOT/key)==val))
    bad=[];n=0
    for p in (RUN/'confirmation').glob('*.json.gz'):
        r=read(p)
        if r['status']!='complete':continue
        n+=1
        for a in r['rows']:
            if a['method']=='v6_pilot_CS':continue
            expected=a['gamma_upper']+a['rho']<=a['epsilon']
            if a['certified']!=expected:bad.append(dict(job=r['job'],method=a['method']))
    checks.append(dict(name='all_saved_certification_inequalities',passed=not bad,datasets=n,failures=bad))
    oldbad=[p for p,h in read(RUN/'e1_source_hashes.json').items() if sha(p)!=h];checks.append(dict(name='old_frozen_inputs_unchanged',passed=not oldbad))
    from .pipeline import run_job,jobkey
    replays=[];manifest=read(ROOT/'manifest.json.gz');families=sorted({j['family'] for j in manifest});cases=(0,6,7,13,16,19,12,21)
    chosen=[next(j for j in manifest if j['family']==name and j['rep']==0 and j['case_index']==cases[i] and j['design']==('Joint' if i%2 else 'Separated')) for i,name in enumerate(families)]
    for j in chosen:
        p=RUN/'confirmation'/f'{jobkey(j)}.json.gz'
        if not p.exists():continue
        a=read(p);b=run_job(j,'confirmation');ok=len(a['rows'])==len(b['rows']) and len(a['terminal'])==len(b['terminal']) and all(x['certified']==y['certified'] and x['false_certification']==y['false_certification'] for x,y in zip(a['rows'],b['rows']))
        difference=max((float(np.max(abs(np.array(x['weights'])-np.array(y['weights'])))) for x,y in zip(a['terminal'],b['terminal']) if 'weights' in x and 'weights' in y),default=0.)
        replays.append(dict(job=j,passed=bool(ok and difference<=1e-7),maximum_terminal_weight_difference=difference))
    checks.append(dict(name='deterministic_replay',passed=all(x['passed'] for x in replays),records=replays));checks.append(dict(name='CPU',passed=used()<43200))
    from .validation import run as independent
    checks.append(dict(name='independent_numerical_audit',passed=independent()))
    checks.append(dict(name='post_hoc_formula_floor_consistency',passed=not read(OUT/'calibration_formula_floor.json')['violations']))
    write(OUT/'audit.json',dict(passed=all(x['passed'] for x in checks),checks=checks));print('AUDIT',all(x['passed'] for x in checks),flush=True)

def archive(filename='v7_records.tar.zst'):
    dest=RUN/'archives';dest.mkdir(exist_ok=True);files=[p for p in RUN.rglob('*') if p.is_file() and 'archives' not in p.relative_to(RUN).parts and 'metrics' not in p.relative_to(RUN).parts];manifest={p.relative_to(RUN).as_posix():sha(p) for p in files};write(dest/'manifest.json',manifest);path=dest/filename
    with path.open('wb') as raw:
      with zstd.ZstdCompressor(level=3).stream_writer(raw) as z:
       with tarfile.open(fileobj=z,mode='w|') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(RUN).as_posix(),recursive=False)
    count=0
    with path.open('rb') as raw:
      with zstd.ZstdDecompressor().stream_reader(raw) as z:
       with tarfile.open(fileobj=z,mode='r|') as tar:
        for member in tar:
            if member.isfile():
                if hashlib.file_digest(tar.extractfile(member),'sha256').hexdigest()!=manifest[member.name]:raise RuntimeError('archive hash mismatch')
                count+=1
    write(OUT/'archive_verification.json',dict(filename=filename,files=count,bytes=path.stat().st_size,sha256=sha(path),verified=count==len(files),excludes='archive itself and stage metrics; final CPU tail is public'))
def dispatch(stage,*args):globals()[stage](*args)
