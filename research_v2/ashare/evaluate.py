"""Continuous execution evaluator and calendar-clustered comparisons."""
import time
import numpy as np
import pandas as pd
from .common import PRIVATE,ROOT,read,dump,period,ledger,seed
from .data import Data
from .book import Book

def load_records(stage):
    return [read(p) for p in sorted((PRIVATE/'records'/stage).glob('*.json'))]

def simulate(stage='test',actions=False):
    tic=time.perf_counter();cpu=time.process_time();data=Data(); records=load_records(stage); daily=[];terminal=[]
    label=stage+'_actions' if actions else stage
    if actions:
        from .actions import ACTION,prices,convert,gate
        gate();old_open,new_open=prices();old_id=int(np.flatnonzero(data.symbols==ACTION['old'])[0]);new_id=int(np.flatnonzero(data.symbols==ACTION['new'])[0])
        conversions=[]
    for cohort,panel in sorted(set((r['cohort'],r['panel']) for r in records)):
        rows=sorted([r for r in records if r['cohort']==cohort and r['panel']==panel],key=lambda r:r['t'])
        methods=sorted(set(a['method'] for r in rows for a in r.get('methods',[])))
        universe=sorted(set(i for r in rows for i in r['ids']))
        if actions and old_id in universe:universe=sorted(set(universe)|{new_id})
        inverse={s:i for i,s in enumerate(universe)}
        events={r['t']+1:r for r in rows}; start=rows[0]['t']+1;end=rows[-1]['t']+22
        books={(m,c):Book(len(universe),c/10000) for m in methods for c in (0,10,20)}
        last_selected=[];active_key=rows[0]['key']
        for day in range(start,end+1):
            r=events.get(day); targets={}
            if r is not None:
                active_key=r['key'];last_selected=[inverse[i] for i in r['ids']]
                for a in r.get('methods',[]):
                    if a.get('weights') is not None:
                        w=np.zeros(len(universe));w[last_selected]=a['weights'];targets[a['method']]=w
            x=data.a
            args=[x[f][day,universe] for f in ('execution_price_ratio','execution_dividend_yield',
                   'execution_buy_allowed','execution_sell_allowed','execution_mark_stale')]
            today=[]
            for (method,cost),book in books.items():
                if actions and data.dates[day]==ACTION['effective'] and old_id in inverse:
                    oi,ni=inverse[old_id],inverse[new_id]
                    value=convert(book,oi,ni,old_open,new_open,args[0][ni],np.nan_to_num(args[1][oi]),np.nan_to_num(args[1][ni]),args[4][ni])
                    if value:conversions.append(dict(cohort=cohort,panel=panel,method=method,cost_bps=cost,old_nav_fraction=value/book.nav))
                out=book.step(*args,target=targets.get(method))
                out.update(date=data.dates[day],cohort=cohort,panel=panel,method=method,cost_bps=cost,
                           period=period(data.dates[day]),window=active_key,interval_valid=day>start)
                today.append(out)
                at_test_end=data.dates[day].startswith('2025') and not data.dates[day+1].startswith('2025')
                if day==end or at_test_end:
                    unresolved=(book.stale_run>=21)&(book.holdings>1e-12)
                    uw=float(book.holdings[unresolved].sum()/book.nav)
                    terminal.append(dict(cohort=cohort,panel=panel,method=method,cost_bps=cost,date=data.dates[day],period=period(data.dates[day]),
                        unresolved_weight=uw,nav=book.nav,nav_haircut30=float(book.nav*(1-.3*uw)),nav_haircut100=float(book.nav*(1-uw)),
                        last_ret=out['ret'],stress_last_ret30=(1+out['ret'])*(1-.3*uw)-1,stress_last_ret100=(1+out['ret'])*(1-uw)-1,
                        stale_symbols=data.symbols[np.array(universe)[unresolved]].tolist()))
            # Same dates for every method in the reliable-mark sensitivity subset.
            reliable=all(v['stale_weight']<1e-10 for v in today if v['cost_bps']==0)
            for v in today:v['common_reliable']=reliable
            daily.extend(today)
    df=pd.DataFrame(daily);(PRIVATE/'evaluation').mkdir(exist_ok=True)
    df.to_parquet(PRIVATE/f'evaluation/daily_{label}.parquet',index=False)
    dump(PRIVATE/f'evaluation/terminal_{label}.json',terminal)
    if actions:
        dump(ROOT/'output/action_affected_portfolios.json',conversions)
        original=pd.read_parquet(PRIVATE/f'evaluation/daily_{stage}.parquet')
        before=df.date<ACTION['effective'];assert np.array_equal(df.loc[before,'nav'].values,original.loc[before,'nav'].values)
    ledger('evaluate_'+label+'_'+str(time.time_ns()),time.perf_counter()-tic,time.process_time()-cpu)
    return df,terminal

def summaries(df):
    rows=[]
    for (cohort,method,cost,phase),g in df.groupby(['cohort','method','cost_bps','period']):
        if phase not in ('test','extension'):continue
        per=[]
        for panel,s in g.groupby('panel'):
            ret=s.ret.to_numpy();riskret=s.loc[s.interval_valid,'ret'].to_numpy();nav=np.cumprod(1+ret);draw=1-nav/np.maximum.accumulate(np.r_[1.,nav])[1:]
            quarter=s[s.interval_valid].assign(quarter=pd.to_datetime(s[s.interval_valid].date).dt.to_period('Q').astype(str)).groupby('quarter').ret.apply(lambda r:np.sqrt(252*np.mean(r**2)))
            per.append(dict(annual_vol=float(np.std(riskret,ddof=1)*np.sqrt(252)),second_moment=float(.5*np.mean(riskret**2)),
                annual_return=float(np.prod(1+ret)**(252/len(ret))-1),mdd=float(draw.max()),
                worst_quarter_rms=float(quarter.max()),turnover=float(s.turnover.mean()*252),
                cash=float(s.cash.mean()),stale=float(s.stale_weight.mean()),stale_max=float(s.stale_weight.max()),
                concentration=float(s.concentration.mean()),unfilled=float(s.unfilled.mean()),
                reliable_fraction=float(s.common_reliable.mean()),days=len(ret)))
        row=dict(cohort=cohort,method=method,cost_bps=cost,period=phase,panels=len(per))
        row.update({k:float(np.mean([p[k] for p in per])) for k in per[0]});rows.append(row)
    return pd.DataFrame(rows)

def paired(df):
    results=[];rng=np.random.default_rng(492841);B=2000;block=63
    d=df[(df.cost_bps==0)&(df.period=='test')&df.interval_valid].copy();d['loss']=.5*d.ret**2
    for c,g in d.groupby('cohort'):
        # Aggregate simultaneous panels before calendar resampling, never treat as independent paths.
        table=g.groupby(['date','method']).loss.mean().unstack('method')
        n=len(table);starts=rng.integers(0,n,size=(B,int(np.ceil(n/block))))
        ix=((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]
        for method in ['B-Tail-0.25','B-Tail-0.5']:
          for baseline in ['MA','EM-Shrink']:
            a=table[method].to_numpy();b=table[baseline].to_numpy();diff=a-b
            boot=diff[ix].mean(1); delta=diff.mean(); centered=boot-delta
            p=float((1+np.sum(abs(centered)>=abs(delta)))/(B+1))
            lo,hi=np.quantile(boot,[.025,.975]);base=b.mean()
            results.append(dict(cohort=c,method=method,baseline=baseline,relative_difference=float(delta/base),
                 ci_low=float(lo/base),ci_high=float(hi/base),p=p,days=n,bootstrap=B,block=block))
    order=np.argsort([r['p'] for r in results]);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,results[i]['p']*(len(results)-rank)));results[i]['holm_p']=previous
    return pd.DataFrame(results)

def diagnostics(records):
    out=[]
    for r in records:
      for a in r.get('methods',[]):
        out.append(dict(key=r['key'],cohort=r['cohort'],period=r['period'],panel=r['panel'],method=a['method'],
          status=a['status'],gap=a.get('gap'),seconds=a.get('seconds'),ess=r.get('posterior',{}).get('ess'),
          max_mass=r.get('posterior',{}).get('max_mass'),objective=a.get('objective'),adversary_ess=a.get('adversary_ess'),
          adversary_max=a.get('adversary_max'),residual=a.get('residual')))
    return pd.DataFrame(out)

def publish_tables(stage='test',actions=False):
    if stage=='fifty':
        # Quarterly selected windows are isolated 21-day checks, not an invented 63-day strategy.
        records=load_records(stage);rows=[]
        manifests={r['key']:r for r in read(PRIVATE/'manifest_all.json')}
        for r in records:
          if r['period']=='test' and manifests[r['key']]['evaluation_end']>'2025-12-31':continue
          for a in r.get('methods',[]):
            if a.get('evaluation'):
                rows.append(dict(cohort=r['cohort'],method=a['method'],period=r['period'],**a['evaluation']))
        frame=pd.DataFrame(rows)
        frame.groupby(['cohort','method','period']).agg(windows=('second_moment','size'),
            second_moment=('second_moment','mean'),cash=('cash_mean','mean'),stale=('stale_mean','mean')).to_csv(ROOT/'output/summary_fifty.csv')
        diagnostics(records).to_csv(PRIVATE/'evaluation/diagnostics_fifty.csv',index=False)
        print('50-asset selected 21-day windows:',len(records));return
    label=stage+'_actions' if actions else stage
    path=PRIVATE/f'evaluation/daily_{label}.parquet'
    if not path.exists():df,terminal=simulate(stage,actions)
    else:df=pd.read_parquet(path);terminal=read(PRIVATE/f'evaluation/terminal_{label}.json')
    folder=ROOT/'output';folder.mkdir(exist_ok=True)
    s=summaries(df);s.to_csv(folder/f'summary_{label}.csv',index=False)
    reliable=df[df.common_reliable].copy();summaries(reliable).drop(columns=['annual_return','mdd']).to_csv(folder/f'reliable_{label}.csv',index=False)
    # Reliable subset return compounding is deliberately NOT interpreted as an executable track record.
    if stage=='test':paired(df).to_csv(folder/('paired_actions.csv' if actions else 'paired.csv'),index=False)
    diag=diagnostics(load_records(stage));diag.to_csv(PRIVATE/f'evaluation/diagnostics_{stage}.csv',index=False)
    ag=diag.groupby(['cohort','method']).agg(instances=('status','size'),open_gap=('status',lambda x:int((x=='gap_open').sum())),
        exceptions=('status',lambda x:int((x=='exception').sum())),ess_median=('ess','median'),ess_min=('ess','min'),
        max_mass_median=('max_mass','median'),gap_max=('gap','max'),seconds_mean=('seconds','mean'),residual_max=('residual','max'))
    ag.to_csv(folder/f'diagnostics_{stage}.csv')
    pd.DataFrame([{k:v for k,v in x.items() if k!='stale_symbols'} for x in terminal]).to_csv(folder/f'terminal_{label}.csv',index=False)
    stress=[]
    for row in terminal:
        if row['cost_bps']!=0:continue
        g=df[(df.cohort==row['cohort'])&(df.panel==row['panel'])&(df.method==row['method'])&(df.period==row['period'])&df.interval_valid]
        base=float(np.mean(g.ret**2))
        for haircut in (30,100):
            loss=.5*(base+(row[f'stress_last_ret{haircut}']**2-row['last_ret']**2)/len(g))
            stress.append(dict(cohort=row['cohort'],panel=row['panel'],method=row['method'],period=row['period'],
                haircut=haircut,second_moment=loss,unresolved_weight=row['unresolved_weight']))
    pd.DataFrame(stress).groupby(['cohort','method','period','haircut']).agg(second_moment=('second_moment','mean'),
        unresolved_weight=('unresolved_weight','mean')).to_csv(folder/f'exit_stress_{label}.csv')
    monthly=df[df.cost_bps==0].assign(month=pd.to_datetime(df[df.cost_bps==0].date).dt.to_period('M').astype(str))
    monthly.groupby(['cohort','method','month']).ret.agg(['count','mean',lambda x:float(np.sqrt(252*np.mean(x**2)))]).to_csv(folder/f'monthly_{label}.csv')
    print(s[(s.cost_bps==0)&(s.period=='test')][['cohort','method','annual_vol','second_moment','cash','stale']].to_string(index=False))

if __name__=='__main__':
    import sys
    publish_tables(sys.argv[1] if len(sys.argv)>1 else 'test',actions='--actions' in sys.argv)
