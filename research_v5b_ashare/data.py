"""Read-only source audit and private, date-indexed caches. No future-based filters."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import *

FIELDS=['return_total','amount','trading_status','market_type','open','close','prev_close_exchange','limit_up','limit_down','return_ex_dividend']
FLAGS=['observed','adjustment_suspect','total_adjustment_suspect']

def audit():
    cache=PRIVATE/'data';cache.mkdir(parents=True,exist_ok=True)
    pf=pq.ParquetFile(SOURCE)
    ids=pq.read_table(SOURCE,columns=['date','symbol']).to_pandas()
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(ids.date).unique()))
    symbols=sorted(ids.symbol.astype(str).unique());shape=(len(dates),len(symbols))
    di=dates.get_indexer(pd.to_datetime(ids.date));si=pd.Index(symbols).get_indexer(ids.symbol.astype(str))
    encoded=di.astype(np.int64)*len(symbols)+si
    duplicate_count=len(encoded)-len(np.unique(encoded))
    if duplicate_count:raise ValueError(f'duplicate identity/date rows: {duplicate_count}')
    arrays={k:np.lib.format.open_memmap(cache/f'{k}.npy',mode='w+',dtype='float64',shape=shape) for k in FIELDS+FLAGS}
    for a in arrays.values():a[:]=np.nan
    offset=0;counts={}
    for batch in pf.iter_batches(batch_size=200000,columns=FIELDS+FLAGS):
        frame=batch.to_pandas();n=len(frame);rr=di[offset:offset+n];cc=si[offset:offset+n]
        for k in arrays:arrays[k][rr,cc]=pd.to_numeric(frame[k],errors='coerce').to_numpy(dtype=float)
        offset+=n
    for k,a in arrays.items():a.flush();counts[k]={'finite':int(np.isfinite(a).sum()),'missing':int(np.isnan(a).sum())}
    r=arrays['return_total'];valid=np.isfinite(r)&(arrays['observed']==1)&(arrays['total_adjustment_suspect']!=1)
    valid&=(r>=-1)
    np.save(cache/'valid.npy',valid)
    ptr=read(MEMBERSHIP/'current.json');version=MEMBERSHIP/ptr['directory']
    membership_files=[MEMBERSHIP/'current.json',version/'daily_membership.parquet',version/'validation.json',version/'audit_result.json']
    for p in membership_files:
        if not p.resolve().is_relative_to(MEMBERSHIP.resolve()):raise ValueError('membership path traversal')
    hashes={p.relative_to(MEMBERSHIP).as_posix():sha(p) for p in membership_files}
    if hashes[ptr['directory']+'/daily_membership.parquet']!=ptr['daily_membership_sha256']:raise ValueError('membership checksum mismatch')
    if not read(version/'audit_result.json')['passed']:raise ValueError('membership audit failed')
    validation=read(version/'validation.json');vd=set(validation['validated_signal_dates'])
    monthends=pd.Series(dates,index=dates).groupby(dates.to_period('M')).max()
    missing=[d.strftime('%Y-%m-%d') for d in monthends if d.strftime('%Y-%m-%d') not in vd]
    ret=r[np.isfinite(r)];raw_prev=np.vstack([np.full(shape[1],np.nan),arrays['close'][:-1]])
    basic=arrays['close']/raw_prev-1
    diff=np.abs(basic-r);ordinary=valid&np.isfinite(diff)&(np.abs(arrays['prev_close_exchange']/raw_prev-1)<1e-6)
    report=dict(source_sha256=sha(SOURCE),source_bytes=SOURCE.stat().st_size,rows=pf.metadata.num_rows,
      shape=shape,date_start=str(dates[0].date()),date_end=str(dates[-1].date()),duplicate_identity_date=duplicate_count,
      fields=counts,return_quantiles=np.quantile(ret,[0,.001,.01,.5,.99,.999,1]),returns_below_minus_one=int((ret< -1).sum()),
      ordinary_close_return_abs_error_quantiles=np.quantile(diff[ordinary],[.5,.95,.99,1]),
      source_units='decimal returns; CNY price and amount; source field dictionary; empirical raw-close identity cross-check',
      membership_hashes=hashes,monthends_unverified=missing,
      test_unverified_monthends=[d for d in missing if '2020'<=d[:4]<='2025'],
      implementation_grade='research_only_company_actions_unverified',
      limitations=['No complete independently verified cash dividend/split/exchange register. Existing execution_dividend is inferred and is not used as factual actions.',
        'Historical symbol is available; permanent entity identifiers and historical announcement-time certification are unavailable.',
        'Membership is daily-union-validated historical queries, not fully announcement-certified point-in-time membership.',
        'Exchange historical board-lot rules are not independently established by these fields.'])
    write(cache/'index.json',dict(dates=[str(d.date()) for d in dates],symbols=symbols,membership_version=ptr['directory'],source_sha256=report['source_sha256']))
    write(OUT/'data_audit.json',report)
    print('AUDIT',shape,'unverified test monthends',report['test_unverified_monthends'],flush=True)

class Data:
    def __init__(self):
        meta=read(CACHE/'index.json');self.dates=meta['dates'];self.symbols=meta['symbols'];self.lookup={s:i for i,s in enumerate(self.symbols)}
        self.a={k:np.load(CACHE/f'{k}.npy',mmap_mode='r') for k in FIELDS+FLAGS+['valid']}
        version=MEMBERSHIP/meta['membership_version'];self.valid_dates=set(read(version/'validation.json')['validated_signal_dates'])
        mem=pd.read_parquet(version/'daily_membership.parquet',columns=['date','index','symbol'])
        mem['date']=pd.to_datetime(mem.date).dt.strftime('%Y-%m-%d');mem['symbol']=mem.symbol.astype(str)
        self.members={(d,k):[self.lookup[s] for s in g.symbol if s in self.lookup] for (d,k),g in mem.groupby(['date','index'],sort=False)}
    def returns(self,end,cols,length):
        start=max(0,end-length+1);x=np.array(self.a['return_total'][start:end+1][:,cols]);x[~self.a['valid'][start:end+1][:,cols]]=np.nan;return x
    def monthends(self):
        return [i for i,d in enumerate(self.dates) if i==len(self.dates)-1 or self.dates[i+1][:7]!=d[:7]]

def universe(data):
    result=[];prior={(domain,seed):[] for domain in ['csi300','csi500'] for seed in [11,29,47]}
    for t in data.monthends():
        day=data.dates[t]
        if day<'2014-01-01':continue
        valid=data.a['valid'];hist=valid[max(0,t-566):t+1].sum(0);recent=valid[max(0,t-62):t+1].sum(0)
        amount=data.a['amount'][max(0,t-19):t+1];count=np.isfinite(amount).sum(0);mean=np.nansum(amount,axis=0)/np.maximum(count,1)
        eligible=(hist>=540)&(recent>=60)&(count>=15)&(mean>0)&(data.a['observed'][t]==1)&(data.a['trading_status'][t]==1)
        for domain in ['csi300','csi500']:
            members=data.members.get((day,domain),[])
            selected=sorted([j for j in members if eligible[j]],key=lambda j:(-mean[j],data.symbols[j]))[:200]
            for seed in [11,29,47]:
                old=set(prior[domain,seed]);rank=sorted(selected,key=lambda j:(0 if j in old else 1,digest(seed,data.symbols[j])))
                status='ready' if day in data.valid_dates and len(rank)>=50 else 'unverified_membership' if day not in data.valid_dates else 'insufficient_universe'
                if status=='ready':prior[domain,seed]=rank[:50]
                result.append(dict(id=f'{day}_{domain}_{seed}',t=t,date=day,domain=domain,seed=seed,phase=phase(day),status=status,
                  members_count=len(members),eligible_count=int(sum(eligible[j] for j in members)),liquidity_top_count=len(selected),ranking=rank,
                  panel_rule='common ordering with previous N=50 membership retention; all scales share prefixes',
                  st_filter='trading_status_equals_1',quarterly=day[5:7] in ['03','06','09','12']))
    write(PRIVATE/'panels.json',result)
    write(OUT/'panel_coverage.json',[{k:v for k,v in x.items() if k!='ranking'} for x in result])
    print('PANELS',len(result),'ready',sum(x['status']=='ready' for x in result),flush=True)
