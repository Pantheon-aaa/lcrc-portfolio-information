"""Column-projected ingestion; estimators only receive past return matrices."""
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import PRIVATE, SOURCE, dump, read, seed, period, ledger

FIELDS = ['return_total', 'amount', 'observed', 'trading_status', 'market_type',
          'total_adjustment_suspect', 'execution_price_ratio', 'execution_dividend_yield',
          'execution_buy_allowed', 'execution_sell_allowed', 'execution_mark_stale']
BOOL = {'observed','total_adjustment_suspect','execution_buy_allowed','execution_sell_allowed','execution_mark_stale'}

def prepare():
    tic=time.perf_counter(); cpu=time.process_time(); folder=PRIVATE/'cache';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metadata.json').exists(): return
    pf=pq.ParquetFile(SOURCE)
    idx=pf.read(columns=['date','symbol']).to_pandas()
    dates=np.sort(idx.date.unique()); symbols=np.sort(idx.symbol.unique())
    di=pd.Index(dates).get_indexer(idx.date); si=pd.Index(symbols).get_indexer(idx.symbol)
    shape=(len(dates),len(symbols)); del idx
    for field in FIELDS:
        dtype=np.bool_ if field in BOOL else np.int16 if field in ('trading_status','market_type') else np.float64
        a=np.lib.format.open_memmap(folder/(field+'.npy'),mode='w+',dtype=dtype,shape=shape)
        a[:]=True if field=='execution_mark_stale' else False if dtype==np.bool_ else -1 if dtype==np.int16 else np.nan
        offset=0
        for b in pf.iter_batches(batch_size=131072,columns=[field]):
            s=b.column(0).to_pandas(); n=len(s)
            v=s.fillna(True if field in ('total_adjustment_suspect','execution_mark_stale') else False if dtype==np.bool_ else -1 if dtype==np.int16 else np.nan).to_numpy(dtype=dtype)
            a[di[offset:offset+n],si[offset:offset+n]]=v;offset+=n
        a.flush(); del a
        print('cached',field,flush=True)
    meta=dict(dates=list(map(str,dates)),symbols=list(map(str,symbols)),shape=shape,
              source=str(SOURCE),bytes=SOURCE.stat().st_size,mtime_ns=SOURCE.stat().st_mtime_ns,
              rows=pf.metadata.num_rows,scope='private licensed data; not for redistribution')
    dump(folder/'metadata.json',meta)
    ledger('prepare',time.perf_counter()-tic,time.process_time()-cpu)

class Data:
    def __init__(self):
        self.meta=read(PRIVATE/'cache/metadata.json'); self.dates=np.array(self.meta['dates']); self.symbols=np.array(self.meta['symbols'])
        stat=SOURCE.stat()
        if self.meta['source']!=str(SOURCE) or self.meta['bytes']!=stat.st_size or self.meta['mtime_ns']!=stat.st_mtime_ns:
            raise RuntimeError('Input changed: use a new ASHARE_PRIVATE directory and rebuild the cache')
        self.a={x:np.load(PRIVATE/'cache'/f'{x}.npy',mmap_mode='r') for x in FIELDS}
    def past(self,t,ids):
        y=np.array(self.a['return_total'][t-251:t+1,ids],dtype=float)
        good=self.a['observed'][t-251:t+1,ids] & ~self.a['total_adjustment_suspect'][t-251:t+1,ids]
        y[~good]=np.nan
        return y

def select(data,t,cohort,panel,old=None,d=20):
    y=data.past(t,np.arange(len(data.symbols))); mask=np.isfinite(y); n=mask.sum(0)
    amounts=data.a['amount'][t-19:t+1]; ng=np.isfinite(amounts).sum(0)
    liquidity=np.nansum(amounts,axis=0)/np.maximum(ng,1)
    eligible=(mask[:189].sum(0)>=42)&(mask[189:].sum(0)>=42)&(ng>=15)
    eligible &= data.a['observed'][t] & (data.a['trading_status'][t]==1) & np.isin(data.a['market_type'][t],[1,4,16,32])
    old=[] if old is None else old
    def pick(allowed,k):
        ix=np.flatnonzero(eligible & allowed)
        ix=ix[np.argsort(-liquidity[ix],kind='stable')[:200]]
        ranking=sorted(ix,key=lambda j:(int(j) not in old,seed('universe',panel,data.symbols[j])))
        return ranking[:k]
    ids=pick(n>=240,d if cohort=='long' else d//2)
    if cohort!='long':ids+=pick((n>=126)&(n<240),d//2)
    if len(ids)!=d:return None,dict(reason='insufficient_eligible_pool',selected=len(ids))
    ids=list(map(int,ids)); counts=n[ids]; first=np.argmax(mask[:,ids],axis=0)
    return ids,dict(valid_counts=counts.tolist(),left_missing=first.tolist(),
        internal_missing=(252-first-counts).tolist(),liquidity_pool_limit=200,
        fit_counts=mask[:189,ids].sum(0).tolist(),score_counts=mask[189:,ids].sum(0).tolist())

def manifest():
    tic=time.perf_counter(); cpu=time.process_time(); data=Data(); out=[]
    starts=np.flatnonzero(data.dates>='2013-01-01')[0]
    # 21 opening-to-opening holding periods need the next opening after 21 intervals.
    schedule=range(starts,len(data.dates)-22,21)
    for d in (20,50):
      for cohort in ('long','limited'):
       for panel in range(3 if d==20 else 1):
        old=[]
        for number,t in enumerate(schedule):
            if d==50 and number%3:continue
            ids,diag=select(data,t,cohort,panel,old,d)
            if ids is not None:old=ids
            key=f'{d}_{cohort}_{panel}_{data.dates[t]}'
            out.append(dict(key=key,t=int(t),date=data.dates[t],period=period(data.dates[t]),d=d,
                            cohort=cohort,panel=panel,ids=ids,diagnostics=diag,
                            seed=seed('scenarios',key),evaluation_end=data.dates[t+22]))
    dump(PRIVATE/'manifest_all.json',out)
    valid=[r for r in out if r['ids'] is not None and r['d']==20 and r['panel']==0]
    # Ten timing windows fixed without loading any future returns or method results.
    timing=[]
    for y in (2013,2015,2017,2018,2019):
      for c in ('long','limited'):
        timing.append(next(r['key'] for r in valid if r['date'].startswith(str(y)) and r['cohort']==c))
    dump(PRIVATE/'timing_keys.json',timing)
    ledger('manifest',time.perf_counter()-tic,time.process_time()-cpu,dict(entries=len(out),skipped=sum(r['ids'] is None for r in out)))
    print('manifest',len(out),'skips',sum(r['ids'] is None for r in out),flush=True)

if __name__=='__main__':
    prepare();manifest()
