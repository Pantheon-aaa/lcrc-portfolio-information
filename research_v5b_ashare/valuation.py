"""Evaluation-only marks. Missing-as-stale is an explicit research assumption."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import *

@dataclass(frozen=True)
class LabelPolicy:
    max_stale:int=63
    consistency_tolerance:float=1e-5

def build(data):
    expected=read(CACHE/'index.json')['source_sha256']
    if sha(SOURCE)!=expected:raise ValueError('source cache is stale')
    dest=PRIVATE/'valuation';dest.mkdir(exist_ok=True);shape=data.a['return_total'].shape
    adjusted=np.lib.format.open_memmap(dest/'adjusted.npy',mode='w+',dtype='float64',shape=shape);adjusted[:]=np.nan
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=200000,columns=['date','symbol','close_adj_total']):
        df=batch.to_pandas();rr=pd.Index(data.dates).get_indexer(pd.to_datetime(df.date).dt.strftime('%Y-%m-%d'));cc=pd.Index(data.symbols).get_indexer(df.symbol.astype(str))
        adjusted[rr,cc]=df.close_adj_total.to_numpy()
    adjusted.flush()
    age=np.lib.format.open_memmap(dest/'age.npy',mode='w+',dtype='int32',shape=shape)
    marks=np.lib.format.open_memmap(dest/'marks.npy',mode='w+',dtype='float64',shape=shape)
    ret=np.lib.format.open_memmap(dest/'returns.npy',mode='w+',dtype='float64',shape=shape)
    mismatch=np.lib.format.open_memmap(dest/'mismatch.npy',mode='w+',dtype='bool',shape=shape)
    exceptions=[];comparison=0
    for j in range(shape[1]):
        raw=data.a['return_total'][:,j];last=-1;level=np.nan;lastprice=np.nan
        for i in range(shape[0]):
            observed=np.isfinite(raw[i]) and data.a['observed'][i,j]==1
            mismatch[i,j]=False
            if observed:
                error=abs(adjusted[i,j]/lastprice-1-raw[i]) if last>=0 and lastprice>0 else 0.
                comparison+=int(last>=0)
                mismatch[i,j]=error>1e-5 or not data.a['valid'][i,j]
                if mismatch[i,j]:exceptions.append(dict(date=data.dates[i],symbol=data.symbols[j],error=error))
                level=1. if last<0 else level*(1+raw[i]);ret[i,j]=raw[i] if last>=0 else np.nan
                age[i,j]=0;last=i;lastprice=adjusted[i,j]
            else:age[i,j]=i-last if last>=0 else 2**30;ret[i,j]=0. if last>=0 else np.nan
            marks[i,j]=level
    for a in [age,marks,ret,mismatch]:a.flush()
    write(dest/'exceptions.json',exceptions)
    write(OUT/'valuation_audit.json',dict(source_sha256=expected,comparisons=comparison,mismatch_records=len(exceptions),max_stale_main=63,
       sensitivities=[20,126],assumption='carry last total-return valuation during missing records, recognize supplier cumulative return on resumption; not certified suspension identification',
       training_cache_read_only=True,source_unchanged=True))

class Valuation:
    def __init__(self):
        self.a={k:np.load(PRIVATE/'valuation'/f'{k}.npy',mmap_mode='r') for k in ['age','marks','returns','mismatch']}
    def window(self,t,cols,H=63,policy=LabelPolicy()):
        sl=np.ix_(np.arange(t+2,t+H+2),cols)
        x=np.array(self.a['returns'][sl]);ages=self.a['age'][sl];bad=self.a['mismatch'][sl]
        available=np.isfinite(x)&(ages<=policy.max_stale)&~bad
        return x,dict(complete=bool(available.all()),invalid_count=int((~available).sum()),approximate_fraction=float((ages>0).mean()),
                      max_stale_age=int(ages.max()),mismatch_count=int(bad.sum()),stale_limit=policy.max_stale)
