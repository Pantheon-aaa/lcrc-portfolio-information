"""A narrowly scoped, sourced corporate-action correction; no fitted decisions change."""
import datetime as dt
import numpy as np
import pyarrow.dataset as ds
from .common import SOURCE,ROOT,PRIVATE,dump

ACTION=dict(old='600837',new='601211',ratio=.62,announcement='2025-03-14',effective='2025-03-17',
    last_old_open='2025-02-05',issuer_announcement='2025-028',
    source='https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=10783319&stockid=',
    exchange_source='https://www.sse.com.cn/disclosure/announcement/listing/stock/c/c_20250226_10773005.shtml',
    original_pdf='https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-03-14/601211_20250314_HTR8.pdf')

def prices():
    dataset=ds.dataset(SOURCE)
    dates=[dt.date.fromisoformat(ACTION['effective']),dt.date.fromisoformat(ACTION['last_old_open'])]
    f=ds.field('symbol').isin([ACTION['old'],ACTION['new']]) & ds.field('date').isin(dates)
    x=dataset.to_table(filter=f,columns=['symbol','date','open']).to_pandas()
    old=float(x[(x.symbol==ACTION['old'])&(x.date==dates[1])].open.iloc[0])
    new=float(x[(x.symbol==ACTION['new'])&(x.date==dates[0])].open.iloc[0])
    assert old>0 and new>0 and ACTION['announcement']<ACTION['effective']
    dump(PRIVATE/'evaluation/action_price_basis.json',dict(old_open=old,new_open=new,**ACTION))
    dump(ROOT/'output/corporate_action_sources.json',dict(**ACTION,
        accounting='Fractional shares as in base backtest. Convert before the first tradable opening, consolidate successor position, no conversion trade fee.',
        scope='One identified absorption merger; not a complete corporate-action database',changes_target_weights=False))
    return old,new

def convert(book,old_index,new_index,old_open,new_open,ratio_new,dy_old,dy_new,stale_new):
    value=book.holdings[old_index]
    if value==0:return 0.
    assert ratio_new>0 and np.isfinite(ratio_new) and not stale_new
    assert abs(dy_old)<1e-12 and abs(dy_new)<1e-12,'Dividend entitlement requires a separate corporate-action implementation'
    converted=value/old_open*ACTION['ratio']*new_open
    # Book.step will apply the successor's opening ratio. Undo that ONE step here
    # so the new lot is not marked twice. self.nav stays at prior opening NAV.
    book.holdings[new_index]+=converted/ratio_new
    book.holdings[old_index]=0.;book.stale_run[old_index]=0
    return float(value)

def gate():
    from .book import Book
    b=Book(2);b.holdings[:]=.5;b.cash=0.;b.nav=1.
    convert(b,0,1,10.,18.,1.1,0.,0.,False)
    out=b.step(np.array([1.,1.1]),np.zeros(2),np.ones(2,bool),np.ones(2,bool),np.zeros(2,bool))
    assert abs(out['nav']-(.5/10*.62*18+.5*1.1))<1e-12
    assert b.holdings[0]==0 and out['fees']==0
    dump(ROOT/'output/corporate_action_gate.json',dict(passed=True,consolidated_nav=out['nav'],no_double_mark=True,no_conversion_fee=True))

if __name__=='__main__':gate();prices()
