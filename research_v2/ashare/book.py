"""Arbitrary-weight, causal opening accounting. No missing-return zero imputation in fitting."""
import numpy as np

class Book:
    def __init__(self,n,cost=0.):
        self.holdings=np.zeros(n); self.cash=1.; self.cost=cost; self.nav=1.
        self.stale_run=np.zeros(n,dtype=int)

    def step(self,ratio,dividend,buy,sell,stale,target=None):
        oldnav=self.nav; h=self.holdings
        safe=np.where(np.isfinite(ratio)&(ratio>0),ratio,1.)
        dy=np.where(np.isfinite(dividend),dividend,0.)
        self.cash+=float(h@dy);h*=safe
        nav=self.cash+h.sum();fees=traded=unfilled=0.
        if target is not None:
            assert np.min(target)>=-1e-10 and abs(target.sum()-1)<1e-8
            goal=nav*target; excess=np.maximum(h-goal,0)
            sales=excess*np.asarray(sell);unfilled+=float((excess-sales).sum())
            h-=sales;self.cash+=sales.sum()*(1-self.cost);traded+=sales.sum();fees+=self.cost*sales.sum()
            deficit=np.maximum(goal-h,0); purchases=deficit*np.asarray(buy)
            unfilled+=float((deficit-purchases).sum())
            scale=min(1.,max(0.,self.cash)/(purchases.sum()*(1+self.cost))) if purchases.sum() else 0.
            purchases*=scale;h+=purchases;self.cash-=purchases.sum()*(1+self.cost)
            fees+=self.cost*purchases.sum();traded+=purchases.sum()
            assert abs(self.cash+h.sum()-(nav-fees))<1e-8*max(1.,nav)
        assert self.cash>=-1e-10 and np.min(h)>=-1e-10
        self.nav=float(self.cash+h.sum()); weight=h/self.nav
        bad=np.asarray(stale)|~np.isfinite(ratio)
        self.stale_run=np.where((h>1e-14)&bad,self.stale_run+1,0)
        return dict(nav=self.nav,ret=self.nav/oldnav-1,turnover=float(traded/(2*nav)),
            fees=float(fees),unfilled=float(unfilled/nav),stale_weight=float(weight@bad),
            stale_run_max=int(self.stale_run.max()),cash=float(self.cash/self.nav),
            concentration=float(weight@weight),max_weight=float(weight.max()))

def isolated_risk(data,t,ids,w):
    # Fresh account is only a validation loss/diagnostic, not the main continuous ledger.
    book=Book(len(ids)); n=len(data.symbols)
    rows=[]
    for day in range(t+1,t+23):
        a=data.a
        r=book.step(a['execution_price_ratio'][day,ids],a['execution_dividend_yield'][day,ids],
            a['execution_buy_allowed'][day,ids],a['execution_sell_allowed'][day,ids],
            a['execution_mark_stale'][day,ids],np.asarray(w) if day==t+1 else None)
        if day>t+1:rows.append(r)
    returns=np.array([r['ret'] for r in rows])
    return dict(second_moment=float(.5*np.mean(returns**2)),stale_mean=float(np.mean([r['stale_weight'] for r in rows])),
                cash_mean=float(np.mean([r['cash'] for r in rows])))
