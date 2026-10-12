"""Fractional total-return units at next close; not an exchange execution simulator."""
import numpy as np
from .common import *
from .data import Data
from .valuation import Valuation

class ProxyBook:
    def __init__(self,cost):self.cash=1e6;self.units={};self.pending=None;self.age=0;self.cost=cost;self.total_fees=0.
    def target(self,target):self.pending=target;self.age=0
    def step(self,prices,valid,trade):
        self.age+=1;held=[i for i,v in self.units.items() if v>0]
        if any(not valid[i] for i in held):return dict(nav=None,cash=self.cash,status='valuation_unresolved',fees=0.,traded=0.,exposure=None)
        nav=self.cash+sum(self.units[i]*prices[i] for i in held);fees=0.;traded=0.;changed=False
        if self.pending is not None and self.age<=5:
            ids=sorted(set(self.units)|set(self.pending));sold=set()
            for i in ids:
                if not trade[i] or not valid[i]:continue
                old=self.units.get(i,0.)*prices[i];amount=max(0.,old-nav*self.pending.get(i,0.))
                if amount>1e-9:
                    self.units[i]=max(0.,self.units.get(i,0.)-amount/prices[i]);self.cash+=amount*(1-self.cost);fees+=amount*self.cost;traded+=amount;sold.add(i);changed=True
            desired={i:max(0.,nav*w-self.units.get(i,0.)*prices[i]) for i,w in self.pending.items() if i not in sold and trade[i] and valid[i]}
            total=sum(desired.values());factor=min(1.,self.cash/(total*(1+self.cost))) if total else 0.
            for i,amount in desired.items():
                amount*=factor
                if amount>1e-9:self.units[i]=self.units.get(i,0.)+amount/prices[i];self.cash-=amount*(1+self.cost);fees+=amount*self.cost;traded+=amount;changed=True
            # Only unfilled targets are carried forward. A fully tradable target
            # must not trigger four additional daily rebalances as prices drift.
            blocked=any((not trade[i] or not valid[i]) and
                        abs(self.units.get(i,0.)*prices[i]-nav*self.pending.get(i,0.))>1e-9
                        for i in ids if np.isfinite(prices[i]))
            blocked=blocked or any((not valid[i]) and self.pending.get(i,0.)>0 for i in ids)
            if not blocked:self.pending=None
        self.total_fees+=fees;end=self.cash+sum(q*prices[i] for i,q in self.units.items() if q>0)
        return dict(nav=end,cash=self.cash,status='approximate',fees=fees,traded=traded,one_way_turnover=traded/(2*nav),exposure=1-self.cash/end,
                    executed_units=dict(self.units) if changed else None)

def run():
    data=Data();v=Valuation();start=next(i for i,d in enumerate(data.dates) if d>='2020-01-01');end=max(i for i,d in enumerate(data.dates) if d<='2025-12-31')
    manifest=[j for j in read(PRIVATE/'manifest.json') if j['stage']=='R2'];summary=[]
    for domain in ['csi300','csi500']:
        for seed in [11,29,47]:
            targets={};keys=set()
            for job in manifest:
                p=job['panel'];path=PRIVATE/'decisions'/f'{job["id"]}.json'
                if p['domain']!=domain or p['seed']!=seed or not path.exists():continue
                row=read(path)
                if row['status']!='complete':targets[p['t']+1]=None;continue
                day={}
                for cell in row['cells']:
                    if cell['tau_index']!=1 or cell['status']!='complete':continue
                    key=f'{cell["method"]}_b{cell["beta"]}';keys.add(key);day[key]=dict(zip(row['columns'],cell['decision']['evaluated_target']))
                targets[p['t']+1]=day
            for key in sorted(keys):
                for bps in [0,10,20,50]:
                    path=PRIVATE/'books'/f'{domain}_{seed}_{key}_{bps}.json'
                    if path.exists():summary.append(read(path)['summary']);continue
                    book=ProxyBook(bps/1e4);daily=[]
                    for t in range(start,end+1):
                        if t in targets:book.target(None if targets[t] is None else targets[t].get(key))
                        prices=v.a['marks'][t];valid=np.isfinite(prices)&(v.a['age'][t]<=63)&~v.a['mismatch'][t]
                        trade=data.a['valid'][t]&(data.a['observed'][t]==1)
                        rec=book.step(prices,valid,trade);rec['date']=data.dates[t];daily.append(rec)
                    nav=np.array([np.nan if r['nav'] is None else r['nav'] for r in daily]);complete=bool(np.isfinite(nav).all());returns=nav[1:]/nav[:-1]-1
                    metrics=dict(domain=domain,seed=seed,method_beta=key,cost_bps=bps,complete=complete,unresolved_days=int(np.isnan(nav).sum()),total_fees=book.total_fees,
                        turnover_sum=sum(r.get('one_way_turnover',0) for r in daily),mean_exposure=float(np.nanmean([r['exposure'] if r['exposure'] is not None else np.nan for r in daily])))
                    if complete:
                        rr=np.r_[nav[0]/1e6-1,returns];vol=float(np.std(rr,ddof=1));metrics.update(annual_return=float((nav[-1]/1e6)**(252/len(nav))-1),annual_vol=vol*np.sqrt(252),max_drawdown=float(np.min(nav/np.maximum.accumulate(np.r_[1e6,nav])[1:]-1)),final_nav=float(nav[-1]),sharpe_zero_rate=float(np.mean(rr)/vol*np.sqrt(252)) if vol else None,expected_shortfall95=float(-np.mean(rr[rr<=np.quantile(rr,.05)])))
                    write(path,dict(summary=metrics,daily=daily));summary.append(metrics)
            print('BOOKS',domain,seed,flush=True)
    write(OUT/'book_summary.json',summary)
