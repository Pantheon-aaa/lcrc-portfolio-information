"""Raw-share research ledger primitives; factual corporate actions must be supplied.

Unknown material corporate actions invalidate valuation rather than becoming zero returns.
No full-day high/low/volume enters the open execution decision.
"""
from dataclasses import dataclass,field
import numpy as np

@dataclass
class Book:
    cash:float=1_000_000.
    shares:dict=field(default_factory=dict)
    unresolved:list=field(default_factory=list)
    fees:float=0.
    trades:list=field(default_factory=list)
    pending:dict=field(default_factory=dict)
    age:int=0
    def action(self,symbol,kind,amount=0.,destination=None):
        held=self.shares.get(symbol,0.)
        if kind=='dividend':self.cash+=held*amount
        elif kind=='split':self.shares[symbol]=held*amount
        elif kind=='exchange':
            self.shares[destination]=self.shares.get(destination,0.)+held*amount;self.shares[symbol]=0.
        elif kind=='unknown' and held:self.unresolved.append(dict(symbol=symbol,reason='unreconstructed_material_action'))
    def value(self,prices):
        if self.unresolved:return None
        if any(q and (s not in prices or not np.isfinite(prices[s])) for s,q in self.shares.items()):return None
        return self.cash+sum(q*prices[s] for s,q in self.shares.items() if q)
    def set_target(self,target):self.pending=dict(target);self.age=0
    def execute(self,opens,buy_allowed,sell_allowed,lag_adv,lot,cost=.002):
        nav=self.value(opens)
        if nav is None:return dict(status='unresolved_valuation')
        if not self.pending or self.age>=5:return dict(status='no_pending_target')
        self.age+=1;ids=sorted(set(self.shares)|set(self.pending));sold=set();bought=sold_value=0.
        for s in ids:
            if not sell_allowed.get(s,False) or s not in opens:continue
            price=opens[s];desired=nav*self.pending.get(s,0)/price;held=self.shares.get(s,0)
            quantity=min(max(held-desired,0.),lag_adv.get(s,0.)*.01/price)
            if quantity<=0:continue
            # Research ledger permits fractional odd-lot liquidation; qualification stays research-only.
            self.shares[s]=held-quantity;value=quantity*price;fee=cost*value;self.cash+=value-fee;self.fees+=fee;sold.add(s);sold_value+=value
            self.trades.append(dict(side='sell',symbol=s,quantity=quantity,price=price,fee=fee))
        for s in ids:
            if s in sold or not buy_allowed.get(s,False) or s not in opens or s not in lot:continue
            price=opens[s];desired=nav*self.pending.get(s,0)/price;held=self.shares.get(s,0);step=lot[s]
            quantity=min(max(desired-held,0.),lag_adv.get(s,0.)*.01/price,self.cash/(price*(1+cost)))
            quantity=np.floor(quantity/step)*step
            if quantity<=0:continue
            value=quantity*price;fee=value*cost;self.cash-=value+fee;self.shares[s]=held+quantity;self.fees+=fee;bought+=value
            self.trades.append(dict(side='buy',symbol=s,quantity=float(quantity),price=price,fee=fee))
        return dict(status='research_execution',buy_value=bought,sell_value=sold_value,one_way_turnover=(bought+sold_value)/(2*nav),cash=self.cash)
