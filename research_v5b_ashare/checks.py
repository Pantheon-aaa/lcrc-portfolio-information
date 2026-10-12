"""Executable acceptance checks. Deferred research stages are not called passed."""
import inspect
import numpy as np
from dataclasses import fields
from scipy.special import logsumexp
from .common import *
from .risk import em_second,shrink,log_density,posterior,fit_models
from .decision import DecisionInput,Solver
from .evaluator import empirical_oracle
from .ledger import Book

def run():
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        if not condition:raise AssertionError(name)
    rng=np.random.default_rng(20261012);x=rng.normal(size=(567,6))*.02
    S,info=em_second(x);check('complete_second_moment',np.array_equal(S,x.T@x/len(x)))
    xm=x.copy();xm[::7,0]=np.nan;xm[::11,2:4]=np.nan;Sm,info=em_second(xm)
    check('missing_EM_PSD_not_zero_fill',np.linalg.eigvalsh(Sm).min()>0 and not np.allclose(Sm,np.nan_to_num(xm).T@np.nan_to_num(xm)/len(x),rtol=1e-5,atol=0),info)
    q=shrink(S,.3);obs=np.isfinite(xm[0]);z=xm[0,obs];qo=q[np.ix_(obs,obs)]
    ref=-.5*(len(z)*np.log(2*np.pi)+np.linalg.slogdet(qo)[1]+z@np.linalg.solve(qo,z))
    check('full_mask_likelihood',abs(log_density(q,xm[0])-ref)<1e-10)
    check('empty_mask_zero_evidence',log_density(q,np.full(6,np.nan))==0)
    Q,_=fit_models(x);u=float(np.ones(6)@Q[5]@np.ones(6)/72);p=np.ones(12)/12
    a=DecisionInput(Q,p,np.log(p),u,.1*u,.2,'synthetic-check');s=Solver(a);r=s.solve('A1B',0)
    check('beta_zero_equals_MA',np.array_equal(r['evaluated_target'],s.mb.w))
    b=DecisionInput(Q*100,p,np.log(p),u*100,.1*u*100,.2,'rescaled');r2=Solver(b).solve('A1B',.1);r1=s.solve('A1B',.1)
    check('return_unit_scaling',np.linalg.norm(r1['evaluated_target']-r2['evaluated_target'],1)<1e-7)
    rejected=False
    try:s.pc.validate(s.f,np.arange(1,13),s.eps)
    except ValueError:rejected=True
    check('cross_posterior_cache_rejected',rejected)
    identical=DecisionInput(np.repeat(Q[:1],12,axis=0),p,np.log(p),u,u,.2,'identical')
    si=Solver(identical);ri=si.solve('A1B',.1)
    check('identical_models_MA',np.linalg.norm(ri['evaluated_target']-si.mb.w,1)<1e-8)
    hinge=si.solve('Hinge',.1);check('zero_hinge_global_mean_tie',np.array_equal(hinge['evaluated_target'],si.mb.w))
    check('decision_interface_past_only',set(f.name for f in fields(DecisionInput))=={'Q','p','log_p','risk_unit','epsilon','upper','shared_cost_id'})
    same=[s.solve(m,.1) for m in ['MA','Greedy','Hinge','CVaR05']]
    check('same_method_input_hash',len(set(r['input_hash'] for r in same+[r1]))==1)
    o=empirical_oracle(Q[0],.2);check('independent_QP_interval',o['lower']<=s.f.VU[0]*u+1e-12 and o['upper']>=s.f.VL[0]*u-1e-12 and o['gap']/u<1e-6)
    old=x.copy();old[400:]*=999
    before,_=fit_models(x[:400]);after,_=fit_models(old[:400]);check('future_mutation_fit_invariance',np.array_equal(before,after))
    book=Book(cash=1000);book.set_target({'a':1.});book.execute({'a':10},{'a':True},{'a':True},{'a':1e8},{'a':1},cost=.002)
    check('ledger_cash_cost_conservation',abs(book.value({'a':10})+book.fees-1000)<1e-9 and book.cash>=0)
    held=book.shares['a'];book.action('a','dividend',.2);check('ledger_dividend',abs(book.cash-(1000-held*10*1.002+held*.2))<1e-9)
    book.action('a','split',2);check('ledger_split',book.shares['a']==held*2)
    book.action('a','exchange',.5,'b');check('ledger_exchange',book.shares['b']==held and book.shares['a']==0)
    book.set_target({'c':1});book.execute({'b':10,'c':10},{'c':False},{'b':False},{},{'c':1});check('pool_exit_blocked_holding_kept',book.shares['b']==held)
    book.action('b','unknown');check('unknown_action_is_unresolved',book.value({'b':10}) is None)
    from .data import Data
    from .valuation import Valuation
    from .risk import frontend
    from .pipeline import inp
    from .bridge import view_data
    from .proxy_book import ProxyBook
    data=Data();v=Valuation();changed=0;carried=0
    for j in range(len(data.symbols)):
        raw=data.a['return_total'][:,j];new=v.a['returns'][:,j];observed=np.isfinite(raw)&(v.a['age'][:,j]==0)&np.isfinite(new)
        changed+=int(np.count_nonzero(raw[observed]!=new[observed]));mask=(v.a['age'][:,j]>0)&np.isfinite(new)
        carried+=int(mask.sum())
        if mask.any():check_condition=bool((new[mask]==0).all());assert check_condition
    check('observed_returns_unchanged',changed==0,dict(carried_cells=carried))
    panel=next(p for p in read(PRIVATE/'panels.json') if p['date']=='2015-03-31' and p['seed']==11)
    class Cutoff:
        def __init__(self,mutation):self.mutation=mutation;self.calls=0
        def returns(self,end,cols,length):
            self.calls+=1;x=data.returns(end,cols,length)
            if end>panel['t']:
                x[max(0,len(x)-(end-panel['t'])):]*=self.mutation
            return x
    a=Cutoff(1);b=Cutoff(10000);f1=frontend(a,panel,N=8,cache=False);f2=frontend(b,panel,N=8,cache=False)
    z1=Solver(inp(f1,.1,.3));z2=Solver(inp(f2,.1,.3));r1=z1.solve('A1B',.1);r2=z2.solve('A1B',.1)
    check('future_mutation_frontend_and_target_invariance',np.array_equal(f1['Q'],f2['Q']) and np.array_equal(f1['scores'],f2['scores']) and np.array_equal(z1.inp.p,z2.inp.p) and np.array_equal(r1['evaluated_target'],r2['evaluated_target']) and a.calls>0)
    pairs=rng.normal(size=(32,2,8));aa=np.arange(4);bb=np.arange(4,8);sep=view_data(pairs,aa,bb,'Separated0');joint=view_data(pairs,aa,bb,'Joint0')
    check('R3_equal_scalar_budget',np.isfinite(sep).sum()==np.isfinite(joint).sum()==32*8)
    mutated=pairs.copy();mutated[:,0,bb]*=1000;mutated[:,1,aa]*=1000
    check('R3_hidden_coordinates_invariant',np.array_equal(sep,view_data(mutated,aa,bb,'Separated0'),equal_nan=True))
    book=ProxyBook(.002);prices=np.array([10.,20.]);valid=np.ones(2,dtype=bool);trade=valid.copy();book.target({0:.5,1:.5});one=book.step(prices,valid,trade)
    check('proxy_cash_and_initial_cost',abs(one['nav']+one['fees']-1e6)<1e-8 and book.cash>=-1e-8)
    units=dict(book.units);drift=book.step(prices*np.array([1.1,.9]),valid,trade)
    check('proxy_filled_target_drifts_without_rebalancing',book.pending is None and book.units==units and drift['traded']==0)
    before=dict(book.units);book.target({1:1.});trade[0]=False;two=book.step(prices,valid,trade)
    check('proxy_pool_exit_missing_trade_kept',book.units[0]==before[0])
    for _ in range(5):book.step(prices,valid,trade)
    before=dict(book.units);trade[:]=True;book.step(prices,valid,trade);check('proxy_pending_expires',book.units==before)
    valid[0]=False;check('proxy_unknown_value_unresolved',book.step(prices,valid,trade)['nav'] is None)
    valid[:]=True;check('proxy_value_recovers',book.step(prices,valid,trade)['nav'] is not None)
    write(OUT/'gate0.json',dict(passed=True,checks=checks,qualification='Research mark and fractional total-return ledger checks; no L1 execution claim.'))
    print('GATE0',len(checks),'checks passed under amended research assumptions',flush=True)
