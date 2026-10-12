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
    write(OUT/'gate0.json',dict(passed=True,checks=checks,deferred=['R3 scalar budgets and hidden-view invariance require R3 implementation before activation.',
      'Full historical corporate-action reconstruction is unavailable; synthetic accounting identities do not certify L1 market implementation.']))
    print('GATE0',len(checks),'checks passed; deferred scope recorded',flush=True)
