import json
import numpy as np
from scipy.special import logsumexp
from core import solve_qp, MaskLikelihood
from ..scenario_inference import ScenarioSet,posterior_from_loglik
from ..risk_envelopes import RiskSpec,Envelope,tail_weights
from ..risk_center import solve_center,conic_reference
from .book import Book
from .common import dump, ROOT
from .model import stable_solve

def run():
    rng=np.random.default_rng(812991); qs=[]
    for _ in range(32):
        L=rng.normal(size=(6,6));qs.append(L@L.T+np.eye(6))
    sc=ScenarioSet(np.array(qs),np.zeros((32,1)),np.repeat(np.arange(4),8),np.full(32,-np.log(32)),upper=.4).prepare()
    post=posterior_from_loglik(sc,rng.normal(size=32));unit=1.;results={}
    for spec in [RiskSpec('MA','mean'),RiskSpec('B-Mean','mean',conditional=True),RiskSpec('tail'),
                 RiskSpec('ctail',conditional=True),RiskSpec('max','max'),RiskSpec('bmax','max',conditional=True),
                 RiskSpec('lp','max',penalty=.01),RiskSpec('soft','soft')]:
        a=solve_center(sc,post,spec,unit); ref=conic_reference(sc,Envelope(sc,post,spec,unit))[2]
        assert a.lower-1e-6<=ref<=a.upper+1e-6,(spec.name,a,ref)
        assert a.gap<3e-6 and a.residual<1e-8
        results[spec.name]=dict(gap=a.gap,reference=ref)
    ma=solve_center(sc,post,RiskSpec('MA','mean'),1.)
    bm=solve_center(sc,post,RiskSpec('BM','mean',conditional=True),1.)
    assert np.max(abs(ma.weights-bm.weights))<1e-7
    w=np.ones(6)/6;r=sc.costs(w)-sc.oracle_lower
    vals=[Envelope(sc,post,s,1).support(r)[0] for s in [RiskSpec('mean','mean'),RiskSpec('ctail',conditional=True),RiskSpec('tail'),RiskSpec('max','max')]]
    assert np.min(np.diff(vals))>=-1e-10
    assert abs(tail_weights(r,np.ones(32)/32,.25)@r-np.sort(r)[-8:].mean())<1e-10
    q=np.eye(2); q2=q.copy();q2[0,1]=q2[1,0]=.5
    y=np.array([[1.,np.nan],[np.nan,2.]])
    assert abs(MaskLikelihood(y)(q)-MaskLikelihood(y)(q2))<1e-12
    assert abs(logsumexp(np.log([.5,.5])+[MaskLikelihood(y)(q),MaskLikelihood(y)(q2)])-MaskLikelihood(y)(q))<1e-12
    b=Book(2,.001)
    a=b.step(np.ones(2),np.zeros(2),np.array([True,False]),np.ones(2,bool),np.zeros(2,bool),np.array([.5,.5]))
    assert abs(b.holdings[0]-.5)<1e-12 and b.holdings[1]==0 and abs(a['nav']-.9995)<1e-12
    a=b.step(np.array([1.1,1]),np.zeros(2),np.ones(2,bool),np.array([False,True]),np.array([True,False]),np.array([0.,1.]))
    assert abs(b.holdings[0]-.55)<1e-12 and a['unfilled']>0 and a['stale_weight']>0
    b=Book(2);b.step(np.ones(2),np.zeros(2),np.ones(2,bool),np.ones(2,bool),np.zeros(2,bool),np.array([.5,.5]))
    a=b.step(np.ones(2),np.array([.02,0]),np.ones(2,bool),np.ones(2,bool),np.zeros(2,bool))
    assert abs(a['nav']-1.01)<1e-12
    results['book_conservation_blocked_trades_dividends']=True
    results['mean_identity_tail_order_uniform_topk_mask_equivalence']=True
    p2=posterior_from_loglik(sc,np.linspace(-1000,0,32))
    spec=RiskSpec('lp-pruning','max',penalty=1.)
    a=stable_solve(sc,p2,spec,1.);env=Envelope(sc,p2,spec,1.)
    fullvalue=env.support(sc.costs(a.weights)-sc.oracle_lower)[0]
    assert abs(fullvalue-a.upper)<1e-7 and a.gap<1e-6
    assert np.max(abs(a.adversary.sum()-1))<1e-12
    results['exact_lp_dominance']=dict(gap=a.gap,removed=a.diagnostics['exact_dominated_atoms_removed'])
    spec=RiskSpec('conditional',conditional=True)
    a=solve_center(sc,post,spec,1.)
    assert np.max(abs(np.bincount(sc.groups,weights=a.adversary)-np.exp(post.log_alpha)))<1e-9
    duplicated=np.repeat(r,2);pmass=np.repeat(post.p/2,2)
    assert abs(tail_weights(duplicated,pmass,.5)@duplicated-tail_weights(r,post.p,.5)@r)<1e-10
    results['fixed_conditional_marginals_replication_invariance']=True
    dump(ROOT/'output/gates.json',results); print(json.dumps(results,indent=2))

if __name__=='__main__':run()
