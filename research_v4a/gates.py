"""Mathematical identities, independent solvers and cache isolation regression tests."""
import inspect,hashlib
from itertools import combinations
import numpy as np
import mpmath as mp
from .common import OUT,write,run,seed
from .families import analytic,analytic_price,make,observations,equivalence,affinity
from .mean_budget import Family,MeanBudget
from .price_oracle import Geometry,PriceCache
from .budgeted_dcl import solve_budgeted_dcl,enumerate_frontier
from .baselines import cvx_price,original_a1

def execute(budget):
    checks={};records=[]
    for d in (3,6,10,20):
        for fraction in (.25,.75):
            budget.check();f,L0=analytic(d,fraction);t=f.metadata['t'];p=np.ones(d)/d;mb=MeanBudget(f,p)
            assert abs(mb.Lhi/L0-1)<1e-6
            assert np.max(abs(f.Q.mean(0)-np.eye(d)))<1e-13
            for ratio in (.5,.75,.875,.95,1.05):
                eps=ratio*L0;geo=Geometry(f,eps);pc=PriceCache(mb,geo)
                for a in analytic_price(d,t,eps):
                    m=a['m'];G=tuple(range(m));center=geo.gamma(G)
                    assert center.lower-1e-7*L0<=a['gamma']<=center.upper+1e-7*L0
                    if not np.isfinite(a['cost']):continue
                    r=pc.get(G,tight=True);ref=cvx_price(f,p,G,eps)
                    error=abs(ref['value']-a['cost'])/L0
                    assert error<1e-6,(d,ratio,m,error)
                    assert r.cost_lower<=a['cost']+1e-6*L0 and r.cost_upper>=a['cost']-1e-6*L0
                    records.append(dict(d=d,fraction=fraction,ratio=ratio,m=m,analytic_cost=a['cost'],lower=r.cost_lower,upper=r.cost_upper,cvx_cost=ref['value'],normalized_error=error))
    checks['analytic_price']=records
    # Nonuniform posterior and a non-symmetric geometry, including empty feasible success.
    refs=[]
    for kind in ('asymmetric','hidden_curvature'):
        f=make(kind,10,6,'gate',0);p=np.random.default_rng(seed('gate-p',kind)).dirichlet(np.ones(6));g=Geometry(f,0).gamma(range(6));eps=.5*(g.lower+g.upper)/2
        answers,pc=enumerate_frontier(f,p,eps,[0,.1,.25,np.inf]);geo=Geometry(f,eps)
        for beta,e in zip([0,.1,.25,np.inf],answers):
            a=solve_budgeted_dcl(f,p,eps,beta,geo)
            assert abs(a['optimal_mass_lower']-e['optimal_mass_lower'])<1e-8,(kind,beta,a,e)
            if a['primary_mass_gap']<1e-8 and a['secondary_price_gap'] is not None and np.isfinite(a['secondary_price_gap']):
                assert abs(a['mean_increment_upper']-e['mean_increment_upper'])<2e-6*pc.mb.Lhi,(kind,beta,'secondary')
            refs.append(dict(kind=kind,beta=beta,mass=a['optimal_mass_lower'],enumeration=e['optimal_mass_lower'],status=a['status']))
        for C,r in pc.cache.items():
            if C and np.isfinite(r.cost_upper):
                independent=cvx_price(f,p,C,eps)
                assert r.cost_lower-1e-6*pc.mb.Lhi<=independent['value']<=r.cost_upper+1e-6*pc.mb.Lhi
            for G,s in pc.cache.items():
                if set(C)<=set(G) and np.isfinite(s.cost_upper):assert r.cost_lower<=s.cost_upper+1e-8*pc.mb.Lhi
        try:pc.validate(f,np.ones(6)/6,eps)
        except ValueError:pass
        else:raise AssertionError('cross-posterior cache accepted')
        b=original_a1(f,p,eps);assert abs(b['optimal_mass_lower']-answers[-1]['optimal_mass_lower'])<1e-8
    checks['enumeration_and_price_cuts']=refs
    f,L0=analytic(3,.25);p=np.ones(3)/3;eps=.875*L0;a=solve_budgeted_dcl(f,p,eps,.00001)
    assert a['optimal_mass_lower']==0 and np.max(abs(a['weights']-p))<1e-9
    checks['empty_success']=a
    a=solve_budgeted_dcl(f,p,eps,.1)
    f2=Family(np.r_[f.Q,f.Q[:1]],f.upper);p2=np.array([1/6,1/3,1/3,1/6]);b=solve_budgeted_dcl(f2,p2,eps,.1)
    assert abs(a['optimal_mass_lower']-b['optimal_mass_lower'])<1e-8
    perm=np.array([2,0,1]);f3=Family(f.Q[:,perm][:,:,perm],f.upper);b=solve_budgeted_dcl(f3,p,eps,.1)
    assert abs(a['optimal_mass_lower']-b['optimal_mass_lower'])<1e-8
    checks['duplication_and_permutation']=True
    f=make('block',12,6,'gate',0);assert equivalence(f,'Separated')==[[0,1,2],[3,4,5]]
    for design in ('Separated','Joint'):
        Y,ll=observations(f,0,32,design,seed('gate-observation',design));assert np.isfinite(Y).sum()==32*12
        if design=='Separated':assert np.ptp(ll[:3])==0 and np.ptp(ll[3:])==0
        else:assert np.min(affinity(f,design)[np.triu_indices(6,1)])>0
    checks['mask_isolation']=True
    assert 'true_model' not in inspect.signature(solve_budgeted_dcl).parameters
    # 80-digit independent analytic QP oracle comparison (full covariance matrices).
    mp.mp.dps=80;f,L0=analytic(20,.25);Q=mp.matrix(f.Q[0].tolist());one=mp.matrix([1]*20);v=mp.lu_solve(Q,one);value=1/(2*sum(v))
    assert f.VL[0]<=float(value)<=f.VU[0]+1e-16
    checks['high_precision_QP']=str(value)
    write(OUT/'gate0.json',dict(passed=True,checks=checks));print(f'Gate 0 passed: {len(records)} analytic pricing comparisons',flush=True)

if __name__=='__main__':run('gate',execute)
