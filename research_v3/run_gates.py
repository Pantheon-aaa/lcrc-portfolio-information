"""Independent numerical and information-isolation gates."""
from itertools import combinations
import inspect
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta
from core import MaskLikelihood
from .common import OUT,write,timed_stage,seed
from .compat_models import Family,e1,embedded,multidirection,simulate,draw_loglik,kl_matrix
from .compatible_qcqp import Compatibility,solve,cvx_reference
from .dcl_master import DCL,enumerate_exact
from .evidence_gate import prepare,approve,scalar_decisions,two_model_action

def run(budget):
    checks={};all_intervals=[]
    for t in (.005,.01,.02):
        f=e1(t);c=Compatibility(f);p=np.ones(3)/3;epsilon=7*t*t/16
        for n in (32,256):
            Y=simulate(f,1,0,n,np.random.default_rng(seed('gate-equivalence',t,n)));ll=f.likelihood(Y)
            assert np.ptp(ll)<1e-9 and np.max(abs(f.posterior(ll)-p))<1e-10
        vals=[]
        for size in (1,2,3):
            for ids in combinations(range(3),size):
                r=c.gamma(ids);v,w=cvx_reference(f,ids)
                assert r.lower-5e-10<=v<=r.upper+5e-10
                assert r.gap<1e-8 and r.gap>=-1e-10
                vals.append(dict(indices=ids,lower=r.lower,upper=r.upper,reference=v))
        a=DCL(c,epsilon).solve(p);enumerated,upper=enumerate_exact(c,p,epsilon)
        assert abs(a['mass_lower']-2/3)<1e-9 and abs(enumerated[0]-a['mass_lower'])<1e-9
        ma=f.mean(p)['w'];assert (f.regret_bounds(ma)[0]>epsilon).all()
        assert p@f.regret_bounds(ma)[0]<=p@f.regret_bounds(a['w'])[1]
        for pair in combinations(range(3),2):assert c.gamma(pair).upper<epsilon
        assert c.gamma(range(3)).lower>epsilon
        hinge=solve(f,p,'hinge',epsilon=epsilon);gamma=epsilon/4
        loss=f.regret_bounds(hinge.w)[1]
        assert p[loss>epsilon+gamma].sum() <= p@np.maximum(loss-epsilon,0)/gamma+1e-12
        all_intervals.append(dict(t=t,epsilon=epsilon,subsets=vals,mass=a['mass_lower'],ma_regrets=f.regret_bounds(ma)))
    checks['E1_and_enumeration']=all_intervals
    f=e1(.02);eps=7*.02**2/16;p=np.array([.2,.35,.45]);c=Compatibility(f)
    a=DCL(c,eps).solve(p);f2=Family(np.r_[f.Q,f.Q[:1]]);p2=np.array([.1,.35,.45,.1])
    a2=DCL(Compatibility(f2),eps).solve(p2)
    assert abs(a['mass_lower']-a2['mass_lower'])<1e-9
    ta=solve(f,p,'tail');tb=solve(f2,p2,'tail');assert abs(ta.upper-tb.upper)<1e-8
    checks['duplication']={'mass':a['mass_lower'],'duplicated_mass':a2['mass_lower'],'tail_difference':ta.upper-tb.upper}
    # Classifying a threshold inside a rigorous interval must not silently relax epsilon.
    c=Compatibility(e1(.01));r=c.gamma(range(3));state,_=c.classify(range(3),(r.lower+r.upper)/2)
    assert state=='indeterminate';checks['boundary_state']=state
    embedded_checks=[]
    for d in (10,20):
        for t in (.1,.03,.01):
            f=embedded(d,t);k=d//2;x=t/(3+2*t);w=np.r_[np.full(k,(1-x)/k),np.full(k,x/k)]
            assert np.linalg.norm(f.oracle[0]-w)<1e-6
            C=np.array([[[1,1-t],[1-t,4]],[[1,1.7],[1.7,4]]]);ff=Family(C)
            assert abs(kl_matrix(f)[0,1]-kl_matrix(ff)[0,1])<1e-10
            rng=np.random.default_rng(seed('full-embedded',d,t));Y=simulate(f,0,30,0,rng)
            reduced=np.c_[Y[:,:k].mean(1),Y[:,k:].mean(1)]
            ll=f.likelihood(Y);rl=ff.likelihood(reduced)
            assert abs((ll[0]-ll[1])-(rl[0]-rl[1]))<1e-9
            Ym=np.full((64,d),np.nan);Ym[:32,:k]=rng.normal(size=(32,k));Ym[32:,k:]=rng.normal(size=(32,k))
            assert abs(np.diff(f.likelihood(Ym))[0])<1e-9
            for pg in (.001,.2,.5,.99,.9999):
                answers,loss=scalar_decisions(pg,t,np.log([pg,1-pg]))
                for name,mode in [('CVaR','tail'),('Minimax','max')]:
                    result=solve(f,np.array([pg,1-pg]),mode)
                    assert abs(np.sum(result.w[k:])-answers[name])<2e-5
            embedded_checks.append(dict(d=d,t=t,KL=kl_matrix(f)[0,1],full_reduced_ll_difference=float(np.diff(ll-rl)[0])))
    checks['embedded_exact_reduction']=embedded_checks
    f=multidirection(10);c=Compatibility(f);epsilon=c.gamma(range(f.M)).upper*.5
    rng=np.random.default_rng(seed('isolation'));pilot=draw_loglik(f,2,16,32,rng)
    library=prepare(f,pilot,c,epsilon);snapshot=library.weights.copy()
    for j in (0,1):
        ll=draw_loglik(f,2,30,0,np.random.default_rng(seed('gate-isolation',j)))
        ans=approve(f,library,ll);assert np.array_equal(snapshot,library.weights)
        assert abs(ans['w'].sum()-1)<1e-10 and ans['w'].min()>-1e-10 and ans['w'].max()<=.2+1e-10
        assert ans['candidate_budget']*len(library.weights)<=.005+1e-14
        # If every approved candidate is safe in model j, any convex mixture is safe there.
        for truth in range(f.M):
            approved=library.weights[ans['approved']]
            if not len(approved) or all(f.costs(w)[truth]<=f.costs(library.baseline)[truth] for w in approved):
                assert f.costs(ans['w'])[truth]<=f.costs(library.baseline)[truth]+1e-10
    for fn in (prepare,approve,two_model_action):assert 'truth' not in inspect.signature(fn).parameters
    checks['pilot_gate_isolation']={'candidate_count':len(library.weights),'geometry':library.generation['geometry']}
    # Rare FP check is diagnostic: a wide binomial CI is not a failed mathematical theorem.
    n=10000;t=.01;C=Family(np.array([[[1,1-t],[1-t,4]],[[1,1.7],[1.7,4]]]))
    m=int(np.ceil(np.log(1/(.05*t))/kl_matrix(C)[0,1]));count=0
    for rep in range(n):
        if rep%1000==0:budget.check()
        ll=draw_loglik(C,1,m,0,np.random.default_rng(seed('gate-fp',rep)))
        count+=two_model_action(ll,t)[1]
    upper=float(beta.ppf(.95,count+1,n-count)) if count<n else 1.
    lower=float(beta.ppf(.05,count,n-count+1)) if count else 0.
    assert lower<=.05*t
    checks['B1_false_positive']={'n':n,'errors':count,'one_sided_95_upper':upper,'one_sided_95_lower':lower,'target':.05*t,'m':m}
    write(OUT/'gate0.json',{'passed':True,'checks':checks});print('Gate 0 passed',flush=True)

if __name__=='__main__':timed_stage('gate_e1',2700,run)
