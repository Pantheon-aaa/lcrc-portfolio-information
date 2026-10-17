import inspect,itertools
from dataclasses import fields
import numpy as np
from scipy.special import logsumexp
from core import feasible
from research_v4a.mean_budget import Family,MeanBudget
from research_v4a.price_oracle import Geometry as PriceGeometry,PriceCache
from research_v6.certificates import Geometry
from .common import *
from .radius import *
from .models import observations
from .decision import DecisionInput,solve

def run():
    checks=[]
    def check(name,passed,**details):
        checks.append(dict(name=name,passed=bool(passed),**details))
        write(OUT/'gate0.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
        if not passed:raise AssertionError(name)
    rng=np.random.default_rng(seed('gate'));d=6;cap=.3;P=projection(d);Q=np.diag(np.arange(1,d+1));Q2=Q+2*np.ones((d,d))
    fam=Family(np.array([Q,Q2]),cap);check('blind_optimizer',np.max(abs(fam.oracle[0]-fam.oracle[1]))<1e-8)
    W=np.array([feasible(rng.random(d),np.full(d,cap)) for _ in range(100)])
    check('blind_regret',max(abs(fam.regret_bounds(w)[0][0]-fam.regret_bounds(w)[0][1]) for w in W)<1e-9)
    check('blind_decision_bound_tiny',envelope(2*np.ones((d,d)),cap)['decision']<1e-9)
    for k in range(12):
        A=rng.normal(size=(d,d));D=(A+A.T)/2;h=.5*np.einsum('ni,ij,nj->n',W,D,W);b=envelope(D,cap)
        check(f'oscillation_bound_{k}',h.max()-h.min()<=b['upper'])
    D=np.outer(np.array([1.,-1.,0,0,0,0]),np.array([1.,-1.,0,0,0,0]));f=Family(np.array([np.eye(d),np.eye(d)+D]),cap)
    check('tangent_perturbation_can_leave_optimizer',np.max(abs(f.oracle[0]-f.oracle[1]))<1e-8)
    alpha=np.array([.4,.6]);fm=Family(np.array([Q,Q2,alpha[0]*Q+alpha[1]*Q2]),cap)
    check('convex_hull_regret',all(fm.regret_bounds(w)[0][2]<=alpha@fm.regret_bounds(w)[1][:2]+1e-10 for w in W))
    X=observations(Q,2048,'Joint',seed('gate','X'));cal=calibrate(X,'Joint')
    check('gaussian_interval_contains_fixed_example',cal.encloses(Q))
    Y=observations(Q,256,'Separated',seed('gate','Y'));old=Y.copy();L,U=matrix_intervals(Y)
    check('unobserved_cross_block_not_zero',np.all(U[:3,3:]>0)&np.all(L[:3,3:]<0))
    check('calibration_does_not_mutate',np.array_equal(old,Y,equal_nan=True))
    check('no_truth_radius_interface',set(inspect.signature(calibrate).parameters)=={'X','design','delta','looks'})
    check('no_truth_decision_interface',set(x.name for x in fields(DecisionInput))=={'Q','p','log_p','upper','epsilon'})
    ff=Family(np.array([Q,np.diag(np.arange(d,0,-1))]),cap);p=np.ones(2)/2;g=Geometry(ff).gamma(3);eps=g['upper']*1.25
    a=solve(DecisionInput(ff.Q,p,np.log(p),cap,eps),betas=(0,.1),family=ff)
    ma=a[0]['weights'];check('zero_budget_equals_MA',all(np.array_equal(z['weights'],ma) for z in a if z['beta']==0))
    check('MA_saturation',all(np.array_equal(z['weights'],ma) for z in a))
    geo=PriceGeometry(ff,eps);pc=PriceCache(MeanBudget(ff,p),geo)
    rejected=False
    try:pc.validate(ff,np.array([.4,.6]),eps)
    except ValueError:rejected=True
    check('cross_posterior_cache_rejected',rejected)
    rw=g['rational_weights'];check('geometry_witness_bounds',g['upper']>=g['lower'] and g['primal_residual']==0)
    bad=feasible(np.r_[1.,np.zeros(d-1)],np.full(d,cap));check('existence_not_arbitrary_portfolio',max(ff.regret_bounds(bad)[0])>eps)
    check('finite_look_union_budget',3*(2*d*d+2*d)*factors(16,d,.05,3)[2]<=.05*(1+1e-14))
    # Exact common optimizer: keep structural zero distinct from numerical noise.
    same=Family(np.array([np.eye(d),2*np.eye(d)]),cap);check('structural_zero',Geometry(same).gamma(3)['upper']==0)
    import cvxpy as cp
    w=cp.Variable(d);t=cp.Variable();cons=[cp.sum(w)==1,w>=0,w<=cap]
    for j in range(2):cons.append(.5*cp.quad_form(w,ff.Q[j])-ff.V[j]<=t)
    prob=cp.Problem(cp.Minimize(t),cons);prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11)
    check('independent_gamma_reference',g['lower']-1e-9<=prob.value<=g['upper']+1e-9)
    for design in ('Joint','Separated'):
        Z=observations(Q,32,design,seed('gate',design));check(design+'_scalar_budget',np.isfinite(Z).sum()==32*d)
    from core import MaskLikelihood
    from .pipeline import block_ll
    from .models import prefix
    for design in ('Joint','Separated'):
        Z=observations(Q,128,design,seed('gate','likelihood',design))
        ll=block_ll(Z,ff.Q,design).sum(0)
        check(design+'_full_mask_likelihood',np.max(abs(ll-np.array([MaskLikelihood(Z)(q) for q in ff.Q])))<1e-9)
        original=prefix(Z,32,design).copy();changed=Z.copy()
        if design=='Joint':changed[32:]*=100
        else:changed[32:128]*=100;changed[160:]*=100
        check(design+'_future_prefix_invariance',np.array_equal(prefix(changed,32,design),original,equal_nan=True))
        a=calibrate(original,design);b=calibrate(prefix(changed,32,design),design)
        check(design+'_future_radius_invariance',a.compare(Q,cap)==b.compare(Q,cap))
    # Cross-block changes cannot alter a separated observation generated with
    # the same fixed seed; test the actual observation operator, not filled data.
    cross=Q.copy();cross[0,4]=cross[4,0]=.1
    check('separated_hidden_cross_covariance',np.array_equal(observations(Q,32,'Separated',7),observations(cross,32,'Separated',7),equal_nan=True))
    scaled=Family(ff.Q*100,cap)
    check('risk_scale_MA_weights',np.max(abs(scaled.mean(p)['w']-ff.mean(p)['w']))<1e-9)
    check('risk_scale_radius',abs(envelope((ff.Q[0]-ff.Q[1])*100,cap)['upper']/envelope(ff.Q[0]-ff.Q[1],cap)['upper']-100)<1e-10)
    check('joint_confidence_projection_is_not_truth_input',set(inspect.signature(radius_bounds).parameters)=={'cal','Q','p','cap'})
    order=np.array([5,2,4,0,3,1]);perm=Family(ff.Q[:,order][:,:,order],cap)
    check('permutation_MA_weights',np.max(abs(perm.mean(p)['w']-ff.mean(p)['w'][order]))<1e-8)
    dup=Family(np.repeat(ff.Q,2,axis=0),cap)
    check('replication_split_mass_MA',np.max(abs(dup.mean(np.ones(4)/4)['w']-ff.mean(p)['w']))<1e-8)
    check('old_source_read_only',not any('ashare' in x for x in inspect.signature(solve).parameters))
    print('GATE',len(checks),'passed',flush=True)
