from time import perf_counter
import numpy as np
from research_v3.dcl_master import DCL
from research_v3.compatible_qcqp import Compatibility
from .mean_budget import MeanBudget
from .price_oracle import Geometry,PriceCache,conic
from .budgeted_dcl import result

def budget_greedy(f,p,epsilon,beta,geometry=None):
    start=perf_counter();deadline=start+30;geo=geometry or Geometry(f,epsilon);pc=PriceCache(MeanBudget(f,p),geo);mb=pc.mb;G=[];r=pc.get(())
    if beta>0:
        for j in sorted(range(f.M),key=lambda j:(-p[j],j)):
            try:z=pc.get(G+[j],deadline)
            except TimeoutError:break
            if z.epsilon_feasible and z.cost_upper<=mb.bounds(beta)[0]:G.append(j);r=z
    w=r.witness_weights;mass=float(p[f.regret_bounds(w)[1]<=epsilon].sum())
    return result(mb,epsilon,beta,w,mass,1.,None,'greedy',start,geo,pc,selected=G)

def budget_surrogate(f,p,epsilon,beta,mode='hinge',rho=.5):
    start=perf_counter();geo=Geometry(f,epsilon);mb=MeanBudget(f,p);pc=PriceCache(mb,geo);bl,bu=mb.bounds(beta);status='surrogate'
    if beta==0 or (mb.Llo<=0 and np.isfinite(beta)):
        w=mb.w;low=up=None;status='MA_endpoint' if beta==0 else 'budget_precision_limited'
    else:
        w,low,up,diag=conic(mb,np.arange(f.M),epsilon,mode,bl,rho,tight=True,deadline=start+30)
        if np.isfinite(bl) and mb.increment(w)[1]>bl:
            lower=0.;upper=1.;target=w.copy()
            for _ in range(45):
                a=(lower+upper)/2;z=(1-a)*mb.w+a*target
                if mb.increment(z)[1]<=bl:lower=a
                else:upper=a
            w=(1-lower)*mb.w+lower*target
        # Certify the returned, possibly repaired portfolio, not the pre-repair point.
        rh=f.regret_bounds(w)[1]
        if mode=='hinge':up=float(p@np.maximum(rh-epsilon,0))
        else:
            order=np.argsort(-rh);remaining=rho;up=0.
            for j in order:
                mass=min(float(p[j]),remaining);up+=mass*rh[j]/rho;remaining-=mass
                if remaining<=0:break
    mass=float(p[f.regret_bounds(w)[1]<=epsilon].sum())
    return result(mb,epsilon,beta,w,mass,1.,None,status,start,geo,pc,surrogate_objective_lower=low,surrogate_objective_upper=up)

def original_a1(f,p,epsilon):
    # Historical objective and historical subset tie break; same current family and evaluator.
    start=perf_counter();a=DCL(Compatibility(f),epsilon).solve(p);mb=MeanBudget(f,p);geo=Geometry(f,epsilon);pc=PriceCache(mb,geo)
    return result(mb,epsilon,np.inf,a['w'],a['mass_lower'],a['mass_upper'],None,a['status'],start,geo,pc)

def cvx_price(f,p,G,epsilon):
    import cvxpy as cp
    mb=MeanBudget(f,p);b=mb.w;s=np.sqrt(f.unit);x=cp.Variable(f.d);w=b+s*x
    g=mb.Q@b;g-=g.mean();obj=.5*cp.quad_form(x,cp.psd_wrap(mb.Q))+g@x/s
    cons=[cp.sum(x)==0,w>=0,w<=f.upper]
    lo,hi=f.regret_bounds(b)
    for j in G:
        grad=f.Q[j]@b;grad-=grad.mean()
        cons.append(.5*cp.quad_form(x,cp.psd_wrap(f.Q[j]))+grad@x/s+(lo[j]+hi[j])/2/f.unit<=epsilon/f.unit)
    pr=cp.Problem(cp.Minimize(obj),cons)
    try:pr.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=300)
    except cp.error.SolverError:pr.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-10,max_iter=300)
    return dict(status=pr.status,value=float(pr.value*f.unit),weights=None if x.value is None else np.array(w.value))
