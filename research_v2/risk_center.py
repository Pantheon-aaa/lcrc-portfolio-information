from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
from core import solve_qp, feasible
from .risk_envelopes import Envelope


@dataclass
class DecisionResultV2:
    weights: np.ndarray
    objective: float
    lower: float
    upper: float
    gap: float
    status: str
    seconds: float
    qp_calls: int
    residual: float
    adversary: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def aggregate_qp(sc,q):
    mass=q.sum()
    return solve_qp(np.einsum('m,mij->ij',q,sc.Q),q@sc.mu,{'upper':sc.upper},sc.kappa*mass,sc.previous)


def bounds(sc,env,w,q,additive=0.):
    vl=np.zeros(len(sc.Q)) if env.spec.absolute else sc.oracle_lower
    vu=np.zeros(len(sc.Q)) if env.spec.absolute else sc.oracle_upper
    U=env.support(sc.costs(w)-vl)[0]+env.support_error
    o=aggregate_qp(sc,q)
    L=o['value_lb']-q@vu+additive
    return float(L),float(U),o


def project_envelope(q,env):
    con=env.dual_constraints()
    if con is None: return q
    lo,hi,A,rhs=con; ans=np.zeros(len(q))
    near_feasible=(np.min(q-lo)>=-1e-14 and np.min(hi-q)>=-1e-14
                   and np.max(abs(A@q-rhs))<1e-12)
    if near_feasible:
        ans=np.clip(q,lo,hi)
        for row,mass in zip(A,rhs):
            ix=np.flatnonzero(row); rem=mass-ans[ix].sum()
            j=ix[np.argmax(hi[ix]-ans[ix] if rem>0 else ans[ix]-lo[ix])]
            ans[j]+=rem
        if env.mode=='budget' and env.evidence@ans>env.budget:
            base=env.budget_anchor; high=float(env.evidence@ans); low=float(env.evidence@base)
            mix=max(0.,min(1.,(env.budget-low)/(high-low)))
            ans=mix*ans+(1-mix)*base
        return ans
    for row,mass in zip(A,rhs):
        ix=np.flatnonzero(row)
        target=mass-lo[ix].sum()
        if target<=1e-16: ans[ix]=lo[ix]; continue
        v=q[ix]-lo[ix]; cap=hi[ix]-lo[ix]
        left=float(np.min(v-cap)-1); right=float(np.max(v)+1)
        for _ in range(60):
            mid=(left+right)/2
            if np.clip(v-mid,0,cap).sum()>target: left=mid
            else: right=mid
        ans[ix]=lo[ix]+np.clip(v-(left+right)/2,0,cap)
        rem=mass-ans[ix].sum(); j=ix[np.argmax(hi[ix]-ans[ix] if rem>0 else ans[ix]-lo[ix])]
        ans[j]+=rem
    if env.mode=='budget' and env.evidence@ans>env.budget:
        base=env.budget_anchor; high=float(env.evidence@ans); low=float(env.evidence@base)
        mix=max(0.,min(1.,(env.budget-low)/(high-low)))
        ans=mix*ans+(1-mix)*base
    return ans


def conic_reference(sc,env):
    """Independent normalized epigraph, including exponential-cone methods."""
    m,d,_=sc.Q.shape; spec=env.spec; u=env.unit
    w=cp.Variable(d); r=cp.Variable(m)
    V=np.zeros(m) if spec.absolute else (sc.oracle_lower+sc.oracle_upper)/2
    costs=cp.hstack([.5*cp.quad_form(w,cp.psd_wrap(Q))/u-mu@w/u-v/u for Q,mu,v in zip(sc.Q,sc.mu,V)])
    if sc.kappa: costs+=sc.kappa/u*cp.norm1(w-sc.previous)
    scenario_constraint=costs<=r
    # Bounded epigraph coordinates prevent essentially zero-probability atoms
    # from creating unbounded numerical slack, without altering their masses.
    cost_upper=.5*np.linalg.eigvalsh(sc.Q)[:,-1]*min(sc.upper,1)+np.max(abs(sc.mu),axis=1)
    if sc.kappa: cost_upper+=sc.kappa*(1+np.abs(sc.previous).sum())
    cons=[cp.sum(w)==1,w>=0,w<=sc.upper,scenario_constraint,
          r<=(cost_upper-V)/u+1e-8,
          r>=(sc.oracle_lower-V)/u-1e-8]
    vals=cp.multiply(env.scale,r)-env.offset/u; objective=0.
    for ix,mass in env.blocks():
        if mass==0: continue
        p=env.p[ix]/mass; x=vals[ix]; mode=env.mode
        if mode=='mean': term=p@x
        elif mode=='max': term=cp.max(x)
        elif mode=='soft':
            lp=env.lp[ix]; lp=lp-np.logaddexp.reduce(lp)
            term=spec.temperature*cp.log_sum_exp(lp+x/spec.temperature)
        elif mode=='lr':
            eta=cp.Variable(nonneg=True); nu=cp.Variable()
            term=nu+eta*(spec.radius-1)+p@cp.rel_entr(eta*np.ones(len(ix)),nu-x)
        elif mode=='budget':
            z=cp.Variable(); gamma=cp.Variable(nonneg=True)
            term=z+gamma*env.budget+(p/spec.rho)@cp.pos(x-z-gamma*env.evidence[ix])
        else:
            z=cp.Variable(); term=z+(p/env.block_rho(ix))@cp.pos(x-z)
        if spec.blend<1: term=(1-spec.blend)*(p@x)+spec.blend*term
        objective+=mass*term
    problem=cp.Problem(cp.Minimize(objective),cons)
    problem.solve(solver='CLARABEL',tol_gap_abs=2e-9,tol_gap_rel=2e-9,tol_feas=1e-10,max_iter=250,time_limit=60.)
    if w.value is None: raise RuntimeError(f'conic reference failed: {problem.status}')
    weights=feasible(np.asarray(w.value),np.full(d,sc.upper))
    q=np.maximum(np.asarray(scenario_constraint.dual_value),0.)
    if env.dual_constraints() is not None: q=project_envelope(q,env)
    return weights,q,float(problem.value*u),str(problem.status)


def solve_center(sc,post,spec,unit,reference=False,time_limit=60.):
    start=perf_counter(); sc.prepare(); env=Envelope(sc,post,spec,unit)
    m,d,_=sc.Q.shape; calls=0; info={'method':spec.name,'confidence_level':None,
        'target':'absolute-model-risk' if spec.absolute else 'model-regret',
        'probability_source':sc.source,'numerical_certificate':'float64 weak-duality, not interval arithmetic'}
    V=np.zeros(m) if spec.absolute else (sc.oracle_lower+sc.oracle_upper)/2
    if env.mode=='naive':
        starts=[np.ones(d)/d,aggregate_qp(sc,post.p)['w']]
        from dataclasses import replace
        for alt in (replace(spec,name='initializer-tail',kind='tail',conditional=False),
                    replace(spec,name='initializer-max',kind='max',conditional=True)):
            starts.append(solve_center(sc,post,alt,unit).weights)
        best=(np.inf,None); details=[]; nfev=0
        def obj(x):
            w=feasible(x,np.full(d,sc.upper))
            return env.support(sc.costs(w)-V)[0]/unit
        for w0 in starts:
            res=minimize(obj,w0,method='COBYLA',bounds=[(0,sc.upper)]*d,
                constraints=[{'type':'ineq','fun':lambda w:1e-8-abs(w.sum()-1)}],
                options={'maxiter':250,'tol':1e-6,'catol':1e-7,'rhobeg':.02})
            w=feasible(res.x,np.full(d,sc.upper)); val=obj(w)*unit; nfev+=res.nfev
            details.append({'value':val,'success':bool(res.success),'nfev':res.nfev,'message':str(res.message)})
            if val<best[0]: best=val,w
            if perf_counter()-start>time_limit: break
        val,q,_=env.support(sc.costs(best[1])-V)
        info.update(starts=details,function_evaluations=nfev,global_optimality=False)
        return DecisionResultV2(best[1],val,float('nan'),val,float('nan'),'heuristic',perf_counter()-start,0,
            abs(best[1].sum()-1),q,info)

    con=env.dual_constraints(); best={'U':np.inf,'L':-np.inf}; last={}
    def update(w,q,add=0.):
        nonlocal calls
        L,U,_=bounds(sc,env,w,q,add); calls+=1
        if U<best['U']: best.update(U=U,w=w.copy())
        if L>best['L']: best.update(L=L,q=q.copy())
    native_done=False
    if con is not None and not reference and m>=128 and sc.kappa==0 and env.mode!='mean':
        try:
            from .native_conic import solve as native_solve
            w,q,val,status=native_solve(sc,env,time_limit)
            update(w,q,-float(q@env.offset))
            info.update(native_conic_status=status,native_conic_value=val)
            native_done=best['U']-best['L']<=1e-6*unit
        except Exception as exc:info['native_conic_error']=repr(exc)
    if con is not None and not reference and not native_done:
        lo,hi,A,rhs=con
        def evaluate(q):
            nonlocal calls
            if 'x' in last and np.array_equal(last['x'],q): return last['ans']
            qr=project_envelope(q,env); o=aggregate_qp(sc,qr); calls+=1; w=o['w']
            loss=sc.costs(w)-V-env.offset
            dual=(o['value_lb']-qr@(np.zeros(m) if spec.absolute else sc.oracle_upper)-qr@env.offset)
            U=env.support(sc.costs(w)-(np.zeros(m) if spec.absolute else sc.oracle_lower))[0]+env.support_error
            if U<best['U']: best.update(U=U,w=w.copy())
            if dual>best['L']: best.update(L=dual,q=qr.copy())
            ans=(-float(qr@loss)/unit,-loss/unit)
            last.update(x=q.copy(),ans=ans)
            return ans
        initial=project_envelope(env.p.copy(),env); evaluate(initial)
        if np.max(hi-lo)>1e-14 and best['U']-best['L']>1e-6*unit:
            class CertifiedStop(Exception): pass
            def callback(q):
                if best['U']-best['L']<=1e-6*unit: raise CertifiedStop()
                if perf_counter()-start>time_limit: raise TimeoutError('dual time budget')
            try:
                constraints=[{'type':'eq','fun':lambda q:A@q-rhs,'jac':lambda q:A}]
                if env.mode=='budget': constraints.append({'type':'ineq','fun':lambda q:env.budget-env.evidence@q,'jac':lambda q:-env.evidence})
                opt=minimize(evaluate,initial,jac=True,method='SLSQP',bounds=list(zip(lo,hi)),
                    constraints=constraints,
                    callback=callback,options={'ftol':1e-12,'maxiter':100})
                evaluate(opt.x); info.update(dual_status=str(opt.message),iterations=int(opt.nit))
            except CertifiedStop: info['dual_status']='stopped at requested primal-dual gap'
            except TimeoutError as e: info['dual_status']=str(e)
    elif not reference and not native_done and env.mode in ('soft','lr','budget','filter','sample','pr'):
        # Primal first-order optimization; lower bounds are recovered independently.
        def objective(w):
            val,q,add=env.support(sc.costs(w)-V)
            grad=np.einsum('m,mij,j->i',q,sc.Q,w)-q@sc.mu
            if sc.kappa: grad+=q.sum()*sc.kappa*np.sign(w-sc.previous)
            return val/unit,grad/unit
        w0=aggregate_qp(sc,post.p)['w']; calls+=1
        opt=minimize(objective,w0,jac=True,method='SLSQP',bounds=[(0,sc.upper)]*d,
            constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(d)},
            options={'ftol':1e-13,'maxiter':250})
        w=feasible(opt.x,np.full(d,sc.upper)); _,q,add=env.support(sc.costs(w)-V)
        update(w,q,add); info.update(primal_status=str(opt.message),iterations=int(opt.nit))
    need_reference=reference or best['U']-best['L']>1e-6*unit
    if need_reference and perf_counter()-start<time_limit:
        info['used_conic_reference']=True
        try:
            w,q,pval,status=conic_reference(sc,env)
            # Conic constraint duals are feasible up to reported numerical residuals.
            # Smooth risk has an explicit valid adversary at the returned action.
            if env.mode in ('soft','lr'):
                _,q,add=env.support(sc.costs(w)-V)
            else: add=-float(q@env.offset)
            update(w,q,add); info.update(reference_value=pval,reference_status=status)
        except Exception as exc:
            info['reference_error']=str(exc)
            if not np.isfinite(best['U']): raise
    w=best['w']; gap=float(best['U']-best['L'])
    residual=max(abs(w.sum()-1),float(np.maximum(-w,0).max()),float(np.maximum(w-sc.upper,0).max()))
    status='converged' if gap>=-1e-9*unit and gap<=1e-6*unit and residual<=1e-8 else 'gap_open'
    info['elapsed_over_limit']=perf_counter()-start>time_limit
    return DecisionResultV2(w,env.support(sc.costs(w)-V)[0],float(best['L']),float(best['U']),gap,status,
        perf_counter()-start,calls,residual,best['q'],info)
