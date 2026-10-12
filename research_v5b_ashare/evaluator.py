"""Future-only module. It is never imported by decision or risk frontend modules."""
import numpy as np
import cvxpy as cp
from core import feasible

def empirical_oracle(Q,upper):
    # A feasible primal point and a first-order convex lower bound also work for singular PSD Q.
    scale=float(np.trace(Q)/len(Q))
    if scale==0:return dict(w=np.ones(len(Q))/len(Q),lower=0.,upper=0.,gap=0.,residual=0.)
    x=cp.Variable(len(Q));problem=cp.Problem(cp.Minimize(.5*cp.quad_form(x,cp.psd_wrap(Q/scale))),[cp.sum(x)==1,x>=0,x<=upper])
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=300)
    if x.value is None:raise ValueError('empirical QP failed')
    w=feasible(np.asarray(x.value),np.full(len(Q),upper));g=Q@w
    v=np.zeros(len(Q));remaining=1.
    for i in np.argsort(g,kind='stable'):v[i]=min(remaining,upper);remaining-=v[i]
    value=float(.5*w@Q@w);pad=64*np.finfo(float).eps*np.linalg.norm(Q)*np.dot(w,w)
    lower=max(0.,value+float(g@(v-w))-pad)
    return dict(w=w,lower=lower,upper=value+pad,gap=value+pad-lower,residual=max(abs(w.sum()-1),-w.min(),w.max()-upper,0.))

def label(data,panel,columns,H=63,stale=63):
    t=panel['t'];end=t+H+1
    if end>=len(data.dates):return dict(status='incomplete_horizon')
    from .common import phase
    if phase(panel['date'])!=phase(data.dates[end]):return dict(status='cross_phase_label',label_end=data.dates[end])
    from .valuation import Valuation,LabelPolicy
    x,quality=Valuation().window(t,columns,H,LabelPolicy(stale))
    if not quality['complete']:return dict(status='unknown_future_returns',quality=quality,label_end=data.dates[end])
    Q=x.T@x/H;oracle=empirical_oracle(Q,min(.2,5/len(columns)))
    return dict(status='complete',Q=Q,oracle=oracle,quality=quality,label_start=data.dates[t+2],label_end=data.dates[end],H=H)

def assess(label,w,epsilon):
    if label['status']!='complete':return dict(status=label['status'],success_lower=0.,success_upper=1.)
    loss=float(.5*w@label['Q']@w);o=label['oracle'];lo=loss-o['upper'];hi=loss-o['lower']
    return dict(status='success' if hi<=epsilon else 'failure' if lo>epsilon else 'indeterminate',loss=loss,regret_lower=lo,regret_upper=hi,
      success_lower=float(hi<=epsilon),success_upper=float(lo<=epsilon))

def crossfit_label(data,panel,columns):
    from .valuation import Valuation,LabelPolicy
    from .risk import shrink
    t=panel['t'];end=t+127
    from .common import phase
    if end>=len(data.dates):return dict(status='incomplete_horizon')
    if phase(panel['date'])!=phase(data.dates[end]):return dict(status='cross_phase_label')
    x,quality=Valuation().window(t,columns,126,LabelPolicy(63))
    if not quality['complete']:return dict(status='unknown_future_returns',quality=quality)
    which=(np.arange(126)//5)%2;A=x[which==0];B=x[which==1];QA=A.T@A/len(A);QB=B.T@B/len(B);upper=min(.2,5/len(columns))
    a=empirical_oracle(shrink(QA,.3),upper)['w'];b=empirical_oracle(shrink(QB,.3),upper)['w']
    reference=float(.25*(a@QB@a+b@QA@b))
    return dict(status='complete',Q_average=(QA+QB)/2,reference=reference,quality=quality,block_counts=[len(A),len(B)])

def assess_crossfit(lab,w):
    if lab['status']!='complete':return dict(status=lab['status'],quality=lab.get('quality'),signed_excess_risk=None)
    loss=float(.5*w@lab['Q_average']@w)
    return dict(status='crossfit_excess_risk',quality=lab['quality'],loss=loss,signed_excess_risk=loss-lab['reference'],block_counts=lab['block_counts'])
