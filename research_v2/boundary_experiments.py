"""Analytic two-model multiasset boundary and hidden-curvature constructions."""
import json
from time import perf_counter,process_time,time
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp
from core import solve_qp,MaskLikelihood,sample_data
from .scenario_inference import ScenarioSet,posterior_from_loglik
from .risk_envelopes import RiskSpec,tail_weights
from .risk_center import solve_center
from .experiment_v2 import OUT,append_row,write_json,source_hash


def solve_1d(Qs,p,t,kind,rho=1.,tau=.1):
    """Reference minimization on the known optimal two-asset face, no QP tolerance floor."""
    A=Qs[:,1,1]+Qs[:,0,0]-2*Qs[:,0,1]; B=Qs[:,0,1]-Qs[:,0,0]
    xmin=np.clip(-B/A,0,1)
    def regrets(x): return .5*A*(x-xmin)**2+(A*xmin+B)*(x-xmin)
    def objective(x):
        r=regrets(x)
        if kind=='max': return float(r.max())
        if kind=='soft': return float(tau*logsumexp(np.log(p)+r/tau))
        return float(tail_weights(r,p,rho)@r)
    # Include exact kinks and endpoints, plus each smooth segment's optimum.
    candidates=[0.,1.]
    coeff=[.5*(A[0]-A[1]),B[0]-B[1],-.5*A[0]*xmin[0]**2-B[0]*xmin[0]+.5*A[1]*xmin[1]**2+B[1]*xmin[1]]
    roots=np.roots(np.trim_zeros(coeff,'f')) if np.any(coeff) else []
    candidates += [float(x.real) for x in roots if abs(x.imag)<1e-10 and 0<x.real<1]
    breaks=sorted(candidates)
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        z=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-15}); candidates.append(float(z.x))
    x=min(candidates,key=objective)
    return x,objective(x),regrets(x)


def run():
    start=perf_counter(); cpu=process_time(); sha=source_hash(); path=OUT/'boundary.jsonl'
    done={(x['d'],x['seed'],x['mechanism'],x['level'],x['probability'],x['method']) for x in
          [json.loads(z) for z in path.read_text().splitlines()]} if path.exists() else set()
    for d in (6,20):
        for seed in range(5):
            for mechanism in ('boundary','hidden_curvature'):
                z1=np.r_[1.,1.,np.full(d-2,1.4)]; z2=z1.copy(); z2[1]=1.2 if mechanism=='boundary' else 1.
                diagonal=np.r_[1.,3.+.1*seed,np.full(d-2,4.)]
                q1=np.outer(z1,z1)+np.diag(diagonal-z1*z1)
                q2=np.outer(z2,z2)+np.diag(diagonal-z2*z2)
                if mechanism=='hidden_curvature':
                    z1[:3]=1.;z2[:3]=1.
                    q1=np.outer(z1,z1)+np.diag(diagonal-z1*z1)
                    q2=np.outer(z2,z2)+np.diag(diagonal-z2*z2)
                    q2[1,2]+= .4+.02*seed; q2[2,1]+=.4+.02*seed
                E=np.zeros((d,d)); E[0,1]=E[1,0]=-1
                if mechanism=='hidden_curvature': E[0,2]=E[2,0]=-.6
                for level in range(8):
                    t=.02/2**level; Qs=np.array([q1+t*E,q2+t*E])
                    for probability,pbad in [('fixed',.2),('sqrt',np.sqrt(t)),('linear',t),('quadratic',t*t),
                                             ('fixed_floor',max(t*t,.01)),('vanishing_floor',max(t*t,t**1.5))]:
                        p=np.array([1-pbad,pbad])
                        for name,kind,rho,tau in [('Mean','tail',1.,.1),('Tail:.5','tail',.5,.1),('Tail:.25','tail',.25,.1),
                            ('Max','max',1.,.1),('Soft-fixed','soft',1.,.01),('Soft-t2','soft',1.,t*t),
                            ('Soft-t1.5','soft',1.,t**1.5)]:
                            key=(d,seed,mechanism,level,probability,name)
                            if key in done: continue
                            if mechanism=='boundary':
                                x,value,r=solve_1d(Qs,p,t,kind,rho,tau)
                                w=np.zeros(d); w[0]=1-x; w[1]=x
                            else:
                                w,value,r=solve_curvature(Qs,p,t,kind,rho,tau)
                                x=w[1]
                            # Check the face against the full d-dimensional optimizer at
                            # coarser scales, where float64 oracle gaps do not dominate.
                            check=None
                            if level in (0,3) and probability in ('linear','quadratic') and name in ('Mean','Tail:.5','Max','Soft-t2'):
                                sc=ScenarioSet(Qs,np.zeros((2,3)),np.zeros(2,int),np.log(p),upper=1.).prepare()
                                post=posterior_from_loglik(sc,np.zeros(2)); spec=RiskSpec(name,kind,rho=rho,temperature=tau/max(t*t,1e-9))
                                ans=solve_center(sc,post,spec,max(t*t,1e-9)); check={'gap':ans.gap,'weight_distance':float(np.linalg.norm(w-ans.weights)),'status':ans.status}
                            h=diagonal[1]-1; c=pbad/t
                            if mechanism=='boundary' and kind=='tail': predicted=max(1-c*.2/rho,0)/h
                            else: predicted=None
                            append_row(path,dict(d=d,seed=seed,mechanism=mechanism,level=level,t=t,probability=probability,
                                pbad=pbad,method=name,x=x,x_over_t=x/t,value=value,value_over_t2=value/t**2,
                                local_prediction=predicted,full_dimension_check=check,scenario_regrets=r,
                                scaled_direction=(w-np.eye(d)[0])/t,diagonal_difference=float(np.max(abs(np.diag(Qs[0])-np.diag(Qs[1])))),source_hash=sha))
            print('boundary',d,seed,flush=True)
    # Data-driven posterior: same diagonal, joint observations identify cross-covariance.
    rng=np.random.default_rng(621039); data_path=OUT/'boundary_data.jsonl'
    if not data_path.exists():
        for m in (0,16,64,256,1024):
            for rep in range(40):
                d=6;t=.01;z1=np.r_[1.,1.,np.full(d-2,1.4)];z2=z1.copy();z2[1]=1.2
                diag=np.r_[1.,3.,np.full(d-2,4.)]; Qs=np.array([np.outer(z,z)+np.diag(diag-z*z) for z in (z1,z2)])
                E=np.zeros((d,d)); E[0,1]=E[1,0]=-1;Qs+=t*E
                # Singleton marginals when not jointly observed: exact equivalence.
                Y=rng.normal(size=(max(m,1)+128,d))@np.linalg.cholesky(Qs[0]).T
                mask=np.zeros_like(Y,dtype=bool);mask[:m]=True
                for i in range(m,len(Y)):mask[i,i%d]=True
                Y[~mask]=np.nan; like=MaskLikelihood(Y); ll=np.array([like(q) for q in Qs]); p=np.exp(ll-logsumexp(ll))
                for rho in (1.,.5,.25):
                    x,value,_=solve_1d(Qs,p,t,'tail',rho)
                    append_row(data_path,dict(m=m,rep=rep,rho=rho,pbad=p[1],x_over_t=x/t,target_true_regret=.5*(2+2*t)*(x-t/(2+2*t))**2,
                        interpretation='data-derived posterior; unequal scalar budgets across m; mechanism only'))
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'boundary','wall_seconds':perf_counter()-start,'process_cpu_seconds':process_time()-cpu,'unix':time(),'source_hash':sha})


def solve_curvature(Qs,p,t,kind,rho,tau):
    d=Qs.shape[1]; U=np.zeros((d,2)); U[0]=-1;U[1,0]=1;U[2,1]=1
    K=np.einsum('ia,mij,jb->mab',U,Qs,U); e=np.array([1.,.6])
    centers=np.array([np.linalg.solve(k,e) for k in K]); assert centers.min()>0
    def losses(v): return .5*np.einsum('mi,mij,mj->m',v-centers,K,v-centers)
    def fun(v):
        r=losses(v)
        if kind=='max': return float(r.max())
        if kind=='soft': return float(tau/t**2*logsumexp(np.log(p)+r*t**2/tau))
        return float(tail_weights(r,p,rho)@r)
    start=p@centers
    if kind=='soft':
        fit=minimize(fun,start,method='SLSQP',bounds=[(0,2)]*2,options={'ftol':1e-13,'maxiter':100})
        v=fit.x
    else:
        q0=1. if kind=='max' else min(1.,p[0]/rho)
        q1=0. if kind=='max' else max(0.,1-p[1]/rho)
        qs=[np.array([q0,1-q0]),np.array([q1,1-q1])]
        cons=[{'type':'ineq','fun':lambda z,q=q:z[2]-q@losses(z[:2])} for q in qs]
        fit=minimize(lambda z:z[2],np.r_[start,fun(start)],method='SLSQP',bounds=[(0,2),(0,2),(0,None)],
            constraints=cons,options={'ftol':1e-13,'maxiter':100}); v=fit.x[:2]
    w=np.eye(d)[0]+t*U@v
    return w,t*t*fun(v),t*t*losses(v)


if __name__=='__main__': run()
