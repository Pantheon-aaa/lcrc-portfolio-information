"""Independent checks of Theorem 6 and its exact Gaussian sufficient experiment."""
import json,time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from core import solve_qp,MaskLikelihood
from .risk_envelopes import tail_weights
from .experiment_v2 import OUT,CFG,write_json,append_row
from .new_generators import seed_for

def covariance(k,c,delta=.2):
    J=np.ones((k,k));A=(1-delta/k)*J+delta*np.eye(k);B=(4-delta/k)*J+delta*np.eye(k)
    return np.block([[A,c*J],[c*J,B]])

def kl(A,B):
    return .5*(np.trace(np.linalg.solve(B,A))-len(A)+np.linalg.slogdet(B)[1]-np.linalg.slogdet(A)[1])

def run():
    begin=time.perf_counter();cpu=time.process_time();a=1.;s=.7;checks=[]
    for k in (5,10):
        for t in (.1,.03,.01):
            Q1=covariance(k,1-a*t);Q2=covariance(k,1+s);d=2*k
            x=a*t/(3+2*a*t);w1=np.r_[np.full(k,(1-x)/k),np.full(k,x/k)]
            w2=np.r_[np.ones(k)/k,np.zeros(k)]
            p1=solve_qp(Q1,constraints={'upper':.2});p2=solve_qp(Q2,constraints={'upper':.2})
            assert np.linalg.norm(p1['w']-w1)<1e-4 and np.linalg.norm(p2['w']-w2)<1e-4
            C1=np.array([[1,1-a*t],[1-a*t,4.]]);C2=np.array([[1,1+s],[1+s,4.]])
            assert abs(kl(Q1,Q2)-kl(C1,C2))<1e-10
            rng=np.random.default_rng(seed_for('embedded-gate',k,t));Y=np.full((10,d),np.nan)
            Y[:5,:k]=rng.normal(size=(5,k));Y[5:,k:]=rng.normal(size=(5,k));like=MaskLikelihood(Y)
            assert abs(like(Q1)-like(Q2))<1e-10
            checks.append({'d':d,'t':t,'KL':kl(Q1,Q2),'min_eigenvalue':min(np.linalg.eigvalsh(Q1).min(),np.linalg.eigvalsh(Q2).min()),
                'qp_distance':max(np.linalg.norm(p1['w']-w1),np.linalg.norm(p2['w']-w2))})
    config={'t':[.04,.02,.01,.005],'log_multipliers':[0,20,80,320],'repetitions':100,'rho':[1.,.5,.25],
        'a':a,'s':s,'dimension':20,'prior':[.5,.5],
        'interpretation':'mechanism diagnostic with public two-candidate family; projected observations have exactly the full-block likelihood ratio'}
    write_json(CFG/'embedded_boundary.json',config)
    path=OUT/'embedded_boundary.jsonl'
    if path.exists():raise RuntimeError('do not overwrite mechanism results')
    for t in config['t']:
        C=[np.array([[1,1-a*t],[1-a*t,4.]]),np.array([[1,1+s],[1+s,4.]])]
        precision=[np.linalg.inv(c) for c in C];ld=[np.linalg.slogdet(c)[1] for c in C]
        xoracle=a*t/(3+2*a*t)
        def losses(x):return np.array([.5*(3+2*a*t)*(x-xoracle)**2,.5*(3-2*s)*x*x+s*x])
        for multiplier in config['log_multipliers']:
            m=int(np.ceil(multiplier*np.log(1/t)))
            for truth in (0,1):
                for rep in range(config['repetitions']):
                    rng=np.random.default_rng(seed_for('embedded',t,m,truth,rep));Y=rng.normal(size=(m,2))@np.linalg.cholesky(C[truth]).T
                    ell=np.array([-.5*(m*v+np.einsum('ni,ij,nj->',Y,P,Y)) for P,v in zip(precision,ld)])
                    p2=expit(ell[1]-ell[0]);p=np.array([1-p2,p2])
                    for rho in config['rho']:
                        def objective(x):v=losses(x);return tail_weights(v,p,rho)@v
                        fit=minimize_scalar(objective,bounds=(0,1),method='bounded',options={'xatol':1e-13})
                        x=min((0.,1.,fit.x),key=objective)
                        append_row(path,{'t':t,'m':m,'log_multiplier':multiplier,'truth':truth,'rep':rep,'rho':rho,
                            'posterior_adverse':p2,'x_over_t':x/t,'regret':float(losses(x)[truth]),'regret_over_t2':float(losses(x)[truth]/t**2)})
    write_json(OUT/'embedded_boundary_gate.json',{'passed':True,'checks':checks})
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'embedded_boundary','wall_seconds':time.perf_counter()-begin,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})
    print('embedded covariance, likelihood and QP checks passed',flush=True)

if __name__=='__main__':run()
