"""Independent formula checks; mathematical proofs remain in THEORY_NOTES."""
import numpy as np
from scipy.optimize import minimize_scalar
from .risk_envelopes import tail_weights
from .experiment_v2 import OUT,write_json

def run():
    rows=[];a=1.3;s=.8;h=2.2
    for t in (.1,.01,.001):
        for c in (.1,1.,3.):
            p2=c*t;p=np.array([1-p2,p2])
            for rho in (1.,.5,.25):
                if rho<1 and not (p2<rho and 1-p2>=rho):continue
                xc=a*a*t*t/(2*h*(a*t+s));root=(a*t-(p2/rho)*(a*t+s))/h
                exact=max(0,root) if rho==1 else max(xc,root)
                def obj(x):
                    R=np.array([h/2*(x-a*t/h)**2,h/2*x*x+s*x]);return tail_weights(R,p,rho)@R
                fit=minimize_scalar(obj,bounds=(0,1),method='bounded',options={'xatol':1e-14})
                x=min((0.,fit.x),key=obj)
                assert abs(x-exact)<2e-8,(t,c,rho,x,exact)
                rows.append({'t':t,'c':c,'rho':rho,'analytic':exact,'numerical':x})
    rng=np.random.default_rng(770023);violations=[]
    for _ in range(200):
        P=rng.dirichlet(np.ones(12)*.2);Q=rng.dirichlet(np.ones(12)*.2);R=rng.uniform(0,2,12)
        vals=[]
        for mass in (P,Q):
            val=0.
            for g in range(3):
                ix=slice(4*g,4*g+4);alpha=mass[ix].sum();val+=alpha*(tail_weights(R[ix],mass[ix]/alpha,.25)@R[ix])
            vals.append(val)
        bound=2/.25*.5*abs(P-Q).sum();assert abs(vals[0]-vals[1])<=bound+1e-12
        violations.append(abs(vals[0]-vals[1])-bound)
    write_json(OUT/'theory_formula_checks.json',{'passed':True,'finite_tail_formula':rows,'max_tv_bound_violation':max(violations)})
    print('finite formula and joint-mass stability checks passed')

if __name__=='__main__':run()
