"""Historical Gaussian bridge; truth is used only after decisions are formed."""
import time
from itertools import combinations
import numpy as np
from scipy.special import logsumexp
from .common import *
from .data import Data
from .risk import em_second,shrink,log_density,frontend
from .decision import DecisionInput,Solver
from .pipeline import panels,inp
from research_v4a.price_oracle import Geometry
from research_v4a.baselines import cvx_price

def dictionary(history,cols,symbols):
    S,_=em_second(history[-252:]);ref=shrink(S,.3);d=len(cols)
    order=np.array(sorted(range(d),key=lambda i:digest(symbols[cols[i]])));a=order[:d//2];b=order[d//2:]
    def power(X,e):
        val,vec=np.linalg.eigh(X);return (vec*val**e)@vec.T
    A=ref[np.ix_(a,a)];B=ref[np.ix_(b,b)];C=power(A,-.5)@ref[np.ix_(a,b)]@power(B,-.5);norm=np.linalg.norm(C,2)
    if norm:C*=min(1,.8/norm)
    Q=[]
    for h in [.9,1.1]:
        for k in [-1,0,1]:
            block=h*np.block([[A,power(A,.5)@(k*C)@power(B,.5)],[power(B,.5)@(k*C).T@power(A,.5),B]])
            q=np.empty_like(block);q[np.ix_(order,order)]=block;Q.append(q)
    ew=np.ones(d)/d;return np.array(Q),float(.5*ew@ref@ew),a,b

def view_data(pairs,a,b,view):
    if view.startswith('Joint'):return pairs[:,int(view[-1])].copy()
    if view=='Full':return pairs.reshape(-1,pairs.shape[-1]).copy()
    orientation=int(view[-1]);x=np.full_like(pairs,np.nan);x[:,orientation,a]=pairs[:,orientation,a];x[:,1-orientation,b]=pairs[:,1-orientation,b]
    return x.reshape(-1,pairs.shape[-1])

def evidence(Q,x):return np.array([sum(log_density(q,row) for row in x) for q in Q])

def enumerate_prices(s):
    out=[]
    for n in range(s.f.M+1):
        for G in combinations(range(s.f.M),n):
            z=s.pc.get(G,time.perf_counter()+30,tight=True);cv=cvx_price(s.f,s.mb.p,G,s.eps)
            out.append(dict(G=G,mass=float(s.mb.p[list(G)].sum()),lower=z.cost_lower,upper=z.cost_upper,feasible=z.epsilon_feasible,status=z.status,
                            w=z.witness_weights,cvx=cv))
    return out

def r1():
    cfg=read(ROOT/'protocol_frozen.json');data=Data()
    jobs=[p for p in panels('development') if p['seed']==11 and p['quarterly'] and p['date'][:4] in ['2015','2016']]
    for p in jobs:
        if p['status']!='ready':write(PRIVATE/'r1'/f'{p["id"]}_unresolved.json',dict(id=p['id'],status=p['status']));continue
        front=frontend(data,p,N=20,strengths=(.1,.6));base=Solver(inp(front,0,cfg['eta']));g=Geometry(base.f,0).gamma(range(6));gamma=(g.lower+g.upper)/2
        for name,eps in [(f'tau{k}',v) for k,v in enumerate(cfg['tau'])]+[(f'geo{v}',v*gamma) for v in [.5,.75,1.05]]:
            path=PRIVATE/'r1'/f'{p["id"]}_{name}.json'
            if path.exists():continue
            start=cpu();s=Solver(inp(front,eps,cfg['eta']));prices=enumerate_prices(s)
            results=[s.solve(m,b) for b in [0,.1,.25,.5,np.inf] for m in ['A1B','Greedy']]
            write(path,dict(id=p['id'],domain=p['domain'],status='complete',threshold=name,epsilon_over_u=eps,prices=prices,results=results,u=front['u'],cpu_seconds=cpu()-start))
            print('R1',p['id'],name,flush=True)

def semisynthetic():
    cfg=read(ROOT/'protocol_frozen.json');data=Data()
    jobs=[p for p in panels('development') if p['date']=='2015-03-31' and p['seed']==11]
    for panel in jobs:
        cols=panel['ranking'][:20];Q,u,a,b=dictionary(data.returns(panel['t'],cols,252),cols,data.symbols)
        uniform=np.ones(6)/6;dummy=Solver(DecisionInput(Q,uniform,np.log(uniform),u,0,.2,panel['id']))
        gg=Geometry(dummy.f,0).gamma(range(6));gamma=(gg.lower+gg.upper)/2
        root=PRIVATE/'semisynthetic'/panel['domain'];root.mkdir(parents=True,exist_ok=True);np.savez_compressed(root/'family.npz',Q=Q,u=u,a=a,b=b)
        for h in [0,1]:
            p=np.zeros(6);p[h*3:h*3+3]=1/3;lp=np.full(6,-np.inf);lp[p>0]=np.log(p[p>0])
            for name,e in [('tau',cfg['tau'][1]),('geo0.5',.5*gamma),('geo0.75',.75*gamma),('geo1.05',1.05*gamma)]:
                dest=root/f'platform_h{h}_{name}.json'
                if not dest.exists():
                    s=Solver(DecisionInput(Q,p,lp,u,e*u,.2,panel['id']));write(dest,dict(h=h,threshold=name,prices=enumerate_prices(s)))
        runs=[(-1,0,0,'prior')]+[(truth,rep,n,view) for truth in range(6) for rep in range(10) for n in [32,128,512] for view in ['Separated0','Joint0']]
        for kk,(truth,rep,n,view) in enumerate(runs):
            dest=root/f't{truth}_r{rep}_n{n}_{view}.json'
            if dest.exists():continue
            start=cpu()
            if n==0:ll=np.zeros(6)
            else:
                seed=int(digest(20261012,'v5b-bridge',panel['domain'],truth,rep)[:16],16);rng=np.random.default_rng(seed)
                pairs=rng.multivariate_normal(np.zeros(20),Q[truth],size=(512,2))[:n];x=view_data(pairs,a,b,view);ll=evidence(Q,x)
            lp=ll-logsumexp(ll);p=np.exp(lp);results=[]
            for name,e in [('tau',cfg['tau'][1]),('geo0.5',.5*gamma),('geo0.75',.75*gamma),('geo1.05',1.05*gamma)]:
                s=Solver(DecisionInput(Q,p,lp,u,e*u,.2,panel['id']))
                for method,beta in [('MA',0.),('A1B',.1),('Greedy',.1),('A1B',.25),('Greedy',.25)]:
                    r=s.solve(method,beta);lo,hi=s.f.regret_bounds(r['evaluated_target'])
                    evaluation=dict(regret_lower=lo*u,regret_upper=hi*u,success_lower=(hi<=e).astype(float),success_upper=(lo<=e).astype(float))
                    results.append(dict(threshold=name,method=method,beta=beta,decision=r,evaluation=evaluation))
            write(dest,dict(status='complete',domain=panel['domain'],truth=truth,rep=rep,n=n,view=view,log_likelihood=ll,posterior=p,results=results,cpu_seconds=cpu()-start))
            if kk%10==0:print('BRIDGE',panel['domain'],kk+1,len(runs),flush=True)
