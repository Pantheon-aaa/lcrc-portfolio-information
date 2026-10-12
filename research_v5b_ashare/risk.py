"""Zero-mean observed-data EM and genuinely prequential expert scoring."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import logsumexp
from .common import *

def em_second(x,max_iter=50,tol=1e-6):
    x=np.asarray(x,float);n,d=x.shape;observed=np.isfinite(x)
    count=observed.sum(0)
    if n==0 or np.any(count==0):raise ValueError('unobserved asset in estimation window')
    if observed.all():return x.T@x/n,dict(status='complete',iterations=0,relative_change=0.,observations=count.tolist())
    diag=np.nansum(x*x,axis=0)/count
    if not np.all(diag>0):raise ValueError('nonpositive historical variance')
    S=np.diag(diag);masks,inv=np.unique(observed,axis=0,return_inverse=True)
    groups=[(mask,np.flatnonzero(inv==k)) for k,mask in enumerate(masks)]
    relative=np.inf
    for it in range(max_iter):
        T=np.zeros_like(S)
        for mask,idx in groups:
            if mask.all():T+=x[idx].T@x[idx];continue
            if not mask.any():T+=len(idx)*S;continue
            missing=~mask;xo=x[np.ix_(idx,mask)]
            Soo=S[np.ix_(mask,mask)];Smo=S[np.ix_(missing,mask)]
            # A scale-relative stabilization is used in the conditional solves only.
            C=cho_solve(cho_factor(Soo+np.eye(mask.sum())*np.trace(S)/d*1e-12,lower=True,check_finite=False),Smo.T,check_finite=False).T
            filled=np.empty((len(idx),d));filled[:,mask]=xo;filled[:,missing]=xo@C.T
            T+=filled.T@filled
            T[np.ix_(missing,missing)]+=len(idx)*(S[np.ix_(missing,missing)]-C@Smo.T)
        T=(T+T.T)/(2*n);relative=float(np.linalg.norm(T-S)/np.linalg.norm(S));S=T
        if relative<=tol:break
    return S,dict(status='converged' if relative<=tol else 'max_iterations',iterations=it+1,relative_change=relative,observations=count.tolist())

def shrink(S,a):
    d=len(S);unit=float(np.trace(S)/d)
    if not np.isfinite(unit) or unit<=0:raise ValueError('invalid historical risk scale')
    return (1-a)*S+a*np.diag(np.diag(S))+np.eye(d)*(1e-8*unit)

def log_density(q,x):
    mask=np.isfinite(x)
    if not mask.any():return 0.
    y=x[mask];factor=cho_factor(q[np.ix_(mask,mask)],lower=True,check_finite=False)
    return float(-.5*(len(y)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+y@cho_solve(factor,y,check_finite=False)))

def posterior(scores,eta):
    z=eta*np.sum(scores,axis=0);lp=z-logsumexp(z);return np.exp(lp),lp

def fit_models(history,lengths=(126,252,504),strengths=(.1,.3,.6,.9)):
    Q=[];diagnostics=[]
    for L in lengths:
        S,info=em_second(history[-L:])
        for a in strengths:
            q=shrink(S,a);Q.append(q);diagnostics.append(dict(L=L,a=a,**info))
    return np.array(Q),diagnostics

def fit_bootstrap(history,offset=0):
    x=history[-252:];Q=[];ds=[]
    for seed in [101,211,307,401,503,601,701,809,907,1009,1103,1201]:
        rng=np.random.default_rng(seed+offset);starts=rng.integers(0,len(x)-9,size=int(np.ceil(len(x)/10)))
        idx=np.concatenate([np.arange(i,i+10) for i in starts])[:len(x)];S,info=em_second(x[idx]);Q.append(shrink(S,.3));ds.append(dict(seed=seed+offset,L=252,a=.3,block_length=10,**info))
    return np.array(Q),ds

def frontend(data,panel,N=50,lengths=(126,252,504),strengths=(.1,.3,.6,.9),score=True,mode='F1',offset=0,cache=True):
    cols=panel['ranking'][:N];t=panel['t']
    if len(cols)<N:raise ValueError('insufficient universe')
    key=digest('F1-EM-v1',t,','.join(map(str,cols)),lengths,strengths,score) if mode=='F1' else digest('F2-EM-v1',t,cols,score,offset)
    path=PRIVATE/'frontends'/f'{key}.npz';meta=path.with_suffix('.json')
    if cache and path.exists() and meta.exists():
        z=np.load(path);return dict(Q=z['Q'],scores=z['scores'],u=float(z['u']),columns=cols,shared_cost_id=key,metadata=read(meta))
    start=cpu();wall=time.perf_counter();history=data.returns(t,cols,max(lengths)+63)
    fit=lambda x:fit_models(x,lengths,strengths) if mode=='F1' else fit_bootstrap(x,offset)
    Q,diag=fit(history)
    S,refdiag=em_second(history[-252:]);ref=shrink(S,.3);ew=np.ones(N)/N;u=float(.5*ew@ref@ew)
    if u<=0 or not np.isfinite(u):raise ValueError('invalid risk unit')
    scores=[];fit_status={};iters=0
    if score:
        for i in range(len(history)-63,len(history)):
            old,ds=fit(history[:i])
            scores.append([log_density(q,history[i])/N for q in old])
            for item in ds:
                fit_status[item['status']]=fit_status.get(item['status'],0)+1;iters+=item['iterations']
    scores=np.array(scores) if score else np.empty((0,len(Q)))
    eigen=np.linalg.eigvalsh(Q)
    metadata=dict(shared_cost_id=key,date=panel['date'],N=N,M=len(Q),input_history_hash=array_hash(history),Q_hash=array_hash(Q),
      u=u,cpu_seconds=cpu()-start,wall_seconds=time.perf_counter()-wall,training_end=t,score_last_day=t,score_fit_last_day=t-1,
      score_fit_status=fit_status,score_iterations=iters,current_models=diag,current_eigen_min=eigen[:,0],current_condition=eigen[:,-1]/eigen[:,0],
      generalized_weights_not_correct_model_posterior=True)
    if cache:
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,Q=Q,scores=scores,u=u);tmp.replace(path);write(meta,metadata)
    return dict(Q=Q,scores=scores,u=u,columns=cols,shared_cost_id=key,metadata=metadata)
