from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from .likelihood import GaussianStream,verified_exclusions
from .common import seed
POLICIES=('equal_cost','joint_only','full_kl','conflict_kl','oracle_allocation')
@dataclass
class SequentialState:
    log_likelihood:np.ndarray
    error:np.ndarray
    confidence_mask:int
    count:np.ndarray
    desired:np.ndarray
    answer:int=-1
    t:int=0
@njit(cache=True)
def advance(ll,err,C,count,desired,answer,t,bank,bounds,used,valid,rate,full,conflict,answers,allowed,cost,policy,oracle,delta,horizon,cost_cap,action_record):
    M=len(ll);threshold=np.log(1/delta);maxadvance=len(action_record)
    for step in range(maxadvance):
        if t>=horizon:return answer,t,0,step
        th=0;best=-np.inf
        for j in range(M):
            if C>>j&1 and ll[j]>best:best=ll[j];th=j
        if policy==4:target=oracle
        elif policy==0:
            target=np.zeros(4)
            for a in allowed:target[a]=1/cost[a]
            target/=target.sum()
        elif policy==1:
            target=np.zeros(4);target[3 if len(allowed)>1 else allowed[0]]=1
        else:
            winner=0
            for g in range(len(answers)):
                if rate[th,g]>rate[th,winner]:winner=g
            if answer<0 or rate[th,answer]<0:answer=winner
            elif rate[th,winner]>rate[th,answer]*(1+max(1,t)**(-.25)):answer=winner
            target=full[th,answer] if policy==2 else conflict[C,th,answer]
        # Joint-only is a fixed one-action design. All adaptive allocations explore.
        desired+=target
        a=0;largest=-np.inf
        for x in allowed:
            z=desired[x]-count[x]
            if z>largest:largest=z;a=x
        if policy>=2:
            low=allowed[0]
            for x in allowed:
                if count[x]<count[low]:low=x
            if count[low]<np.sqrt(t)/len(allowed):a=low
        if used[a]>=valid[a]:
            desired-=target
            return answer,t,10+a,step
        if np.dot(count,cost)+cost[a]>cost_cap:
            desired-=target
            return answer,t,3,step
        z=bank[a,used[a]];e=bounds[a,used[a]];used[a]+=1
        err+=e+16*np.finfo(np.float64).eps*(np.abs(ll)+np.abs(z)+1)
        ll+=z;count[a]+=1;t+=1;action_record[step]=a
        mx=np.max(ll);logq=mx+np.log(np.exp(ll-mx).sum())-np.log(M)
        for j in range(M):
            if C>>j&1 and logq-ll[j]-(np.max(err)+err[j])>threshold+1e-10:
                return answer,t,1,step+1
    return answer,t,2,maxadvance
def simulate(f,geometry,table,constants,truth,rep,ratio,delta,premium,allowed,policy,horizon=100000,checkpoints=(),stream_namespace='confirmation'):
    tic=time.process_time();eps=ratio*(geometry[(1<<f.M)-1]['lower']+geometry[(1<<f.M)-1]['upper'])/2
    cost=np.array([f.d]*3+[premium*f.d],float);pidx=POLICIES.index(policy)
    streamseed=seed(f.name,truth,rep,stream_namespace)
    stream=GaussianStream(f,truth,streamseed,constants)
    state=SequentialState(np.zeros(f.M),np.zeros(f.M),(1<<f.M)-1,np.zeros(4),np.zeros(4))
    n=256;bank=np.zeros((4,n,f.M));bound=np.zeros_like(bank);used=np.zeros(4,np.int64);valid=used.copy()
    actions=[];events=[];snapshots={};status='budget_exhausted';stop=None
    oracle=table['full'][truth,np.argmax(table['rates'][truth])].copy() if policy=='oracle_allocation' else np.zeros(4)
    # Oracle only controls allocation; the certification rule remains identical.
    while True:
        C=state.confidence_mask
        if C and table['ok'][C] and stop is None:
            stop=dict(t=state.t,confidence_mask=C,weights=geometry[C]['weights'],certificate=geometry[C],cost=float(state.count@cost));status='certified'
        if state.t in checkpoints:snapshots[state.t]=dict(log_likelihood=state.log_likelihood.copy(),error=state.error.copy(),cost=float(state.count@cost))
        if (stop is not None and not checkpoints) or state.t>=horizon or C==0:
            if C==0:status='empty_confidence_set'
            break
        remaining=min([horizon]+[x for x in checkpoints if x>state.t])-state.t if checkpoints else horizon-state.t
        record=np.zeros(min(n,remaining),np.int8)
        state.answer,state.t,code,nread=advance(state.log_likelihood,state.error,C,state.count,state.desired,state.answer,state.t,bank,bound,used,valid,table['rates'],table['full'],table['conflict'],table['answers'],np.array(allowed),cost,pidx,oracle,delta,horizon,horizon*(max(cost) if checkpoints else f.d),record)
        actions.extend(record[:nread].tolist())
        if code>=10:
            a=code-10;bank[a],bound[a]=stream.chunk(a,n);valid[a]=n;used[a]=0
        elif code==1:
            excluded=verified_exclusions(state.log_likelihood,state.error,C,delta)
            if not excluded:raise ArithmeticError('fast crossing not verified by interval exponential')
            for j in excluded:state.confidence_mask&=~(1<<j)
            events.append(dict(t=state.t,excluded=excluded,log_likelihood=state.log_likelihood.copy(),error=state.error.copy(),confidence_mask=state.confidence_mask))
        elif code==3:break
    false=False;undetermined=False
    if stop:
        r=stop['certificate'];false=r['regret_lower'][truth]>eps;undetermined=r['regret_lower'][truth]<=eps<r['regret_upper'][truth]
    rate=float(table['rates'][truth].max());T=0 if np.isinf(rate) else 1/rate if rate>0 else np.inf
    return dict(family=f.name,kind=f.metadata.get('kind','special'),truth=truth,rep=rep,ratio=ratio,delta=delta,premium=premium,allowed=list(allowed),policy=policy,epsilon=eps,seed=streamseed,status=status,certified=stop is not None,false_certification=false,certification_truth_indeterminate=undetermined,stop_t=stop['t'] if stop else None,stop_cost=stop['cost'] if stop else None,restricted_cost=stop['cost'] if stop else horizon*f.d,censor_cost=horizon*f.d,counts=state.count,log_likelihood=state.log_likelihood,error=state.error,confidence_mask=state.confidence_mask,events=events,stop=stop,actions=np.array(actions,dtype=np.int8),snapshots=snapshots,T_candidate=T,cpu_seconds=time.process_time()-tic)
