"""80-digit active-set KKT audit and log-posterior underflow diagnostic."""
import numpy as np
import mpmath as mp
from .common import OUT,rows,write,run
from .families import make
from .mean_budget import MeanBudget
from .api import solve_budgeted_dcl,shared_optimizer_certificate

def exact_qp(Q,warm,upper):
    d=len(warm);lo=[i for i in range(d) if warm[i]<1e-8];hi=[i for i in range(d) if warm[i]>upper-1e-8];free=[i for i in range(d) if i not in lo+hi];w=mp.matrix(d,1)
    for i in hi:w[i]=mp.mpf(upper)
    if not free:raise ValueError('audit active set has no free coordinate')
    K=mp.matrix(len(free)+1);rhs=mp.matrix(len(free)+1,1)
    for a,i in enumerate(free):
        for b,j in enumerate(free):K[a,b]=Q[i,j]
        K[a,len(free)]=1;K[len(free),a]=1;rhs[a]=-sum(Q[i,j]*w[j] for j in hi)
    rhs[len(free)]=1-sum(w);sol=mp.lu_solve(K,rhs);y=sol[len(free)]
    for a,i in enumerate(free):w[i]=sol[a]
    grad=Q*w+mp.matrix([y]*d)
    valid=all(w[i]>=0 and w[i]<=mp.mpf(upper) for i in free) and all(grad[i]>=0 for i in lo) and all(grad[i]<=0 for i in hi)
    if not valid:raise RuntimeError('high precision active set failed KKT signs')
    return (w.T*Q*w)[0]/2,w

def execute(budget):
    mp.mp.dps=80;records=[];chosen=set();sources=list(rows(OUT/'S2_confirmation_evidence.jsonl'))
    # First n=1024 dataset per family/mask, predetermined independently of outcomes.
    for r in sources:
        key=(r['family'],r['mask_design'])
        if r['n_blocks']!=1024 or key in chosen:continue
        chosen.add(key);budget.check();f=make(r['kind'],r['d'],6,'confirmation',int(r['family'].rsplit('-',1)[1]));p=np.array(r['posterior']);mb=MeanBudget(f,p)
        ll=[mp.mpf(x) for x in r['log_observation_density']];mx=max(ll);pmp=[mp.exp(x-mx) for x in ll];den=sum(pmp);pmp=[x/den for x in pmp]
        matrices=[mp.matrix(q.tolist()) for q in f.Q];values=[exact_qp(Q,w,f.upper)[0] for Q,w in zip(matrices,f.oracle)];Qbar=sum((pj*Q for pj,Q in zip(pmp,matrices)),mp.matrix(f.d));V,wm=exact_qp(Qbar,mb.w,f.upper);L=V-sum(pj*v for pj,v in zip(pmp,values))
        outside=max(mp.mpf(mb.Llo)-L,L-mp.mpf(mb.Lhi),mp.mpf(0));oracle_outside=max(max(mp.mpf(f.VL[j])-v,v-mp.mpf(f.VU[j]),mp.mpf(0)) for j,v in enumerate(values))
        records.append(dict(dataset=r['dataset'],Lstar_80_digit=str(L),float_interval=[mb.Llo,mb.Lhi],interval_outside=str(outside),oracle_interval_outside=str(oracle_outside),posterior_min_80_digit=str(min(pmp)),float_zero_weights=int(np.count_nonzero(p==0)),KKT_checked=True))
        # Explicitly report tiny non-directed float violations rather than hiding them.
        if outside>mp.mpf('1e-12')*f.unit or oracle_outside>mp.mpf('1e-14'):raise RuntimeError('high precision contradiction beyond declared float arithmetic accuracy')
        if len(records)==20:break
    from .mean_budget import Family
    f=Family(np.repeat(np.eye(6)[None,:,:],3,axis=0),.2);a=solve_budgeted_dcl(f,np.ones(3)/3,1e-4,.1)
    assert shared_optimizer_certificate(f)['certified'] and a['budget_lower']==0 and a['posterior_mean_regret_MA_upper']==0
    distinct=make('asymmetric',10,6,'gate',77);assert not shared_optimizer_certificate(distinct)['certified']
    write(OUT/'high_precision_audit.json',dict(digits=80,records=records,structural_zero_example=a,underflow_not_structural=True,qualification='Float primal objective is not outward-rounded; sub-ulp oracle violations are disclosed, not claimed rigorous interval arithmetic.'))
    print(f'High precision: {len(records)} fixed cases checked',flush=True)

if __name__=='__main__':run('high_precision_audit',execute)
