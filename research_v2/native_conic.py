"""Direct Clarabel epigraph, avoiding modeling-language setup overhead.

Independent from the weighted-QP dual and retained CVXPY reference. Each returned
point is evaluated by their existing weak-duality bounds before use.
"""
import numpy as np
import scipy.sparse as sp
import clarabel
from core import feasible

def solve(sc,env,time_limit=60.):
    if sc.kappa:raise NotImplementedError('use the existing common-penalty solver')
    if env.mode not in ('tail','max','count','filter','sample','budget'):
        raise NotImplementedError(env.mode)
    M,d,_=sc.Q.shape;u=env.unit;spec=env.spec;blocks=env.blocks();G=len(blocks)
    hinge=env.mode!='max';budget=env.mode=='budget'
    ir=d;iz=d+M;iy=iz+G;ig=iy+(M if hinge else 0);N=ig+int(budget)
    c=np.zeros(N);group=np.zeros(M,dtype=int)
    for g,(ix,mass) in enumerate(blocks):
        group[ix]=g;c[iz+g]=mass*spec.blend
        if hinge:c[iy+ix]=spec.blend*env.p[ix]/env.block_rho(ix)
    c[ir:ir+M]=(1-spec.blend)*env.p
    if budget:c[ig]=env.budget
    matrices=[];rhs=[];cones=[]
    A=sp.csc_matrix((np.ones(d),(np.zeros(d),np.arange(d))),shape=(1,N))
    matrices.append(A);rhs.append(np.ones(1));cones.append(clarabel.ZeroConeT(1))
    # Nonnegative slacks implement lower/upper bounds and tail epigraph inequalities.
    rr=[];cc=[];vv=[];bb=[];nr=0
    def add(cols,values,b):
        nonlocal nr
        rr.extend([nr]*len(cols));cc.extend(cols);vv.extend(values);bb.append(b);nr+=1
    for j in range(d):add([j],[-1],0);add([j],[1],sc.upper)
    V=np.zeros(M) if spec.absolute else (sc.oracle_lower+sc.oracle_upper)/2
    upper=.5*np.linalg.eigvalsh(sc.Q)[:,-1]*min(sc.upper,1)+np.max(abs(sc.mu),axis=1)
    for j in range(M):
        add([ir+j],[1],(upper[j]-V[j])/u+1e-8)
        add([ir+j],[-1],-(sc.oracle_lower[j]-V[j])/u+1e-8)
        cols=[ir+j,iz+group[j]];vals=[env.scale[j],-1.]
        if hinge:
            cols.append(iy+j);vals.append(-1.);add([iy+j],[-1.],0.)
        if budget:cols.append(ig);vals.append(-env.evidence[j])
        add(cols,vals,env.offset[j]/u)
    if budget:add([ig],[-1.],0.)
    matrices.append(sp.csc_matrix((vv,(rr,cc)),shape=(nr,N)));rhs.append(np.array(bb));cones.append(clarabel.NonnegativeConeT(nr))
    # (t+1,t-1,sqrt(2)*Lw) belongs to SOC iff t >= .5 ||Lw||^2.
    W=np.zeros((M,d+2,d));W[:,:2,:]=-sc.mu[:,None,:]/u
    W[:,2:,:]=-np.sqrt(2/u)*np.linalg.cholesky(sc.Q).transpose(0,2,1)
    rows=(np.arange(M)[:,None]*(d+2)+np.arange(2)).reshape(-1)
    R=sp.csc_matrix((-np.ones(2*M),(rows,np.repeat(np.arange(M),2))),shape=(M*(d+2),M))
    soc=sp.hstack([sp.csc_matrix(W.reshape(-1,d)),R,sp.csc_matrix((M*(d+2),N-d-M))],format='csc')
    b=np.zeros((M,d+2));b[:,0]=V/u+1;b[:,1]=V/u-1
    matrices.append(soc);rhs.append(b.reshape(-1));cones.extend([clarabel.SecondOrderConeT(d+2) for _ in range(M)])
    A=sp.vstack(matrices,format='csc');b=np.concatenate(rhs)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=250
    settings.tol_gap_abs=2e-9;settings.tol_gap_rel=2e-9;settings.tol_feas=1e-10;settings.time_limit=time_limit
    answer=clarabel.DefaultSolver(sp.csc_matrix((N,N)),c,A,b,cones,settings).solve()
    x=np.array(answer.x);dual=np.array(answer.z)[1+nr:].reshape(M,d+2)
    q=np.maximum(dual[:,0]+dual[:,1],0)
    from .risk_center import project_envelope
    q=project_envelope(q,env)
    return feasible(x[:d],np.full(d,sc.upper)),q,float(answer.obj_val*u-(1-spec.blend)*(env.p@env.offset)),str(answer.status)
