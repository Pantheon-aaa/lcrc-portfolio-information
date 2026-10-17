"""Independent numerical checks used at final audit, not another method."""
from fractions import Fraction as F
import numpy as np
import mpmath as mp
from scipy.stats import chi2
from .common import *

def run():
    from .pipeline import load,run_job,jobkey
    from research_v6.certificates import Geometry,quad,mat,up,down
    import cvxpy as cp
    checks=[];mp.mp.dps=70
    # Re-evaluate the chi-square inverse using the gamma CDF in high precision.
    # This is an error diagnostic, not a formal interval special-function proof.
    for d in (10,12):
      alpha=.05/(3*(2*d*d+2*d))
      for n in (16,128,1024):
       for prob in (alpha/2,1-alpha/2):
        x=chi2.ppf(prob,n);cdf=mp.gammainc(mp.mpf(n)/2,0,mp.mpf(float(x))/2,regularized=True)
        error=float(abs(cdf-mp.mpf(float(prob))))
        checks.append(dict(name='chi2_inverse',d=d,n=n,probability=prob,absolute_CDF_error=error,passed=error<1e-12))
    names=sorted({j['family'] for j in read(ROOT/'manifest.json.gz')})
    def positive_definite(Q):
        A=mat(Q);pivots=[]
        for k in range(len(A)):
            pivot=A[k][k];pivots.append(pivot)
            if pivot<=0:return False,min(pivots)
            for i in range(k+1,len(A)):
                for j in range(i,len(A)):
                    A[i][j]-=A[i][k]*A[k][j]/pivot;A[j][i]=A[i][j]
        return True,min(pivots)
    for name in names:
        raw,lib,truths=load(name);f,g=lib['all'];stored=raw['libraries']['all']['gamma'];rw=[F(x) for x in stored['rational_weights']]
        from .models import make
        rebuilt=make(raw['kind'],raw['d'],raw['index'],raw['split'])
        checks.append(dict(name='frozen_generator_reproduces_library',family=name,passed=np.array_equal(rebuilt.Q,f.Q),family_file_sha256=sha(RUN/'families'/f'{name}.json')))
        pd=[positive_definite(q) for q in f.Q]
        checks.append(dict(name='exact_rational_positive_definiteness',family=name,passed=all(x[0] for x in pd),minimum_LDL_pivot=float(min(x[1] for x in pd))))
        actual=max(x[1] for x in g.regret(rw));checks.append(dict(name='rational_center_regret',family=name,passed=up(actual)<=stored['upper']))
        w=cp.Variable(f.d);t=cp.Variable();cons=[cp.sum(w)==1,w>=0,w<=f.upper]
        # Independently build the convex QCQP. It checks a reference numerical
        # value against the rational bracket, not formal validity of CVXPY.
        for j in range(f.M):cons.append(.5*cp.quad_form(w,f.Q[j])-f.V[j]<=t)
        problem=cp.Problem(cp.Minimize(t),cons);problem.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11)
        checks.append(dict(name='independent_CVXPY_gamma',family=name,value=problem.value,status=problem.status,lower=stored['lower'],upper=stored['upper'],passed=stored['lower']-1e-9<=problem.value<=stored['upper']+1e-9))
        # Evaluator regret is checked with separate exact-rational weak duality.
        for ci in (0,6,7,13,16,19):
            c=raw['cases'][ci];tf=truths[c['id']];exact=Geometry(tf);library=raw['libraries'][c['library']]['gamma'];r=[F(x) for x in library['rational_weights']];lo,hi=exact.regret(r)[0]
            flo,fhi=tf.regret_bounds(np.array(library['weights']))
            checks.append(dict(name='truth_regret_independent_rational',family=name,case=c['id'],rational_lower=down(lo),rational_upper=up(hi),floating_lower=flo[0],floating_upper=fhi[0],passed=flo[0]<=up(hi)+1e-12 and fhi[0]>=down(lo)-1e-12))
    # Check the float pilot reference's separation from its log threshold and
    # reconstruct every saved confidence set. This does not upgrade it to the
    # original v6 interval implementation or a software-level formal proof.
    from .models import observations
    from .pipeline import block_ll
    from scipy.special import logsumexp
    margin=np.inf;near=[];mismatch=[];count=0
    for path in (RUN/'confirmation').glob('*.json.gz'):
        r=read(path)
        if r['status']!='complete':continue
        j=r['job'];raw,lib,truths=load(j['family']);c=raw['cases'][j['case_index']];f,_=lib[c['library']]
        X=observations(truths[c['id']].Q[0],1024,j['design'],r['pilot_seed']);cum=np.cumsum(block_ll(X,f.Q,j['design']),axis=0);loge=logsumexp(cum,axis=1)[:,None]-np.log(f.M)-cum
        for a in r['rows']:
            if a['method']!='v6_pilot_CS' or a['ratio']!=.5:continue
            e=np.max(loge[:a['n']//2],axis=0);distance=float(np.min(abs(e-np.log(20))));margin=min(margin,distance)
            if distance<1e-7:near.append(dict(job=j,n=a['n'],log_threshold_margin=distance))
            if np.flatnonzero(e<np.log(20)).tolist()!=a['confidence_set']:mismatch.append(dict(job=j,n=a['n']))
        count+=1
    write(OUT/'pilot_likelihood_audit.json',dict(datasets=count,minimum_log_threshold_margin=margin,review_threshold=1e-7,near=near,mismatch=mismatch,qualification='margin diagnostic and exact replay of the floating reference; not formal interval likelihood'))
    checks.append(dict(name='pilot_reference_reconstruction',passed=not mismatch and not near,datasets=count,minimum_log_threshold_margin=margin))
    write(OUT/'independent_numerical_audit.json',dict(passed=all(c['passed'] for c in checks),checks=checks,qualification='rational geometry plus independent convex reference; high precision special-function residuals are diagnostics'))
    return all(c['passed'] for c in checks)
