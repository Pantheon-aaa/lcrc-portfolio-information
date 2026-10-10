from time import perf_counter
import numpy as np
from scipy.special import logsumexp
from .compatible_qcqp import solve,Result

def common(fam,comp,ll,delta=.05):
    p=fam.posterior(ll);start=perf_counter();a=fam.mean(p);lo,hi=fam.regret_bounds(a['w'])
    out={'MA':Result(a['w'],float(p@lo)-a['gap'],float(p@hi),'solved',perf_counter()-start,{'qp_gap':a['gap']})}
    out['CVaR']=solve(fam,p,'tail');out['Minimax']=comp.gamma(range(fam.M))
    j=int(np.argmax(ll));w=fam.oracle[j]
    out['MLE']=Result(w,0.,float(fam.regret_bounds(w)[1][j]),'oracle',0.,{'chosen_model':j})
    logq=logsumexp(ll-np.log(fam.M));keep=np.flatnonzero(ll>=logq+np.log(delta))
    out['LCRC']=comp.gamma(keep)
    return p,out

def evaluate(fam,w,p,epsilon,truth):
    lo,hi=fam.regret_bounds(w);good=hi<=epsilon;bad=lo>epsilon
    return dict(weights=w,regret_lower=float(lo[truth]),regret_upper=float(hi[truth]),regret=float((lo[truth]+hi[truth])/2),
      true_success=bool(good[truth]),true_possible_success=bool(not bad[truth]),
      success_indeterminate=bool(not good[truth] and not bad[truth]),posterior_mass_lower=float(p[good].sum()),
      posterior_mass_upper=float(p[~bad].sum()),posterior_mean_lower=float(p@lo),posterior_mean_upper=float(p@hi),
      residual=max(abs(w.sum()-1),float(-w.min()),float(w.max()-fam.upper),0.))
