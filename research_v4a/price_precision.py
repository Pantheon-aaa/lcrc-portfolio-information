"""Independent 80-digit weak-duality price audit, with no tolerance relaxation."""
import numpy as np
import mpmath as mp
from .mean_budget import MeanBudget
from .precision_audit import exact_qp

class MPPrices:
    def __init__(self,f,p):
        mp.mp.dps=80;self.f=f;self.mb=MeanBudget(f,p);self.p=[mp.mpf(x) for x in self.mb.p];self.Q=[mp.matrix(q.tolist()) for q in f.Q]
        self.V=[exact_qp(Q,w,f.upper)[0] for Q,w in zip(self.Q,f.oracle)]
        self.Qbar=sum((pj*Q for pj,Q in zip(self.p,self.Q)),mp.matrix(f.d));self.Fstar,_=exact_qp(self.Qbar,self.mb.w,f.upper);self.Lstar=self.Fstar-sum(pj*v for pj,v in zip(self.p,self.V))
    def project(self,w):
        w=mp.matrix(np.asarray(w,float).tolist());i=max(range(self.f.d),key=lambda j:min(w[j],mp.mpf(self.f.upper)-w[j]));w[i]+=1-sum(w);return w
    def feasible(self,w,G,eps):
        return all(0<=x<=mp.mpf(self.f.upper) for x in w) and abs(sum(w)-1)<mp.mpf('1e-70') and all((w.T*self.Q[j]*w)[0]/2-self.V[j]<=mp.mpf(eps) for j in G)
    def check(self,G,eps,r,center):
        q=[mp.mpf(x) for x in r.diagnostics['dual_weights']];mix=self.mb.p.copy();mix[list(G)]+=np.array([float(x) for x in q]);warm=self.f.mean(mix/mix.sum())['w']
        Qmix=self.Qbar+sum((lam*self.Q[j] for lam,j in zip(q,G)),mp.matrix(self.f.d));value,_=exact_qp(Qmix,warm,self.f.upper)
        low=max(mp.mpf(0),value-self.Fstar-sum(lam*(self.V[j]+mp.mpf(eps)) for lam,j in zip(q,G)))
        w=self.project(r.witness_weights)
        if not self.feasible(w,G,eps):
            c=self.project(center)
            if not self.feasible(c,G,eps):return dict(status='high_precision_primal_open',dual_lower=str(low),Lstar=str(self.Lstar),certificate_consistent=None)
            left=mp.mpf(0);right=mp.mpf(1)
            for _ in range(100):
                a=(left+right)/2
                if self.feasible((1-a)*c+a*w,G,eps):left=a
                else:right=a
            w=(1-left)*c+left*w
        up=(w.T*self.Qbar*w)[0]/2-self.Fstar
        # A numerical certificate is inconsistent only if it excludes an independently
        # feasible primal/dual interval, not merely an inaccurate CVXPY point value.
        violation=max(mp.mpf(r.cost_lower)-up,low-mp.mpf(r.cost_upper),mp.mpf(0));roundoff_bound=mp.mpf('1e-65')
        gap=up-low;normalized=gap/self.Lstar if self.Lstar>0 else mp.inf
        return dict(status='high_precision_verified' if normalized<=mp.mpf('1e-6') else 'high_precision_price_open',dual_lower=str(low),primal_upper=str(up),Lstar=str(self.Lstar),normalized_gap=str(normalized),certificate_consistent=violation<=roundoff_bound,interval_exclusion=str(violation),primal_weights=[str(x) for x in w],dual_multipliers=[str(x) for x in q])
