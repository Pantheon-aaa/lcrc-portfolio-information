from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linprog, brentq


@dataclass(frozen=True)
class RiskSpec:
    name: str
    kind: str = 'tail'
    rho: float = .5
    conditional: bool = False
    reference: bool = False
    blend: float = 1.
    penalty: float = 0.
    temperature: float = .1
    radius: float = .1
    budget_fraction: float = .5
    absolute: bool = False


def tail_weights(values,p,rho):
    p=np.asarray(p); p=p/p.sum()
    if rho>=1: return p.copy()
    order=np.argsort(-values,kind='stable'); cap=p[order]/rho
    q=np.minimum(cap,np.maximum(0,1-np.r_[0,np.cumsum(cap[:-1])]))
    out=np.zeros(len(p)); out[order]=q
    # Remove accumulated roundoff, preserving support and nonnegative weights.
    out/=out.sum()
    return out


class Envelope:
    """Fixed-data convex risk functional plus its adversarial maximizing weights."""
    def __init__(self,sc,post,spec,unit):
        self.sc=sc; self.post=post; self.spec=spec; self.unit=unit
        self.groups=sc.groups; self.n=len(sc.Q)
        self.alpha=np.exp(post.log_alpha)
        self.lp=post.log_joint.copy()
        if spec.reference:
            self.lp=sc.log_reference.copy()
            if spec.conditional:
                for s in np.unique(self.groups):
                    ix=self.groups==s
                    self.lp[ix]=self.lp[ix]-logsumexp(self.lp[ix])+post.log_alpha[s]
        self.p=np.exp(self.lp)
        self.offset=spec.penalty*unit*(post.loglik.max()-post.loglik)
        self.scale=np.ones(self.n)
        self.mode=spec.kind
        self.block_rhos={}
        if spec.kind=='count':
            for ix,mass in self.blocks(): self.p[ix]=mass/len(ix)
        if spec.kind=='filter':
            # Local or global filtering followed by count-based tail.
            for ix,mass in self.blocks():
                if mass==0: continue
                local=self.p[ix]/self.p[ix].sum(); order=np.argsort(-local,kind='stable')
                k=min(len(order),int(np.searchsorted(np.cumsum(local[order]),.95))+1)
                keep=order[:k]; new=np.zeros(len(ix)); new[keep]=mass/k; self.p[ix]=new
                self.block_rhos[int(ix[0])]=np.ceil(spec.rho*k)/k
        if spec.kind=='sample':
            rng=np.random.default_rng(89125)
            for ix,mass in self.blocks():
                if mass==0: continue
                draw=rng.choice(len(ix),size=len(ix),p=self.p[ix]/self.p[ix].sum())
                self.p[ix]=np.bincount(draw,minlength=len(ix))*mass/len(ix)
        if spec.kind=='pr':
            for ix,mass in self.blocks():
                if mass==0: continue
                self.scale[ix]=self.p[ix]/mass
                self.p[ix]=mass/len(ix)
        self.budget=None
        if spec.kind=='budget':
            caps=self.p/spec.rho
            d=post.loglik.max()-post.loglik
            low=linprog(d,A_eq=np.ones((1,self.n)),b_eq=[1.],bounds=list(zip(np.zeros(self.n),caps)),method='highs')
            if not low.success: raise ValueError('empty evidence envelope')
            self.budget=float(low.fun+spec.budget_fraction*(self.p@d-low.fun)); self.evidence=d
            self.budget_anchor=low.x

    def blocks(self):
        if self.spec.conditional:
            return [(np.flatnonzero(self.groups==s),self.alpha[s]) for s in np.unique(self.groups)]
        return [(np.arange(self.n),1.)]

    def block_rho(self,ix):
        return self.block_rhos.get(int(ix[0]),self.spec.rho)

    def support(self,values):
        """Return value, gradient wrt original regrets, and additive dual term."""
        self.support_error=0.
        v=self.scale*np.asarray(values)-self.offset; q=np.zeros(self.n); additive=0.
        for ix,mass in self.blocks():
            if mass==0: continue
            p=self.p[ix]/mass; x=v[ix]
            if self.mode=='mean': r=p
            elif self.mode=='max':
                r=np.zeros(len(ix)); r[int(np.argmax(x))]=1
            elif self.mode=='soft':
                tau=self.spec.temperature*self.unit
                lp=self.lp[ix]-logsumexp(self.lp[ix]); z=lp+x/tau
                lz=logsumexp(z); r=np.exp(z-lz)
                additive+=mass*(tau*lz-r@x)
            elif self.mode=='lr':
                # Reverse-KL sphere, full positive reference support.
                shift=x.max(); y=(x-shift)/max(self.unit,1e-14)
                if np.ptp(y)<1e-14: r=p
                else:
                    def candidate(eps):
                        a=p/(eps-y); return a/a.sum()
                    def distance(eps):
                        a=candidate(eps); return float(np.sum(p*np.log(p/a)))
                    lo=1e-15; hi=1.
                    while distance(hi)>self.spec.radius: hi*=2
                    if distance(lo)<=self.spec.radius:
                        raise FloatingPointError('reverse-KL endpoint needs higher precision')
                    eps=brentq(lambda e:distance(e)-self.spec.radius,lo,hi,xtol=1e-14)
                    r=candidate(eps)
            elif self.mode=='budget':
                fit=linprog(-x,A_ub=self.evidence[None,:],b_ub=[self.budget],
                    A_eq=np.ones((1,len(x))),b_eq=[1.],bounds=list(zip(np.zeros(len(x)),p/self.spec.rho)),method='highs')
                if not fit.success: raise RuntimeError(fit.message)
                r=fit.x
                # A feasible maximizing LP point gives a lower support value.
                # Repair an LP dual point to obtain an independent upper value.
                gamma=max(0.,-float(fit.ineqlin.marginals[0]));z=-float(fit.eqlin.marginals[0])
                slack=np.maximum(x-z-gamma*self.evidence[ix],0.)
                upper=z+gamma*self.budget+(p/self.spec.rho)@slack
                cushion=5e-12*(1+abs(z)+abs(gamma*self.budget)+np.abs((p/self.spec.rho)*slack).sum())
                self.support_error+=mass*max(0.,upper-float(r@x)+cushion)
            elif self.mode=='naive':
                k=max(1,int(np.ceil(self.spec.rho*len(x)))); order=np.argsort(-x,kind='stable')[:k]
                r=np.zeros(len(x)); r[order]=p[order]/p[order].sum()
            else: r=tail_weights(x,p,self.block_rho(ix))
            if self.spec.blend<1: r=(1-self.spec.blend)*p+self.spec.blend*r
            q[ix]=mass*r
        additive-=float(q@self.offset)
        gradient=q*self.scale
        return float(gradient@values+additive),gradient,float(additive)

    def dual_constraints(self):
        """SLSQP q constraints for exact polyhedral mean/tail/max/penalty envelopes."""
        spec=self.spec
        if self.mode not in ('tail','max','mean','budget','filter','sample','count'): return None
        lower=(1-spec.blend)*self.p
        cap=np.ones(self.n) if self.mode=='max' else self.p/spec.rho
        if self.mode=='filter':
            for ix,mass in self.blocks(): cap[ix]=self.p[ix]/self.block_rho(ix)
        if self.mode=='mean': cap=self.p; lower=self.p.copy()
        cap=(1-spec.blend)*self.p+spec.blend*cap
        blocks=self.blocks(); A=np.zeros((len(blocks),self.n)); rhs=[]
        for j,(ix,mass) in enumerate(blocks): A[j,ix]=1; rhs.append(mass)
        return lower,np.minimum(cap,1),A,np.array(rhs)
