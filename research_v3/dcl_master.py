from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from .compatible_qcqp import Compatibility

class DCL:
    def __init__(self,compatibility,epsilon):
        self.comp=compatibility;self.fam=compatibility.fam;self.epsilon=epsilon;self.cuts=[]
    def solve(self,p,time_limit=30.):
        start=perf_counter();M=self.fam.M;node_count=0;iterations=0
        # Certified singletons provide an incumbent even when a larger set is indeterminate.
        order=sorted(range(M),key=lambda j:(-p[j],j));best=None
        for j in order:
            state,r=self.comp.classify([j],self.epsilon)
            if state=='compatible':best=(float(p[j]),(j,),r);break
        if best is None:return dict(status='indeterminate',w=self.fam.mean(p)['w'],mass_lower=0.,mass_upper=1.,seconds=perf_counter()-start)
        upper=1.;status='timeout';tie_gap=None
        while perf_counter()-start<time_limit:
            n=len(self.cuts);A=np.zeros((n,M));rhs=[]
            for k,C in enumerate(self.cuts):A[k,list(C)]=1;rhs.append(len(C)-1)
            con=LinearConstraint(A,-np.inf,np.array(rhs)) if n else None
            options={'time_limit':max(.01,time_limit-(perf_counter()-start)),'mip_rel_gap':0.}
            opt=milp(-p,integrality=np.ones(M),bounds=Bounds(np.zeros(M),np.ones(M)),constraints=con,options=options)
            iterations+=1;node_count+=int(getattr(opt,'mip_node_count',0) or 0)
            if opt.x is None:status='milp_timeout';break
            upper=min(upper,float(-opt.mip_dual_bound)+1e-10)
            z=opt.x>.5;G=tuple(np.flatnonzero(z));objective=float(p@z)
            # Lexicographic subset tie break, only within floating mass tolerance.
            if opt.success:
                At=np.vstack([A,p]);lb=np.r_[np.full(n,-np.inf),objective-1e-12];ub=np.r_[rhs,np.inf]
                lex=milp(-2.**np.arange(M-1,-1,-1),integrality=np.ones(M),bounds=Bounds(np.zeros(M),np.ones(M)),
                    constraints=LinearConstraint(At,lb,ub),options=options)
                if lex.success:G=tuple(np.flatnonzero(lex.x>.5))
            state,r=self.comp.classify(G,self.epsilon)
            if state=='compatible':
                mass=float(p[list(G)].sum())
                if mass>=best[0]-1e-12:best=(mass,G,r)
                status='optimal' if upper-best[0]<=2e-9 else 'open_mass_gap'
                break
            if state=='indeterminate':status='indeterminate';break
            C=list(G)
            for j in G:
                if len(C)<=1:break
                trial=[k for k in C if k!=j]
                trial_state,_=self.comp.classify(trial,self.epsilon)
                if trial_state=='incompatible':C=trial
            cut=tuple(C)
            if cut in self.cuts:raise RuntimeError('duplicate violated conflict cut')
            self.cuts.append(cut)
        mass,G,center=best
        tied=self.comp.witness(G,p,self.epsilon,center)
        lo,hi=self.fam.regret_bounds(tied.w)
        return dict(status=status,w=tied.w,mass_lower=float(p[hi<=self.epsilon].sum()),
          mass_upper=max(float(upper),float(p[hi<=self.epsilon].sum())),selected=G,
          seconds=perf_counter()-start,cuts=len(self.cuts),cut_sets=self.cuts.copy(),
          iterations=iterations,milp_nodes=node_count,tie_gap=tied.gap,
          selected_max_regret_upper=float(hi[list(G)].max()),compatibility_gap=center.gap)

def enumerate_exact(comp,p,epsilon):
    M=comp.fam.M;best=(0.,(),None);unresolved=0.
    for size in range(1,M+1):
        for ids in combinations(range(M),size):
            state,r=comp.classify(ids,epsilon);mass=float(p[list(ids)].sum())
            if state=='compatible' and (mass>best[0]+1e-12 or (abs(mass-best[0])<=1e-12 and ids<best[1])):best=(mass,ids,r)
            elif state=='indeterminate':unresolved=max(unresolved,mass)
    return best, max(best[0],unresolved)

def greedy(comp,p,epsilon):
    ids=list(range(comp.fam.M))
    while ids:
        state,r=comp.classify(ids,epsilon)
        if state=='compatible':return comp.witness(ids,p,epsilon,r)
        ids.remove(min(ids,key=lambda j:(p[j],-j)))
    raise RuntimeError('no certified singleton')
