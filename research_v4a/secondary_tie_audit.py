"""Non-symmetric equal-mass subsets test global, rather than within-subset, ties."""
import numpy as np
from .common import OUT,write,run
from .families import make
from .mean_budget import Family
from .price_oracle import Geometry
from .budgeted_dcl import solve_budgeted_dcl,enumerate_frontier
from .api import solve_budgeted_dcl as public_solve


def execute(budget):
    checks=[]
    for kind in ('asymmetric','hidden_curvature'):
        budget.check();f=make(kind,10,6,'gate',0);p=np.ones(6)/6;g=Geometry(f,0).gamma(range(6));eps=.25*(g.lower+g.upper)
        answers,pc=enumerate_frontier(f,p,eps,[.1,.25,np.inf])
        for beta,reference in zip((.1,.25,np.inf),answers):
            a=solve_budgeted_dcl(f,p,eps,beta);mass=reference['optimal_mass_lower'];bl,bu=pc.mb.bounds(beta)
            candidates=[(G,r) for G,r in pc.cache.items() if r.epsilon_feasible and r.cost_upper<=bl and abs(sum(p[list(G)])-mass)<=1e-12]
            certified=a['primary_mass_gap']<=1e-8 and np.isfinite(a['secondary_price_gap'])
            assert a['optimal_mass_lower']<=reference['optimal_mass_upper']+1e-8 and reference['optimal_mass_lower']<=a['optimal_mass_upper']+1e-8
            if certified:
                assert abs(a['mean_increment_upper']-reference['mean_increment_upper'])<=1e-6*pc.mb.Lhi
            checks.append(dict(kind=kind,beta=beta,status=a['status'],same_mass_feasible_subsets=len(candidates),candidate_price_upper_span=float(np.ptp([r.cost_upper for G,r in candidates])) if candidates else 0.,Lstar=pc.mb.Lhi,solver_price=[a['mean_increment_lower'],a['mean_increment_upper']],enumeration_price=[reference['mean_increment_lower'],reference['mean_increment_upper']],global_secondary_certified=certified))
    shared=Family(np.stack([np.eye(6),np.eye(6)]),.2);a=public_solve(shared,[.5,.5],0.,.1)
    assert a['structural_certificate']['certified'] and a['budget_lower']==a['budget_upper']==0
    assert a['posterior_mean_regret_MA_lower']==a['posterior_mean_regret_MA_upper']==0
    write(OUT/'secondary_tie_audit.json',dict(passed=True,checks=checks,structural_zero_case=a))
    print('Global equal-mass secondary ties and structural zero case checked',flush=True)


if __name__=='__main__':run('gate_secondary_ties',execute)
