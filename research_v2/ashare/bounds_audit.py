"""Analytic lower bound for global likelihood-penalized maximum regret.

For an ML atom j*, d[j*]=0 and R[j*](w)>=0 for EVERY feasible w.
Thus min_w max_j (R[j](w)-lambda*d[j]) >= 0. This tightens only a
certificate; weights, evaluations, hyperparameters, and original bounds remain saved.
"""
import shutil
from .common import ROOT,PRIVATE,read,dump

def run():
    unit=read(ROOT/'configs/base.json')['risk_unit'];changes=[]
    for stage in ('timing','tune','test','audit','fifty'):
        archive=PRIVATE/'records_archive'/f'{stage}_before_analytic_bound'
        for p in (PRIVATE/'records'/stage).glob('*.json'):
            r=read(p);updated=False
            for a in r.get('methods',[]):
                if a['method']!='C-LP-Max' or a.get('lower') is None or a.get('analytic_lower_bound_applied'):continue
                old=dict(lower=a['lower'],upper=a['upper'],gap=a['gap'],status=a['status'])
                if a['lower']<0 and a['upper']>=0:
                    if not updated:
                        archive.mkdir(parents=True,exist_ok=True)
                        if not (archive/p.name).exists():shutil.copy2(p,archive/p.name)
                    a['lower']=0.;a['gap']=a['upper']
                    a['status']='converged' if a['gap']<=1e-6*unit and a.get('residual',0)<=1e-8 else 'gap_open'
                    a['original_solver_bound']=old;a['analytic_lower_bound_applied']=True;updated=True
                    changes.append(dict(stage=stage,key=r['key'],old_gap_unit=old['gap']/unit,new_gap_unit=a['gap']/unit,
                                        old_status=old['status'],new_status=a['status']))
            if updated:dump(p,r)
    dump(ROOT/'output/analytic_bound_audit.json',dict(proof='ML atom has zero evidence penalty and nonnegative true regret for every feasible decision',
        modifies_weights=False,modifies_evaluations=False,changes=changes))
    print('bounds tightened',len(changes),'open gaps closed',sum(x['old_status']=='gap_open' and x['new_status']=='converged' for x in changes))

if __name__=='__main__':run()
