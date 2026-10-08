import json,time
import numpy as np
from .new_generators import make_block_family,scenarios,data_instance
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec,Envelope
from .risk_center import bounds,solve_center,conic_reference
from .native_conic import solve
from .experiment_v2 import OUT,write_json

def run():
    records=[]
    for seed in (981,982):
        f=make_block_family(seed,d=6);sc=scenarios(f,128,seed,upper=.5)
        p=posterior(sc,data_instance(f,np.zeros(3),64,16,seed+4));u=.1
        for spec in [RiskSpec('tail'),RiskSpec('ctail',conditional=True),RiskSpec('max','max'),
            RiskSpec('cmax','max',conditional=True),RiskSpec('filter','filter'),RiskSpec('cf','filter',conditional=True),
            RiskSpec('budget','budget',reference=True),RiskSpec('absolute',absolute=True),RiskSpec('lp','max',penalty=.1),RiskSpec('blend',blend=.5)]:
            e=Envelope(sc,p,spec,u);w,q,val,status=solve(sc,e);L,U,_=bounds(sc,e,w,q,-q@e.offset)
            wc,qc,ref,st=conic_reference(sc,e)
            assert L-2e-7*u<=ref<=U+2e-7*u,(spec.name,L,U,ref)
            fallback=None
            if U-L>=1e-6*u:
                recovered=solve_center(sc,p,spec,u)
                assert recovered.gap<1e-6*u,(spec.name,recovered.gap)
                fallback={'status':recovered.status,'gap':recovered.gap,'diagnostics':recovered.diagnostics}
            lo,hi,A,b=e.dual_constraints()
            assert np.max(abs(A@q-b))<1e-10 and np.min(q-lo)>-1e-12 and np.max(q-hi)<1e-12
            records.append({'seed':seed,'method':spec.name,'gap':U-L,'reference_difference':val-ref,
                'weight_distance':np.linalg.norm(w-wc),'status':status,'fallback':fallback})
    write_json(OUT/'native_conic_gate.json',{'passed':True,'checks':records});print('native conic independent checks passed')

if __name__=='__main__':run()
