"""Ten fixed instances: complete cold model setup versus reused scenario oracles."""
import json,time,psutil
import numpy as np
from .experiment_v2 import OUT,CFG,write_json,append_row
from .new_generators import family,truth_points,data_instance,scenarios,importance_scenarios,seed_for
from .scenario_inference import posterior
from .risk_envelopes import RiskSpec
from .risk_center import solve_center

def run():
    start=time.perf_counter();cpu=time.process_time();rows=[]
    manifest=json.loads((CFG/'pilot_manifest.json').read_text());unit=json.loads((OUT/'development_setup.json').read_text())['risk_unit']
    specs=[RiskSpec('A-Mean','mean',rho=1),RiskSpec('A-Tail-P:0.5'),RiskSpec('B-Tail-P:0.5',conditional=True),
        RiskSpec('B-Max','max',conditional=True),RiskSpec('C-Soft:0.1','soft'),RiskSpec('H-RankThenP','naive')]
    for r in manifest:
        f=family(r['kind'],r['family_seed']);Y=data_instance(f,truth_points(r['point_seed'])[r['point_id']],r['n'],r['m'],r['data_seed'])
        for spec in specs:
            tic=time.perf_counter()
            sc=(importance_scenarios(f,Y,256,seed_for('timing',r['family_seed'])) if r['kind']=='G2' else scenarios(f,128,seed_for('timing',r['family_seed'])))
            build=time.perf_counter()-tic;infer=time.perf_counter();p=posterior(sc,Y);infer=time.perf_counter()-infer
            ans=solve_center(sc,p,spec,unit);cold=time.perf_counter()-tic
            tic=time.perf_counter();again=solve_center(sc,p,spec,unit);cached=time.perf_counter()-tic
            rows.append(r|{'method':spec.name,'cold_total_seconds':cold,'setup_seconds':build,'inference_seconds':infer,
                'cached_decision_seconds':cached,'status':again.status,'gap':again.gap,'rss_bytes':psutil.Process().memory_info().rss,
                'cold_warm_weight_distance':float(np.linalg.norm(ans.weights-again.weights)),
                'timing_scope':'serial audit process on shared desktop while other experiment processes may run; excludes interpreter startup and data generation'})
        print('timing',len(rows)//len(specs),'/10',flush=True)
    write_json(OUT/'final_timing_audit.json',rows)
    append_row(OUT/'runtime_ledger.jsonl',{'stage':'final_timing','wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu,'unix':time.time()})

if __name__=='__main__':run()
