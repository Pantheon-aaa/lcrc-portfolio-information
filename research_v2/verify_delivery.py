"""Required manifests, method coverage, feasibility and recorded scientific gates."""
import json,collections,hashlib
import numpy as np
from .experiment_v2 import ROOT,OUT,CFG,load_rows,write_json
from .registry import all_specs,core_specs

def run():
    selection=json.loads((CFG/'selection.json').read_text())
    screen={s.name for s in core_specs()}|{s['name'] for s in selection['extra_screen_specs']}|{'REF-EB','B-Plugin'}
    confirm={s['name'] for s in selection['confirmation_specs']}
    expected={'development':{s.name for s in all_specs()},'G1':screen,'G2':screen,'confirm':confirm,
              'G3_d20':confirm-{'REF-EB'}|{'Equal-Weight','EM-shrink'},'G3_d50':confirm-{'REF-EB'}|{'Equal-Weight','EM-shrink'}}
    repair=json.loads((CFG/'selection_repair.json').read_text())
    extra=set(repair['tradeoff_candidates'])
    expected.update(blend_development={'A-Blend','B-Blend'},repair_screen=extra|{'A-Blend','B-Blend'},repair_confirmation=extra-confirm)
    checks=[]
    for stage,names in expected.items():
        source={'blend_development':['development'],'repair_screen':['G1','G2'],'repair_confirmation':['confirm']}.get(stage,[stage])
        manifest=[r for s in source for r in json.loads((CFG/f'{s}_manifest.json').read_text())];scheduled={r['instance_id'] for r in manifest}
        rows=load_rows(stage);keys=[(r['instance_id'],r['method']) for r in rows]
        assert len(keys)==len(set(keys)),(stage,'duplicate row')
        assert set(keys)=={(i,n) for i in scheduled for n in names},(stage,'incomplete manifest/method pairs',len(keys),len(scheduled)*len(names))
        failures=[];open_gaps=[];residuals=[]
        for r in rows:
            if r['failure']:failures.append((r['instance_id'],r['method'],r.get('error')));continue
            w=np.array(r['weights']);res=max(abs(w.sum()-1),max(-w.min(),0),max(w.max()-.2,0));residuals.append(res)
            assert np.isfinite(w).all() and res<1e-7
            assert r['regret_scaled']>=-1e-6
            if r.get('status')=='converged':
                assert r['gap']<=1.01e-6*r.get('unit',.03827751737525126)
                assert r['gap']>=-1e-8*r.get('unit',.03827751737525126)
            if r.get('status')=='gap_open':open_gaps.append((r['instance_id'],r['method'],r['gap']))
        checks.append({'stage':stage,'instances':len(scheduled),'methods':len(names),'rows':len(rows),
            'failures':failures,'open_gaps':open_gaps,'max_feasibility_residual':max(residuals,default=None)})
    gates={}
    for name in ('gate_v2','native_conic_gate','embedded_boundary_gate','theory_formula_checks'):
        gate=json.loads((OUT/(name+'.json')).read_text());assert gate['passed'];gates[name]=True
    ledger=load_rows('runtime_ledger');total=sum(r['wall_seconds'] for r in ledger)
    assert total<86400,('recorded budget exceeded',total)
    write_json(OUT/'delivery_validation.json',{'passed':True,'checks':checks,'gates':gates,
        'recorded_experiment_process_wall_hours':total/3600,
        'interpretation':'complete scheduled rows; does not assert every solver converged or any algorithm won'})
    print('Delivery checks passed:',[(c['stage'],c['rows'],len(c['failures']),len(c['open_gaps'])) for c in checks])

if __name__=='__main__':run()
