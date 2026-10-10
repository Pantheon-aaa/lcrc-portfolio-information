"""Delivery inventory; fail on missing scheduled windows or silent solver failures."""
import hashlib
import importlib.metadata
import platform
import sys
import numpy as np
import pandas as pd
from .common import ROOT,PRIVATE,read,dump
from .run import tasks

def run():
    cfg=read(ROOT/'configs/base.json');inv={};allrecords=[]
    for stage in ('scores','tune','test','audit','fifty'):
        expected=tasks(stage);missing=[];failures=[];open_gap=[];count=0
        for task in expected:
            name=f"{task['row']['key']}_{task.get('size',256)}_{task.get('scramble',0)}.json"
            p=PRIVATE/'records'/stage/name
            if not p.exists():missing.append(name);continue
            r=read(p);allrecords.append(r)
            if r['status']!='ok':failures.append(name)
            for a in r.get('methods',[]):
                count+=1
                if a.get('weights') is None:failures.append(name+':'+a['method']);continue
                w=np.array(a['weights']);assert abs(w.sum()-1)<1e-8 and w.min()>-1e-10 and w.max()<=.2+1e-8
                if a['status']=='gap_open':open_gap.append(dict(key=r['key'],method=a['method'],gap_unit=a['gap']/cfg['risk_unit']))
        inv[stage]=dict(expected=len(expected),missing=missing,failures=failures,method_rows=count,open_gap=open_gap)
    freeze=read(ROOT/'configs/test_freeze.json');modified=[]
    for name,digest in freeze['source_hashes'].items():
        if name in ('model.py','data.py','book.py','run.py','common.py') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:modified.append(name)
    inv['decision_sources_modified_after_freeze']=modified
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','pyarrow','cvxpy','clarabel','scikit-learn','psutil','matplotlib']}
    dump(ROOT/'output/environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions,
        blas_threads=1,max_workers=4))
    ledger=[read(p) for p in (PRIVATE/'ledger').glob('*.json')]
    # Parent stage ledger includes child CPU; do NOT add individual task CPU again.
    inv['timing']=dict(measured_cpu_seconds=sum(r['cpu_seconds'] for r in ledger),
        sequential_stage_wall_seconds=sum(r['wall_seconds'] for r in ledger),
        wall_span_seconds=max(r['finished_unix'] for r in ledger)-min(r['finished_unix']-r['wall_seconds'] for r in ledger),
        task_wall_seconds=sum(r.get('seconds',0) for r in allrecords),
        scope='retained ingest, manifest, experiment-stage, LP-repair and execution-evaluation ledgers; excludes editing, short unmetered checks/reporting, and the first evaluation ledger overwritten by its diagnostic rerun',
        completeness='measured retained CPU is a lower bound, not complete machine-wide CPU accounting',
        ledger_overwrite_fix='new evaluation runs use unique time_ns names; the overwritten first evaluation duration cannot be recovered')
    action_gate=ROOT/'output/corporate_action_gate.json'
    inv['corporate_action_gate']=read(action_gate) if action_gate.exists() else None
    inv['corrected_ledger_present']=(PRIVATE/'evaluation/daily_test_actions.parquet').exists()
    if inv['corrected_ledger_present']:
        tx=pd.read_csv(ROOT/'output/terminal_test_actions.csv')
        inv['corrected_terminal_unresolved_max']=float(tx.unresolved_weight.max())
    dump(ROOT/'output/runtime_ledger.json',ledger)
    private_inventory=[]
    for folder in ['records','evaluation']:
        for p in sorted((PRIVATE/folder).rglob('*')):
            if p.is_file():private_inventory.append(dict(relative=str(p.relative_to(PRIVATE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(PRIVATE/'delivery_inventory.json',private_inventory)
    inv['private_inventory_sha256']=hashlib.sha256((PRIVATE/'delivery_inventory.json').read_bytes()).hexdigest()
    inv['private_files']=len(private_inventory)
    meta=read(PRIVATE/'cache/metadata.json')
    dump(ROOT/'output/data_fingerprint.json',{k:meta[k] for k in ['bytes','mtime_ns','rows','shape']})
    inv['complete']=not modified and all(not inv[s]['missing'] and not inv[s]['failures'] for s in ('scores','tune','test','audit','fifty'))
    dump(ROOT/'output/verification.json',inv)
    print({k:v for k,v in inv.items() if k not in ('scores','tune','test','audit','fifty')})
    assert inv['complete'],inv

if __name__=='__main__':run()
