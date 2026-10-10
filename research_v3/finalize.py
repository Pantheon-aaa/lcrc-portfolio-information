"""Read finalized ledgers and produce a precise accounting/provenance snapshot."""
import json,time,hashlib
from .common import ROOT,OUT,rows,write

def main():
    ledger=list(rows(OUT/'runtime_ledger.jsonl'));cpu=sum(r['cpu_seconds'] for r in ledger)
    starts=[r['unix']-r['wall_seconds'] for r in ledger];ends=[r['unix'] for r in ledger]
    intervals=sorted(zip(starts,ends));merged=[]
    for a,b in intervals:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    verification=json.loads((OUT/'delivery_verification.json').read_text())
    archive=json.loads((OUT/'archive_manifest.json').read_text())
    assert verification['passed'] and archive['verified_lossless']
    stages={stage:sum(r['cpu_seconds'] for r in ledger if r['stage']==stage) for stage in sorted({r['stage'] for r in ledger})}
    config=json.loads((ROOT/'configs'/'protocol.json').read_text())
    assert cpu<21600 and all(stages[s]<config['cpu_limits'][s] for s in stages)
    result={'recorded_cpu_seconds':cpu,'recorded_cpu_hours':cpu/3600,
      'sum_process_wall_seconds':sum(r['wall_seconds'] for r in ledger),
      'observed_wall_span_seconds':max(ends)-min(starts),'union_active_wall_seconds':sum(b-a for a,b in merged),
      'stage_cpu_seconds':stages,'all_main_rows':sum(x['rows'] for x in verification['counts'].values()),
      'verification_passed':True,'lossless_archives_verified':True,
      'compressed_bytes':sum(r['compressed_bytes'] for r in archive['records']),
      'accounting_limit':'Ledger covers timed experiment, calibration, audit, report and packaging processes. Initial environment checks, interactive debugging, Python import startup, and this short snapshot are not fully metered. Observed wall span includes idle gaps, excludes work before the first ledger event.',
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}}
    write(OUT/'final_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
