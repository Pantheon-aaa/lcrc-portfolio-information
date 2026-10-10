"""Audited E4 label correction only; preserves all numeric outcomes and old bytes."""
import gzip,hashlib,json
from .common import OUT,write,timed_stage

def run(budget):
    manifest=OUT/'audit_history'/'metadata_correction.json'
    if manifest.exists():return
    records=[]
    for p in sorted(OUT.glob('E4_*.jsonl')):
        if 'completed' in p.name:continue
        budget.check();old=p.read_bytes();data=[json.loads(s) for s in old.splitlines()]
        changed=0
        for r in data:
            assert r['epsilon_ratio']==.5
            spec=json.loads((OUT/'geometry'/f"{r['family']}.json").read_text())
            actual=(spec['gamma_lower']+spec['gamma_upper'])/2
            assert abs(r['epsilon']-actual)<1e-15
            r['epsilon_ratio']=1.;changed+=1
        backup=OUT/'audit_history'/(p.name+'.before_metadata_fix.gz')
        with gzip.GzipFile(filename=str(backup),mode='wb',mtime=0) as z:z.write(old)
        new=('\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in data)+'\n').encode('utf-8')
        p.write_bytes(new)
        records.append({'file':p.name,'rows':changed,'old_sha256':hashlib.sha256(old).hexdigest(),
          'new_sha256':hashlib.sha256(new).hexdigest(),'backup':str(backup.relative_to(OUT))})
    write(manifest,{'reason':'E4 epsilon was Gamma(all), but metadata incorrectly said epsilon_ratio=.5. Only this label changed; no observations, decisions, thresholds, or metrics changed.',
      'records':records})

if __name__=='__main__':timed_stage('confirmation',3600,run)
