"""Lossless local Git packaging; never uploads or deletes raw experiment records."""
import gzip,hashlib,shutil,time,json
from .common import OUT,write,timed_stage

def run(budget):
    records=[]
    for p in sorted(OUT.glob('*.jsonl')):
        if p.name=='runtime_ledger.jsonl':continue
        budget.check();out=p.with_suffix(p.suffix+'.gz')
        rawhash=hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open('rb') as src,out.open('wb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0,compresslevel=6) as z:shutil.copyfileobj(src,z)
        with gzip.open(out,'rb') as f:restoredhash=hashlib.sha256(f.read()).hexdigest()
        if restoredhash!=rawhash:raise RuntimeError('archive mismatch')
        records.append({'raw':p.name,'compressed':out.name,'raw_bytes':p.stat().st_size,'compressed_bytes':out.stat().st_size,'sha256':rawhash})
    write(OUT/'archive_manifest.json',{'verified_lossless':True,'records':records})
    print(f'Lossless local archives verified: {len(records)} files',flush=True)

if __name__=='__main__':timed_stage('confirmation',3600,run)
