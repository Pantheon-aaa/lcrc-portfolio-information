from pathlib import Path
import os,json,hashlib,time,gzip
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
RUN=Path(os.environ.get('V6_RUN_ROOT',str(ROOT/'runs')))
BASE='d07e06b739ad2fce00aa35e4b233e19fcb352231'
KINDS=('asymmetric','hidden_curvature','block')
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'inf' if x>0 else '-inf' if x<0 else None
    return x
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)
def read(p):
    p=Path(p)
    if not p.exists() and Path(str(p)+'.gz').exists():
        with gzip.open(str(p)+'.gz','rt',encoding='utf8') as z:return json.load(z)
    return json.loads(p.read_text(encoding='utf8'))
def append(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:f.write(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'))+'\n')
def rows(p):
    if Path(p).exists():
        with Path(p).open(encoding='utf8') as f:
            for l in f:
                if l.strip():yield json.loads(l)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seed(*a):return int.from_bytes(hashlib.sha256(('v6-20261013|'+'|'.join(map(str,a))).encode()).digest()[:8],'little')
def sources():return {p.name:sha(p) for p in ROOT.glob('*.py')}
def used():return sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl'))
def check_budget(start,allowance):
    if time.process_time()-start>allowance or used()+time.process_time()>34500:raise TimeoutError('optimization CPU allowance')
