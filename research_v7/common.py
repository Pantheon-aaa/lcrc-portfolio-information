from pathlib import Path
import os,json,gzip,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
RUN=Path(os.environ.get('V7_RUN_ROOT',r'D:\Portfolio_Optimization_paper\private_runs\integrated_v7'))
OLD=Path(os.environ.get('V7_V5B_ROOT',r'D:\Portfolio_Optimization_paper\private_runs\ashare_v5b'))
BASE='7a2039ea57347e05aa4cbb2a442d0d9cfb19e914'
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'inf' if x>0 else '-inf' if x<0 else None
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    text=json.dumps(clean(x),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if p.suffix=='.gz':
        with gzip.open(tmp,'wt',encoding='utf8') as z:z.write(text)
    else:tmp.write_text(text,encoding='utf8')
    tmp.replace(p)
def read(p):
    p=Path(p)
    if p.suffix=='.gz':
        with gzip.open(p,'rt',encoding='utf8') as z:return json.load(z)
    return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    if Path(p).exists():
        with Path(p).open(encoding='utf8') as f:
            for l in f:yield json.loads(l)
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8') as f:f.write(json.dumps(clean(x),ensure_ascii=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seed(*x):return int.from_bytes(hashlib.sha256(('v7-20261017|'+'|'.join(map(str,x))).encode()).digest()[:8],'little')
def used():return sum(r['cpu_seconds'] for r in rows(RUN/'runtime.jsonl'))
def sources():return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
def number(x):return np.inf if x=='inf' else -np.inf if x=='-inf' else x
