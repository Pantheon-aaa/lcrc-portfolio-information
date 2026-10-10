from pathlib import Path
import json, hashlib, time, os, sys, gzip
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
CFG = ROOT / 'configs'
TOTAL_CPU = 6*3600

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8], 'little')

def clean(obj):
    if isinstance(obj, dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, np.generic): return clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj

def write(path, obj, frozen=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    obj=clean(obj)
    if frozen and path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=obj: raise RuntimeError(f'frozen config differs: {path}')
        return
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def append(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(clean(obj),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')

def rows(path):
    path=Path(path)
    if not path.exists() and Path(str(path)+'.gz').exists():
        with gzip.open(str(path)+'.gz','rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
        return
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)

def source_hash():
    h=hashlib.sha256()
    for p in sorted(ROOT.glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

class Budget:
    """Single experiment process; persistent CPU ledger, including failed stages."""
    def __init__(self,stage,limit):
        self.stage=stage;self.limit=limit;self.cpu=time.process_time();self.wall=time.perf_counter()
        self.used=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl'))
        self.same=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl') if r['stage']==stage)
    def check(self):
        elapsed=time.process_time()-self.cpu
        if self.used+elapsed>=TOTAL_CPU or self.same+elapsed>=self.limit: raise TimeoutError('CPU budget exhausted')
    def finish(self,status='complete'):
        append(OUT/'runtime_ledger.jsonl',dict(stage=self.stage,status=status,cpu_seconds=time.process_time()-self.cpu,
          wall_seconds=time.perf_counter()-self.wall,unix=time.time(),pid=os.getpid(),source_hash=source_hash()))

def timed_stage(stage,limit,fn):
    b=Budget(stage,limit);status='complete'
    try: b.check();return fn(b)
    except Exception as exc:
        status='budget_exhausted' if isinstance(exc,TimeoutError) else 'failed'
        append(OUT/'failures.jsonl',dict(stage=stage,error=repr(exc),source_hash=source_hash()));raise
    finally:b.finish(status)
