from pathlib import Path
import json,hashlib,time,gzip,os,traceback,psutil
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
LIMIT=21600.

def seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'inf' if x>0 else ('-inf' if x<0 else None)
    return x
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def append(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def rows(path,fast=False):
    path=Path(path)
    if fast and path.exists():
        import orjson
        with path.open('rb') as f:
            for line in f:
                if line.strip():yield orjson.loads(line)
        return
    if not path.exists() and list(path.parent.glob(path.name+'.part*.zst')):
        import zstandard,io
        for shard in sorted(path.parent.glob(path.name+'.part*.zst')):
            with shard.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as reader,io.TextIOWrapper(reader,encoding='utf-8') as text:
                for line in text:
                    if line.strip():yield json.loads(line)
        return
    if not path.exists() and list(path.parent.glob(path.name+'.part*.gz')):
        for shard in sorted(path.parent.glob(path.name+'.part*.gz')):
            with gzip.open(shard,'rt',encoding='utf-8') as f:
                for line in f:
                    if line.strip():yield json.loads(line)
        return
    if not path.exists() and Path(str(path)+'.gz').exists():path=Path(str(path)+'.gz')
    if path.exists():
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
def digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
def source_hash():return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted(ROOT.glob('*.py')))).hexdigest()
def decision_path(split):
    cfg=json.loads((ROOT/'config_frozen.json').read_text())
    if 'numerical_revision' in cfg:return OUT/f'S2_{split}_validated.jsonl'
    return OUT/f'S2_{split}.jsonl'
class Budget:
    def __init__(self,stage,allocation=LIMIT):
        self.stage=stage;self.start=time.perf_counter();self.prior=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl'));self.allocation=allocation
        self.live=OUT/'live'/f'{os.getpid()}.json';write(self.live,dict(pid=os.getpid(),stage=stage,created=psutil.Process().create_time()))
    def check(self):
        used=sum(r['cpu_seconds'] for r in rows(OUT/'runtime_ledger.jsonl'));active=0.
        for file in (OUT/'live').glob('*.json'):
            try:
                r=json.loads(file.read_text());proc=psutil.Process(r['pid'])
                if abs(proc.create_time()-r['created'])<1:active+=sum(proc.cpu_times()[:2])
            except (OSError,ValueError,psutil.Error):pass
        finalization=self.stage in ('summarize','audit','archive','provenance','posterior_precision','delivery_check','audit_resolution')
        reserve=30 if finalization else 350
        if used+active>LIMIT-reserve or time.process_time()>self.allocation:raise TimeoutError('CPU budget exhausted')
    def finish(self,status):
        append(OUT/'runtime_ledger.jsonl',dict(stage=self.stage,status=status,cpu_seconds=time.process_time(),wall_seconds=time.perf_counter()-self.start,unix=time.time(),pid=os.getpid(),source_hash=source_hash()))
        self.live.unlink(missing_ok=True)
def run(stage,fn,allocation=LIMIT):
    b=Budget(stage,allocation);status='complete'
    try:b.check();return fn(b)
    except Exception as e:
        status='budget_exhausted' if isinstance(e,TimeoutError) else 'failed'
        append(OUT/'failures.jsonl',dict(stage=stage,error=repr(e),traceback=traceback.format_exc()));raise
    finally:b.finish(status)
