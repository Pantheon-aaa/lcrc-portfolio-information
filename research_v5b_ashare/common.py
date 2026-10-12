from pathlib import Path
import os,json,hashlib,time
import numpy as np

ROOT=Path(__file__).resolve().parent
PRIVATE=Path(os.environ['ASHARE_RUN_ROOT'])
SOURCE=Path(os.environ['ASHARE_SOURCE'])
MEMBERSHIP=Path(os.environ['ASHARE_MEMBERSHIP'])
OUT=ROOT/'output'
BASE='4bb7273fc586ec665bfbee0803eaa3d965bb15a2'
CACHE=Path(os.environ['ASHARE_CACHE_ROOT'])
PRIOR_CPU=271.6875

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'inf' if x>0 else '-inf' if x<0 else None
    return x

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(clean(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def rows(path):
    if Path(path).exists():
        with Path(path).open(encoding='utf-8') as f:
            for line in f:
                if line.strip():yield json.loads(line)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(*args):return hashlib.sha256('|'.join(map(str,args)).encode()).hexdigest()
def array_hash(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def source_hashes():return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
def cpu():return time.process_time()
def params():return read(ROOT/'protocol.json')
def phase(date):
    year=int(date[:4])
    return 'development' if year<=2016 else 'validation' if year<=2019 else 'test' if year<=2025 else 'extension'
