from pathlib import Path
import hashlib
import json
import os
import time

ROOT = Path(__file__).parent
PRIVATE = Path(os.environ.get('ASHARE_PRIVATE', r'D:\Portfolio_Optimization_paper\private_runs\ashare_v1'))
SOURCE = Path(os.environ.get('ASHARE_SOURCE', r'D:\A股日频数据\量化研究\research\a_share_research.parquet'))

def seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def period(date):
    y = int(str(date)[:4])
    return 'development' if y <= 2016 else 'validation' if y <= 2019 else 'test' if y <= 2025 else 'extension'

def ledger(name, wall, cpu, extra=None):
    p = PRIVATE / 'ledger'; p.mkdir(parents=True, exist_ok=True)
    dump(p / (name + '.json'), dict(task=name, finished_unix=time.time(), wall_seconds=wall,
         cpu_seconds=cpu, **(extra or {})))
