"""Lossless, bounded-size public record shards and verified local restoration.

Only canonical top-level JSONL results are packed. Historical/redundant local
archives remain local and are inventoried separately; no empirical rows are dropped.
"""
import argparse, gzip, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PUB=ROOT/'publication'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def pack():
    folder=PUB/'records';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in sorted((ROOT/'results').glob('*.jsonl')):
        pieces=[];buffer=bytearray();count=0;part=0;total=0
        def flush():
            nonlocal part,count,total,buffer
            if not buffer:return
            target=folder/f'{source.stem}.{part:03d}.jsonl.gz'
            target.write_bytes(gzip.compress(bytes(buffer),compresslevel=6,mtime=0))
            pieces.append({'path':target.relative_to(ROOT).as_posix(),'rows':count,
                'uncompressed_bytes':len(buffer),'sha256':sha(target),
                'uncompressed_sha256':hashlib.sha256(buffer).hexdigest()})
            total+=count;part+=1;count=0;buffer=bytearray()
        with source.open('rb') as f:
            for line in f:
                if buffer and len(buffer)+len(line)>16*1024*1024:flush()
                buffer.extend(line);count+=1
        flush()
        records.append({'source':source.relative_to(ROOT).as_posix(),'rows':total,
            'bytes':source.stat().st_size,'sha256':sha(source),'parts':pieces})
        print(source.name,total,'rows',len(pieces),'parts',flush=True)
    index={'schema':1,'encoding':'lossless original UTF-8 JSONL; Python nonfinite numbers retained',
        'scope':'all canonical top-level results/*.jsonl; archives and duplicate worker shards excluded',
        'records':records}
    (PUB/'records_index.json').write_text(json.dumps(index,indent=2),encoding='utf8')
    historical=[]
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir():continue
        for f in sorted(folder.rglob('*')):
            if f.is_file():historical.append({'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,
                'sha256':sha(f),'published':False,'reason':'superseded run or duplicate worker partition; retained locally'})
    (PUB/'local_archive_inventory.json').write_text(json.dumps(historical,indent=2),encoding='utf8')
    print('Compressed bytes',sum((ROOT/p['path']).stat().st_size for r in records for p in r['parts']),flush=True)

def restore(verify_only=False):
    index=json.loads((PUB/'records_index.json').read_text())
    for entry in index['records']:
        relative=Path(entry['source'])
        target=(ROOT/relative).resolve()
        if not target.is_relative_to((ROOT/'results').resolve()) or target.suffix!='.jsonl':
            raise ValueError('invalid result destination')
        existing=target.exists()
        if existing and sha(target)!=entry['sha256']:
            raise ValueError(f'{relative}: existing result differs; preserve it before restoration')
        total=hashlib.sha256();rows=0
        temp=target.with_suffix('.restore.tmp')
        out=None if verify_only or existing else temp.open('xb')
        try:
            for piece in entry['parts']:
                f=(ROOT/piece['path']).resolve()
                if not f.is_relative_to((PUB/'records').resolve()):raise ValueError('invalid shard path')
                if sha(f)!=piece['sha256']:raise ValueError(f'corrupt compressed shard: {f.name}')
                data=gzip.decompress(f.read_bytes())
                if hashlib.sha256(data).hexdigest()!=piece['uncompressed_sha256']:raise ValueError('corrupt payload')
                total.update(data);rows+=data.count(b'\n')
                if out is not None:out.write(data)
            if rows!=entry['rows'] or total.hexdigest()!=entry['sha256']:raise ValueError('record checksum mismatch')
        finally:
            if out is not None:out.close()
        if out is not None:temp.replace(target)
    print('Verified',len(index['records']),'canonical datasets and all lossless shards.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pack','restore','verify'])
    command=parser.parse_args().command
    if command=='pack':pack()
    else:restore(command=='verify')
