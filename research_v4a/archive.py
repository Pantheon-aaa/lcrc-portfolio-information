"""Lossless line-preserving gzip shards, checked by SHA256 and tracked in Git."""
import gzip,hashlib,json,zstandard
from .common import ROOT,OUT,rows,write,run

def archive(budget):
    manifest=[]
    for src in sorted(OUT.glob('*.jsonl')):
        if src.name=='runtime_ledger.jsonl':continue
        budget.check();sha=hashlib.sha256();chunks=[];buf=bytearray();index=0;count=0
        def flush():
            nonlocal buf,index
            if not buf:return
            name=src.name+f'.part{index:04d}.zst';target=OUT/name
            raw=bytes(buf);compressed=zstandard.ZstdCompressor(level=1,threads=0).compress(raw)
            target.write_bytes(compressed)
            if zstandard.ZstdDecompressor().decompress(target.read_bytes())!=raw:raise RuntimeError('lossless archive verification failed')
            chunks.append(dict(path=name,compressed_bytes=len(compressed),raw_bytes=len(raw),sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
            buf=bytearray();index+=1
        with src.open('rb') as f:
            for line in f:
                sha.update(line);buf.extend(line);count+=1
                if len(buf)>=32*2**20:budget.check();flush()
        flush();manifest.append(dict(raw_file=src.name,raw_bytes=src.stat().st_size,lines=count,sha256=sha.hexdigest(),chunks=chunks))
    write(OUT/'archive_manifest.json',manifest)
    print(f'Losslessly archived {len(manifest)} streams, {sum(len(x["chunks"]) for x in manifest)} shards',flush=True)

if __name__=='__main__':run('archive',archive)
