"""Parallel frozen-manifest partitions; no change in datasets or tuning."""
import json,subprocess,sys,time
from pathlib import Path
from .experiment_v2 import ROOT,OUT,CFG,load_rows,write_json

def merge(stage):
    paths=sorted(OUT.glob(stage+'-shard-*.jsonl'))
    rows=load_rows(stage); keys={(r['instance_id'],r['method']) for r in rows}
    for p in paths:
        for line in p.read_text(encoding='utf8').splitlines():
            r=json.loads(line); key=r['instance_id'],r['method']
            if key in keys: raise RuntimeError('Duplicate completion '+str(key))
            keys.add(key);rows.append(r)
    temp=OUT/(stage+'.merged.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    temp.replace(OUT/(stage+'.jsonl'))
    # Shards remain archived for provenance, never pooled twice.
    folder=OUT/'parallel_shards';folder.mkdir(exist_ok=True)
    for p in paths:p.replace(folder/p.name)
    print('merged',stage,len(rows),flush=True)

def run():
    jobs=[('G2',i,3) for i in range(3)]+[('confirm',i,5) for i in range(5)]
    write_json(CFG/'parallel_execution.json',{'jobs':jobs,'blas_threads_per_process':1,
        'reason':'timing-based repartition; frozen datasets and candidates unchanged',
        'solver_revision':'stop dual iterations only after already required numerical gap',
        'maximum_concurrent_experiment_processes':8})
    active=[]
    for stage,i,n in jobs:
        log=(OUT/f'{stage}-shard-{i}of{n}.log').open('a',encoding='utf8')
        log.write('\nRESUME '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');log.flush()
        p=subprocess.Popen([sys.executable,'-m','research_v2.experiment_v2',stage,'--shard',f'{i}/{n}'],
            cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        active.append((stage,i,n,p,time.monotonic(),log))
    last=0
    while any(p.poll() is None for _,_,_,p,_,_ in active):
        elapsed=sum(time.monotonic()-tic for _,_,_,p,tic,_ in active if p.poll() is None)
        ledger=sum(x['wall_seconds'] for x in load_rows('runtime_ledger'))
        if elapsed+ledger>=86400 and not (CFG/'budget_stop.json').exists():
            write_json(CFG/'budget_stop.json',{'reason':'cumulative process wall-time budget reached','unix':time.time()})
        if time.monotonic()-last>60:
            print('workers',[(s,i,p.poll()) for s,i,_,p,_,_ in active],
                  'cumulative hours',round((elapsed+ledger)/3600,3),flush=True);last=time.monotonic()
        time.sleep(10)
    for s,i,n,p,tic,log in active:
        log.close()
        if p.returncode:raise RuntimeError(f'{s} shard {i}/{n} failed; inspect log')
    for stage in ('G2','confirm'):merge(stage)

if __name__=='__main__':run()
