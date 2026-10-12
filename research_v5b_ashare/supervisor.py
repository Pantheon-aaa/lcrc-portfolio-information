"""External cumulative process CPU/memory watchdog with a protected finalization reserve."""
import sys,subprocess,time,os
import psutil
from .common import PRIVATE,OUT,rows,append,read,write,source_hashes,cpu,PRIOR_CPU

def main():
    stage=sys.argv[1];start=time.perf_counter();own_start=cpu()
    final=stage in ('evaluate','report','archive','provenance','delivery-check')
    cap=86400 if final else 69120
    ledger=PRIVATE/'runtime_ledger.jsonl';prior=PRIOR_CPU+sum(r['cpu_seconds'] for r in rows(ledger))
    if prior>=cap-120:raise SystemExit('Reserved CPU boundary reached; no new stage launched')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    initial_source_hashes=source_hashes()
    run_id=f'{stage}_{time.time_ns()}';env=dict(os.environ,ASHARE_STAGE_RUN_ID=run_id)
    child=subprocess.Popen([sys.executable,'-B','-m','research_v5b_ashare.experiments',stage,*sys.argv[2:]],env=env)
    process=psutil.Process(child.pid);used=0.;peak=0;reason=None;tree_cpu={}
    while child.poll() is None:
        try:
            rss=0
            for proc in [process,*process.children(recursive=True)]:
                try:
                    tree_cpu[proc.pid]=max(tree_cpu.get(proc.pid,0.),sum(proc.cpu_times()[:2]));rss+=proc.memory_info().rss
                except psutil.Error:pass
            used=sum(tree_cpu.values());peak=max(peak,rss)
            if prior+used+(cpu()-own_start)>=cap-120:reason='cpu_reserve_boundary'
            elif rss>psutil.virtual_memory().total*.75:reason='process_memory_limit'
            if reason:
                for descendant in process.children(recursive=True):descendant.kill()
                child.kill();child.wait();break
        except psutil.Error:pass
        time.sleep(.5)
    metric=PRIVATE/'stage_metrics'/f'{run_id}.json'
    if metric.exists():used=max(used,read(metric)['cpu_seconds'])
    # Conservative 1 second covers final child output/exit after its own meter.
    record=dict(stage=stage,arguments=sys.argv[2:],pid=child.pid,run_id=run_id,process_tree_cpu=tree_cpu,cpu_seconds=used+cpu()-own_start+1.,
      child_cpu_seconds=used,exit_allowance_seconds=1.,wall_seconds=time.perf_counter()-start,
      peak_rss_bytes=peak,exit_code=child.returncode,stop_reason=reason,status='complete' if child.returncode==0 else 'failed',
      unix_end=time.time(),source_hashes=source_hashes(),initial_source_hashes=initial_source_hashes)
    append(ledger,record);records=list(rows(ledger));write(OUT/'runtime_snapshot.json',dict(records=records,prior_v5_cpu=PRIOR_CPU,cumulative_cpu=PRIOR_CPU+sum(r['cpu_seconds'] for r in records),cpu_limit_seconds=86400,optimization_limit_seconds=69120))
    print('STAGE',stage,'CPU',round(record['cpu_seconds'],2),'CUMULATIVE',round(prior+record['cpu_seconds'],2),flush=True)
    raise SystemExit(child.returncode or (1 if reason else 0))

if __name__=='__main__':main()
