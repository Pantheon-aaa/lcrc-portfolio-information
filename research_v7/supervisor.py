import sys,os,time,subprocess,psutil
from .common import *
def main():
    stage=sys.argv[1];prior=used();own=time.process_time();wall=time.perf_counter()
    cap=43200 if stage in ('report','audit','archive') else 34560
    if prior>=cap-120:raise SystemExit('CPU reserve reached')
    runid=f'{stage}-{time.time_ns()}';env=dict(os.environ,V7_STAGE_ID=runid)
    child=subprocess.Popen([sys.executable,'-B','-m','research_v7.experiments',*sys.argv[1:]],env=env)
    proc=psutil.Process(child.pid);tree={};peak=0;reason=None
    while child.poll() is None:
        try:
            rss=0
            for p in [proc,*proc.children(recursive=True)]:
                try:tree[p.pid]=max(tree.get(p.pid,0),sum(p.cpu_times()[:2]));rss+=p.memory_info().rss
                except psutil.Error:pass
            peak=max(peak,rss)
            if prior+sum(tree.values())+time.process_time()-own>=cap-120:reason='cpu_boundary'
            if rss>psutil.virtual_memory().total*.75:reason='memory_boundary'
            if reason:
                for p in proc.children(recursive=True):p.kill()
                child.kill();child.wait();break
        except psutil.Error:pass
        time.sleep(.5)
    metric=RUN/'metrics'/f'{runid}.json';cpu=max(sum(tree.values()),read(metric)['cpu_seconds'] if metric.exists() else 0)
    row=dict(stage=stage,args=sys.argv[2:],cpu_seconds=cpu+time.process_time()-own+1,wall_seconds=time.perf_counter()-wall,unix_end=time.time(),exit_code=child.returncode,reason=reason,peak_rss=peak,sources=sources())
    append(RUN/'runtime.jsonl',row);write(OUT/'runtime.json',dict(records=list(rows(RUN/'runtime.jsonl')),cpu_seconds=used(),cap=43200,execution_cap=34560))
    print('STAGE',stage,'CPU',round(row['cpu_seconds'],2),'TOTAL',round(used(),2),flush=True)
    raise SystemExit(child.returncode or 0)
if __name__=='__main__':main()
