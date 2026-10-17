"""Two long-lived numerical workers; all jobs retain the frozen ordering."""
import os,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from .common import *

def execute(j):
    from .pipeline import run_job
    start=time.process_time()
    try:r=run_job(j,'confirmation')
    except Exception:r=dict(job=j,status='failed',error=traceback.format_exc(),cpu=time.process_time()-start)
    return r,os.getpid(),time.process_time()

def batch(reps,limit):
    from .pipeline import manifest,jobkey
    pending_jobs=iter(j for j in manifest('confirmation',reps) if not (RUN/'confirmation'/f'{jobkey(j)}.json.gz').exists())
    own=time.process_time();worker_cpu={};count=0;cost=[];exhausted=False
    metric=RUN/'metrics'/f"{os.environ.get('V7_STAGE_ID','manual')}.children.json"
    def spent():return sum(worker_cpu.values())+time.process_time()-own
    with ProcessPoolExecutor(max_workers=2) as pool:
        live={}
        while True:
            while len(live)<2 and not exhausted and spent()<max(0,limit-70):
                try:j=next(pending_jobs)
                except StopIteration:exhausted=True;break
                live[pool.submit(execute,j)]=j
            if not live:break
            done,_=wait(live,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                j=live.pop(future)
                r,pid,worker_total=future.result();worker_cpu[pid]=worker_total
                write(RUN/'confirmation'/f'{jobkey(j)}.json.gz',r);count+=1;cost.append(r['cpu'])
                write(metric,dict(cpu_seconds=sum(worker_cpu.values())+len(worker_cpu),worker_totals=worker_cpu))
                if count%128==0:print('confirmation',count,'CPU',round(spent(),1),r['status'],flush=True)
    write(OUT/'confirmation_run.json',dict(new=count,cpu=spent(),costs=cost,worker_totals=worker_cpu,workers=2,manifest_exhausted=exhausted))
    write(metric,dict(cpu_seconds=sum(worker_cpu.values())+len(worker_cpu),worker_totals=worker_cpu))

def smoke():
    from .pipeline import manifest,jobkey
    jobs=list(manifest('development',1))[:2];workers={};checks=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for j,(r,pid,cpu) in zip(jobs,pool.map(execute,jobs)):
            workers[pid]=max(workers.get(pid,0),cpu);old=read(RUN/'development'/f'{jobkey(j)}.json.gz')
            ok=r['status']==old['status'] and len(r['rows'])==len(old['rows']) and len(r['terminal'])==len(old['terminal'])
            for a,b in zip(r['rows'],old['rows']):
                ok=ok and a['certified']==b['certified'] and a['false_certification']==b['false_certification'] and a.get('rho')==b.get('rho')
            for a,b in zip(r['terminal'],old['terminal']):
                ok=ok and np.array_equal(a.get('weights'),b.get('weights'))
            checks.append(dict(job=j,passed=bool(ok)))
    write(RUN/'metrics'/f"{os.environ.get('V7_STAGE_ID','manual')}.children.json",dict(cpu_seconds=sum(workers.values())+len(workers)))
    write(OUT/'parallel_smoke.json',dict(passed=all(x['passed'] for x in checks),checks=checks))
    if not all(x['passed'] for x in checks):raise AssertionError('parallel execution differs from serial development')
