"""Prefetch frozen N=100 historical inputs with the unchanged frontend.

No decisions/evaluations occur here. Unique temporary names permit harmless
duplicate estimation if the sequential decision worker reaches the same job.
"""
import os
import numpy as np
from .common import *
from .data import Data
from .risk import frontend

def run():
    data=Data();jobs=[j for j in read(PRIVATE/'manifest.json') if j['stage']=='R4-N' and j['N']==100 and j['panel']['status']=='ready']
    # Reverse prefetch order leaves the sequential decision queue unchanged and
    # keeps the two workers well separated for most of the run.
    jobs=list(reversed(jobs));log=read(PRIVATE/'prefetch_log.json') if (PRIVATE/'prefetch_log.json').exists() else []
    for i,j in enumerate(jobs):
        if (PRIVATE/'decisions'/f'{j["id"]}.json').exists():continue
        p=j['panel'];cols=p['ranking'][:100]
        if len(cols)<100:continue
        key=digest('F1-EM-v1',p['t'],','.join(map(str,cols)),(126,252,504),(.1,.3,.6,.9),True)
        path=PRIVATE/'frontends'/f'{key}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():continue
        f=frontend(data,p,N=100,cache=False)
        assert f['shared_cost_id']==key
        # The frontend executes exactly the frozen estimator/scoring code.
        duplicate=path.exists() and meta.exists()
        if duplicate:
            with np.load(path) as existing:
                assert np.array_equal(existing['Q'],f['Q']) and np.array_equal(existing['scores'],f['scores'])
        else:
            tmp=path.with_name(f'{key}.prefetch{os.getpid()}.tmp.npz')
            np.savez_compressed(tmp,Q=f['Q'],scores=f['scores'],u=f['u']);tmp.replace(path)
            tempmeta=meta.with_name(f'{key}.prefetch{os.getpid()}.tmp')
            tempmeta.write_text(json.dumps(clean(f['metadata']),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tempmeta.replace(meta)
        log.append(dict(id=j['id'],shared_cost_id=key,cpu_seconds=f['metadata']['cpu_seconds'],Q_hash=array_hash(f['Q']),scores_hash=array_hash(f['scores']),duplicate_verified=duplicate))
        write(PRIVATE/'prefetch_log.json',log)
        print('PREFETCH',i+1,len(jobs),j['id'],round(f['metadata']['cpu_seconds'],2),flush=True)
