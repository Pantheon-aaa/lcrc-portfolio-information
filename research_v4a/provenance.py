import os,platform,hashlib,importlib.metadata,subprocess
import numpy as np
from .common import ROOT,OUT,write,run

def execute(budget):
    names=['numpy','scipy','cvxpy','clarabel','osqp','pandas','pyarrow','matplotlib','psutil','highspy','mpmath','pyscipopt','orjson','zstandard'];versions={n:importlib.metadata.version(n) for n in names}
    info=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),packages=versions,thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},delivery_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))})
    try:
        import scipy.linalg
        from threadpoolctl import threadpool_info
        info['loaded_threadpools']=threadpool_info()
    except ImportError:info['loaded_threadpools']='threadpoolctl unavailable'
    try:
        from scipy.optimize._highspy._core import _Highs
        info['scipy_HiGHS_version']=_Highs().version()
    except Exception as ex:info['scipy_HiGHS_version']=repr(ex)
    try:
        from pyscipopt import Model
        m=Model();info['SCIP_version']=[m.getMajorVersion(),m.getMinorVersion(),m.getTechVersion()]
    except Exception as ex:info['SCIP_version']=repr(ex)
    write(OUT/'environment.json',info)
    (ROOT/'requirements.txt').write_text('\n'.join(n+'=='+versions[n] for n in names)+'\n',encoding='utf-8')
    print('Captured exact environment and source versions',flush=True)

if __name__=='__main__':run('provenance',execute)
