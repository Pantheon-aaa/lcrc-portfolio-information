import sys,time,os,traceback
from .common import *
def main():
    stage=sys.argv[1];start=time.process_time()
    try:
        if stage in ('gate','diagnose','prepare','develop','freeze','confirm','calibrate','cache-check'):
            from .pipeline import dispatch
            if stage=='confirm':
                from .stream_cache import install
                install()
            dispatch(stage,*sys.argv[2:])
        elif stage in ('terminal','stress','stress-profile'):
            from .terminal import dispatch
            dispatch(stage,*sys.argv[2:])
        else:
            from .reporting import dispatch
            dispatch(stage,*sys.argv[2:])
    finally:write(RUN/'metrics'/f'{os.environ.get("V6_STAGE_ID","manual")}.json',dict(cpu_seconds=time.process_time()-start))
if __name__=='__main__':main()
