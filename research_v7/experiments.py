import sys,time,os
from .common import *
def main():
    start=time.process_time();stage=sys.argv[1]
    try:
        if stage=='gate':
            from .checks import run
        elif stage=='e1':
            from .real_diagnostic import run
        elif stage in ('prepare','develop','freeze','confirm','e3'):
            from .pipeline import dispatch
            dispatch(stage,*sys.argv[2:]);return
        else:
            from .reporting import dispatch
            dispatch(stage,*sys.argv[2:]);return
        run()
    finally:
        runid=os.environ.get('V7_STAGE_ID','manual');child=RUN/'metrics'/f'{runid}.children.json'
        extra=read(child)['cpu_seconds'] if child.exists() else 0
        write(RUN/'metrics'/f'{runid}.json',dict(cpu_seconds=time.process_time()-start+extra,child_cpu_seconds=extra,stage=stage))
if __name__=='__main__':main()
