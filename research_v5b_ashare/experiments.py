"""Stage dispatcher; every experiment must be launched through supervisor."""
import os,sys,time,traceback
from .common import *

def main():
    stage=sys.argv[1]
    try:
        if stage=='audit-data':
            from .data import audit,Data,universe
            audit();universe(Data())
        else:
            from . import pipeline
            getattr(pipeline,stage.replace('-','_'))(*sys.argv[2:])
    finally:write(PRIVATE/'stage_metrics'/f'{os.environ.get("ASHARE_STAGE_RUN_ID",os.getpid())}.json',dict(cpu_seconds=time.process_time(),stage=stage,pid=os.getpid()))

if __name__=='__main__':main()
