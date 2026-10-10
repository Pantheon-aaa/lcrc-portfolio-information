"""Clarify uninstrumented baseline timers without changing numerical results."""
from .common import PRIVATE,ROOT,read,dump

def run():
    changed=0
    for p in (PRIVATE/'records').rglob('*.json'):
        r=read(p);dirty=False
        for a in r.get('methods',[]):
            if a['status']=='baseline' and not a.get('timing_note'):
                a['uninstrumented_timer_placeholder']=a.get('seconds')
                a['seconds']=None
                a['timing_note']='Standalone baseline time was not instrumented; included in whole-window process CPU/wall time.'
                dirty=True;changed+=1
        if dirty:dump(p,r)
    dump(ROOT/'output/timer_annotation.json',dict(annotated_baseline_rows=changed,weights_changed=False,evaluation_changed=False,
        meaning='An original zero timer placeholder is not a measured zero runtime. Exported method times are null for these baselines.'))
    print('baseline timer annotations',changed)

if __name__=='__main__':run()
