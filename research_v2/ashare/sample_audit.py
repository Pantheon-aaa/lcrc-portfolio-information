import numpy as np
import pandas as pd
from .common import PRIVATE,ROOT,read
from .run import tasks

def run():
    output=[]
    for c in ('long','limited'):
        rows=[t['row'] for t in tasks('test') if t['row']['cohort']==c]
        for phase in ('test','extension'):
            subset=[r for r in rows if r['period']==phase]
            ids=set(i for r in subset for i in r['ids']);counts=[];left=[];internal=[]
            limited_count=left_dom=internal_dom=0
            for r in subset:
                ds=r['diagnostics']
                counts.extend(ds['valid_counts']);left.extend(ds['left_missing']);internal.extend(ds['internal_missing'])
                for n,l,j in zip(ds['valid_counts'],ds['left_missing'],ds['internal_missing']):
                    if n<240:
                        limited_count+=1;left_dom+=l>=j;internal_dom+=j>l
            output.append(dict(cohort=c,period=phase,windows=len(subset),panels=len(set(r['panel'] for r in subset)),
                unique_security_codes=len(ids),first_signal=min(r['date'] for r in subset),last_signal=max(r['date'] for r in subset),
                last_evaluation=max(r['evaluation_end'] for r in subset),valid_count_median=float(np.median(counts)),
                left_missing_median=float(np.median(left)),internal_missing_median=float(np.median(internal)),
                limited_slots=limited_count,left_dominated_slots=int(left_dom),internal_dominated_slots=int(internal_dom)))
    df=pd.DataFrame(output);df.to_csv(ROOT/'output/sample.csv',index=False);print(df.to_string(index=False))

if __name__=='__main__':run()
