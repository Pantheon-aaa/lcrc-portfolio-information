"""Conditional E3 entry point. The gate reads synthetic aggregates only."""
from collections import defaultdict
from .common import *

def planned():
    jobs=[]
    for domain in ('hs300','zz500'):
      for n in (20,50):
       for library in ('F1','block_bootstrap','PCA'):
        for year in (2014,2015,2016):
         months=range(1,13) if year<2016 else (3,6,9)
         for month in months:
          jobs.append(dict(domain=domain,N=n,panel=11,library=library,year=year,month=month,phase='calibration' if year<2016 else 'development_evaluation',signal='last historical trading date in month',label='63 days from t+2; end within year 2016'))
    return jobs

def gate():
    from .pipeline import jobkey
    manifest=read(ROOT/'manifest.json.gz');missing=[];failed=[];rates=defaultdict(list);method_errors=0;contradictions=0
    for j in manifest:
        p=RUN/'confirmation'/f'{jobkey(j)}.json.gz'
        if not p.exists():missing.append(j);continue
        r=read(p)
        if r['status']!='complete':failed.append(j);continue
        method_errors+=sum('error' in a for a in r['terminal'])
        contradictions+=sum(a['method'] in ('data_decision','data_operator') and a['certified'] and a['confidence_event'] and a['false_certification'] for a in r['rows'])
        if r['panel']=='correct' or r['perturbation']=='blind':
            # Each truth case and each of the two public designs has equal weight.
            a=[x for x in r['rows'] if x['method']=='data_decision' and x['n']==2048 and x['ratio']==1.25]
            rates[j['family']].extend(float(x['certified']) for x in a)
    means={k:float(np.mean(v)) for k,v in rates.items()}
    passed_n=sum(v>=.10 for v in means.values())
    e1=read(OUT/'e1_records.json');traceable=sum(x['status']=='complete' for x in e1)==396 and (RUN/'e1_source_hashes.json').exists()
    if traceable:traceable=all(sha(p)==h for p,h in read(RUN/'e1_source_hashes.json').items())
    ok=not missing and not failed and not method_errors and not contradictions and read(OUT/'gate0.json')['passed'] and traceable and passed_n>=2
    reasons=[]
    if missing:reasons.append('confirmation_incomplete')
    if failed:reasons.append('confirmation_failed_instances')
    if method_errors:reasons.append('terminal_method_errors')
    if contradictions:reasons.append('certificate_contradiction_on_confidence_event')
    if passed_n<2:reasons.append('fewer_than_two_families_with_10_percent_data_certification')
    if not traceable:reasons.append('E1_traceability_not_confirmed')
    result=dict(passed=ok,family_rates=means,families_above_threshold=passed_n,threshold=.10,missing=len(missing),failed=len(failed),method_errors=method_errors,numerical_contradictions=contradictions,E1_traceable=traceable,reasons=reasons,scope='known/convex plus blind; n2048; epsilon/Gamma1.25; both views equally weighted',not_a_new_performance_test=True)
    write(OUT/'e3_gate.json',result)
    write(ROOT/'e3_manifest.json',dict(planned=planned(),status='eligible' if ok else 'not_run_gate_failed',reason=reasons))
    return result

def run():
    result=gate()
    print('E3_GATE',result,flush=True)
    if result['passed']:
        # Do not silently broaden scope: execution is supplied only if the
        # prespecified gate is reached; this branch is explicitly incomplete.
        write(OUT/'e3_status.json',dict(status='eligible_implementation_pending',completed=False))
    else:
        write(OUT/'e3_status.json',dict(status='not_run_gate_failed',completed=False,reason=result['reasons'],real_extension_data_read=False))
