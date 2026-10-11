"""Verify complete balanced grids and interval aggregation in the delivered analysis."""
import json,collections,itertools
import numpy as np
from .common import ROOT,OUT,rows,write,run,decision_path


def execute(budget):
    cfg=json.loads((ROOT/'config_frozen.json').read_text());checks={};errors=[]
    s1=list(rows(OUT/'S1_completed.jsonl'));s3=list(rows(OUT/'S3_computation.jsonl'));platforms=list(rows(OUT/'S2_platforms.jsonl'))
    checks['other_grids']=dict(S1_queries=len(s1),S3_queries=len(s3),platform_queries=len(platforms))
    if len(s1)!=280 or len({r['query'] for r in s1})!=280:errors.append(dict(error='S1 grid incomplete or duplicated'))
    if len(s3)!=600 or len({(r['case'],r['method'],str(float(r['beta']))) for r in s3})!=600:errors.append(dict(error='S3 grid incomplete or duplicated'))
    if len(platforms)!=840 or len({r['key'] for r in platforms})!=840:errors.append(dict(error='platform grid incomplete or duplicated'))
    for split,expected in [('development',444),('confirmation',4344)]:
        completed=[r['dataset'] for r in rows(OUT/f'S2_{split}_validated_completed.jsonl')]
        grid=json.loads((OUT/'audit.json').read_text())['counts'][f'S2_{split}']['dataset_grid']
        counts=grid['counts'];methods=grid['method_rows'];bad=grid['malformed']
        checks[split]=dict(completed=len(completed),expected=expected,unique_datasets=len(counts),rows=sum(counts.values()),method_rows=dict(methods),malformed_datasets=bad)
        if len(completed)!=expected or len(set(completed))!=expected or set(counts)!=set(completed) or bad:
            errors.append(dict(stage=split,error='incomplete, duplicated or malformed grid'))
    cells=list(rows(OUT/'stratum_summary.jsonl'))
    confirmation_reps=cfg['confirmation_repetitions'];rep_errors=[{k:r[k] for k in ('family','method','beta','n','repetitions')} for r in cells if r['n']>0 and r['repetitions']!=confirmation_reps]
    checks['stratum_repetitions']=dict(expected=confirmation_reps,cells=len(cells),errors=rep_errors)
    if rep_errors:errors.append(dict(error='unbalanced confirmation repetitions'))
    pairs=json.loads((OUT/'family_paired_effects.json').read_text());checks['family_pairs']=len(pairs)
    if len(pairs)!=48:errors.append(dict(error='expected 12 families x 4 paired comparisons'))
    for r in pairs:
        a=[x for x in cells if x['family']==r['family'] and x['method']=='A1-B' and float(x['beta'])==r['beta'] and x['n']>0 and x['scale']=='common']
        b=[x for x in cells if x['family']==r['family'] and x['method']==r['baseline'] and float(x['beta'])==r['beta'] and x['n']>0 and x['scale']=='common']
        lower=np.mean([x['success_lower'] for x in a])-np.mean([x['success_upper'] for x in b])
        upper=np.mean([x['success_upper'] for x in a])-np.mean([x['success_lower'] for x in b])
        if max(abs(lower-r['lower']),abs(upper-r['upper']))>1e-12:errors.append(dict(error='stream paired grouping disagrees with independent cell aggregation',family=r['family'],beta=r['beta'],baseline=r['baseline']))
    checks['paired_reaggregation']=not any('aggregation' in r['error'] for r in errors)
    write(OUT/'delivery_check.json',dict(passed=not errors,checks=checks,errors=errors))
    print(json.dumps(dict(passed=not errors,errors=len(errors),datasets=checks['confirmation']['completed'])),flush=True)
    if errors:raise RuntimeError('delivery completeness or aggregation check failed')


if __name__=='__main__':run('audit',execute)
