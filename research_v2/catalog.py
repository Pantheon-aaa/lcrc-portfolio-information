"""Map every approved algorithm identifier to implementation and executed coverage."""
import json,collections,platform,sys,importlib.metadata
from pathlib import Path
from .experiment_v2 import ROOT,OUT,CFG,load_rows,write_json
from .registry import all_specs

def run():
    selection=json.loads((CFG/'selection.json').read_text());coverage={};states={}
    for stage in ('development','G1','G2','confirm','G3_d20','G3_d50','blend_development','repair_screen','repair_confirmation'):
        rows=load_rows(stage);coverage[stage]=collections.Counter(r['method'] for r in rows)
        states[stage]=collections.Counter((r['method'],r.get('status','exception')) for r in rows)
    text=['# Algorithm catalog and execution trace','',
        'All methods share the supplied scenario and observation information. Reference masses, posterior masses and adverse decision weights have separate representations.',
        'Objectives are implemented in `risk_envelopes.py`; convex optimization and independent bounds in `risk_center.py`; the direct Clarabel epigraph is in `native_conic.py`. Full specifications are frozen in `configs/selection.json` and protocol revisions.','',
        '| Algorithm | Functional / parameter | Conditional | Development rows | G1 rows | G2 rows | Confirm rows |','|---|---|---|---:|---:|---:|---:|']
    for s in all_specs():
        parameter=(f'tau/u={s.temperature}' if s.kind=='soft' else f'r={s.radius}' if s.kind=='lr' else
                   f'budget fraction={s.budget_fraction}' if s.kind=='budget' else f'lambda/u={s.penalty}' if s.penalty else f'rho={s.rho}')
        text.append('| '+ ' | '.join([s.name,s.kind+'; '+parameter,str(s.conditional)]+[str(coverage[st][s.name]) for st in ('development','G1','G2','confirm')])+' |')
    text += ['', '## Supplementary exploratory coverage','',
        'These rows do not enlarge the original confirmatory comparison family. The original confirmation had already been inspected when the selector was repaired.', '',
        '| Method | Blend development | Repair screening | Reused test panel |','|---|---:|---:|---:|']
    for name in sorted(set().union(*(set(coverage[st]) for st in ('blend_development','repair_screen','repair_confirmation')))):
        text.append('| '+name+' | '+' | '.join(str(coverage[st][name]) for st in ('blend_development','repair_screen','repair_confirmation'))+' |')
    text += ['', '## Aliases, triggered methods and additional comparators','',
        '- REF-MA is A-Mean, REF-MAX is A-Max. B-Mean is exactly A-Mean on a common joint posterior; both run in development as a correctness check.',
        '- A-ATK and B-ATK coincide with reference tails only when the relevant reference atoms have equal mass and rho agrees with k/M. Under importance quadrature they are separate implemented methods.',
        '- A-Blend and B-Blend are implemented. An erroneous worst-group selection criterion initially made the blend trigger unreachable. The corrected criterion activated both, and blend_development / repair_screen contain the supplementary runs. Original confirmation is kept separate.',
        '- B-Plugin uses the outer node with largest discrete posterior mass and solves its conditional tail at rho=0.5. It is a grid-based plug-in comparator, not a continuous MAP optimizer.',
        '- REF-EB learns a four-component prior mixture from independent development marginal likelihoods; it does not use true parameter labels.',
        '- Equal Weight, a fixed diagonal EM shrinkage comparator, affine MLE, old LCRC (a fixed 24-instance subset), and four privileged finite-grid oracle priors are in baseline_audit. These do not replace same-information MA/EB comparisons.',
        '- H-RankThenP and its conditional version use four fixed starts and derivative-free search. They have no global gap. Their maximum normalized start-value spreads in development were approximately 0.245u and 0.472u, respectively; this is substantial numerical instability, not merely absence of a convex gap. A future heuristic candidate must have all four finite starts within 1e-4u on the whole development panel. Neither qualifies.',
        '- H-pR is a convex count Top-k objective applied to probability-times-regret. It has representation dependence, not an automatic nonconvexity problem.',
        '- H-PostSampleK samples each finite posterior once and applies equal-count tails without multiplying the likelihood again. Additional fixed-action sampling convergence diagnostics are saved in posterior_sampling_tail.json.',
        '- Conditional H variants preserve outer posterior group masses. Filtering first retains at least 95% posterior mass; subsequent integer k is ceil(rho*K), where K is retained count.',
        '- B-Max main experiments maximize over supplied finite conditional scenarios. The high-resolution audit additionally uses exact hidden-box vertices for G2, keeping an approximate posterior outer marginal. Only that audit has exact hidden maximization.','',
        '## Evidence status','',
        'A nonzero row count means attempted execution, not a favorable result. Numerical exceptions, open gaps and heuristic statuses remain in the records. Performance eligibility and approximation sensitivity are discussed in REPORT_ZH.md. The separate boundary and embedded-information experiments are mechanism studies, not extra observations in the algorithm comparison.','',
        '## Explicit scope limits','',
        'No real A-share backtest, public push, DFL training, nonlinear-shrinkage campaign or LF-PQP comparison was performed in this round. Five-parameter cases are a fixed precision diagnostic, not a broad confirmation claim. General simultaneous outer/inner decay and continuous-nuisance optimality are open theoretical results.']
    (ROOT/'docs/ALGORITHM_CATALOG.md').write_text('\n'.join(text),encoding='utf8')
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','osqp','pandas','matplotlib','psutil','pyarrow')}
    import psutil
    write_json(ROOT/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':versions,
        'logical_cpus':psutil.cpu_count(),'physical_cores':psutil.cpu_count(logical=False),'memory_bytes':psutil.virtual_memory().total,
        'cpu_model':'Intel Core Ultra 5 125H','blas_threads':1,'main_batch_processes':8,
        'parent_git_commit':'ee75423c1a2b3bda55ec260b8aad8b43aa6c5982','local_branch':'research/conditional-tail-v2'})
    write_json(OUT/'algorithm_coverage.json',{st:dict(c) for st,c in coverage.items()})

if __name__=='__main__':run()
