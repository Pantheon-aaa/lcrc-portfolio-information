# Conditional-tail portfolio research v2

This research branch implements the approved 2026-10-08 plan. The experiment was completed locally and is now being published at the user's request, with v1 research code/results preserved. No A-share backtest is run.

**For GPT or human review, start with [the detailed report](REVIEW_REPORT_ZH.md) and [core-code guide](docs/CODE_REVIEW_GUIDE.md).** Elapsed time between protocol freezing and the last experiment was 2.556 hours; 13.442 hours denotes summed concurrent experiment-task time. Recorded process CPU time was 13.268 hours. These exclude unrecorded checks and document preparation.

The repository stores canonical JSONL in [lossless gzip shards](publication/README.md). After cloning, run `python -m research_v2.publication verify`, then `python -m research_v2.publication restore` before existing-result validation or summaries. Small summaries, figures and gate JSON remain directly readable. Historical local archives are inventoried separately and are not all uploaded.

Start with `docs/ASSESSMENT_ZH.md` for the conclusions, `REPORT_ZH.md` for detailed evidence, `docs/THEORY_NOTES.md` for proofs, and `docs/LITERATURE_AND_CLAIMS.md` for the novelty boundary. Mathematical propositions in the notes are conditional research derivations, not externally certified novelty claims. No method is declared a confirmed average-risk winner over Model Averaging.

A plain-language theory explanation is in `docs/THEORY_EXPLAINED_ZH.md`. The editable standalone TeX source is `docs/THEORY_NOTES.tex`, and the compiled reading copy is `output/pdf/THEORY_NOTES.pdf`. `docs/ALGORITHM_CATALOG.md` traces every approved algorithm. Numerical revisions and archival scope are in `docs/IMPLEMENTATION_HISTORY.md`.

`docs/PLAN_COVERAGE_ZH.md` maps the approved research plan to executed evidence, diagnostic-only checks and unresolved claims. Heuristic records use Python JSON NaN for unknown global bounds; strict JSON consumers should convert nonfinite values to null, never to zero.

## Reproduction

Run from the parent repository root, using its Python environment and `requirements.txt`:

```bash
python -m research_v2.gates
python -m research_v2.native_gate
python -m research_v2.experiment_v2 init
python -m research_v2.experiment_v2 setup
python -m research_v2.experiment_v2 pilot
python -m research_v2.experiment_v2 development
python -m research_v2.experiment_v2 select
python -m research_v2.experiment_v2 G1
python -m research_v2.experiment_v2 G2
python -m research_v2.boundary_experiments
python -m research_v2.g3_experiments --d 20
python -m research_v2.g3_experiments --d 50
python -m research_v2.extra_audits baselines
python -m research_v2.extra_audits resolution
python -m research_v2.extra_audits mechanisms
python -m research_v2.experiment_v2 confirm
python -m research_v2.posterior_audit
python -m research_v2.ablation_audit
python -m research_v2.high_resolution_audit
python -m research_v2.five_parameter_audit
python -m research_v2.embedded_boundary
python -m research_v2.theory_checks
python -m research_v2.timing_audit
python -m research_v2.selection_repair
python -m research_v2.verify_delivery
python -m research_v2.catalog
python -m research_v2.final_assessment
python -m research_v2.report
```

`init` and `select` refuse to overwrite frozen files. For a delivered-config rerun, retain the existing `configs/`, preserve or relocate `results/` yourself, run gates/setup, then skip init/select and use the recorded selection. The final selector corrects a worst-group criterion bug and therefore need not reproduce the historical frozen selection on a fresh project. `selection_repair` separately reconstructs this transparent exploratory supplement. It does not replace or relabel original confirmation. Performance runs append JSONL and resume by instance/method key. Failed rows remain in the denominator. To retry a numerical failure, make an explicit archived revision; it is not silently discarded.

`python -m research_v2.budget_bounds_audit` was used once to repair historical C-Budget support upper bounds while retaining original holdings; the final solver already uses the corrected LP dual bound. It is not required for newly computed rows. Original rows and the audit explanation remain archived. See `docs/IMPLEMENTATION_HISTORY.md` for both numerical and selection revisions.

For the same frozen G2/confirmation manifests in parallel, use `python -m research_v2.batch_runner` instead of the two serial commands. It runs three G2 partitions and five confirmation partitions, preserves completed method rows and merges only unique keys. The 24-hour limit is cumulative experiment-process wall time, not elapsed desktop time. Run the serial stages and independent audits separately; do not simultaneously dispatch duplicate manifests.

Some audit scripts deliberately refuse to overwrite their previous outputs. For a clean rerun, use a fresh results directory while retaining the frozen configuration files. The final source reproduces the final method definitions; historical numerical-source hashes are not complete historical source snapshots. Unknown heuristic lower bounds are not global certificates.

To rebuild the math reading copy: `python -m research_v2.build_theory_tex`, then compile `docs/THEORY_NOTES.tex` with the native editor when available or an existing TeX installation. The local native compiler had an environment error; the delivered PDF was successfully built with the existing TeX Live installation.

## Important implementation decisions

- G1 uses the original public affine family and fixed nested quadrature.
- An ESS/scramble diagnostic exposed inadequate fixed-grid resolution in G2. The planned fallback now uses nested importance quadrature with a 25% uniform proposal component, fitted using observed data, and an explicit prior/proposal density ratio. G2 uses 256 scenarios; G1 uses 128. This is an approximation, not a continuous coverage proof.
- Counts and reference probabilities cease to be equivalent with nonuniform quadrature masses. A-ATK/B-ATK are therefore explicit count-based diagnostic methods under the importance revision.
- Approximate QP oracles retain their lower/upper objective bounds. New tail objectives do not occupy a 95% confidence field.
- Confirmatory candidate selection uses independent future realized losses in development, never evaluator true covariance/regret. The test oracle and true parameter appear only in evaluation records.
- Historical code corrections and pre-importance experiments are archived. The same manifests and random seeds are retained after corrections. Protocol revisions are explicit; this is not an externally preregistered study.
- G3 bootstrap scenarios are a learned reference construction scored on a separate past half-window, not an exact parametric posterior. The EB prior fitted to G1/G2 is not silently transferred to this different bootstrap space.

All timings are measured on a shared desktop with single-thread BLAS. Numerical bounds use float64 cushions and independent checks, not interval arithmetic. Method names, uncertainty measures and evidence labels distinguish algorithm optimization, posterior integration and statistical performance.
