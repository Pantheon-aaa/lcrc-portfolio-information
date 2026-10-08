# Synthetic result records

Top-level files are the final corrected runs. They are generated entirely from public synthetic families; no market records are included.

- `gate0.json`: QP, likelihood, cover, stability and finite-model checks.
- `gate1.json`, `s1.csv`: 120 structural optimization records and gate outcome.
- `tuning.json`: independent development risk scale and shrinkage choice.
- `pilot_instances.parquet`, `pilot_timing.json`: ten-instance CPU timing pilot.
- `S2_instances.parquet`: 160 paired instances, seven methods, 1,120 rows.
- `S3_instances.parquet`: 160 setting instances, seven methods, 1,120 rows. Cell-budget variants reuse data and are not independent repetitions.
- `*_errors.json`: execution failures, retained even when empty.
- `summary.csv`, `paired.csv`: aggregate statistics and unadjusted paired intervals.
- `post_pilot_audit.json`: 16 fixed-instance grid/quadrature diagnostics.
- `*.png`: final figures.

## Superseded records — not final evidence

`initial_mle_ties/` preserves the first performance results, before a deterministic midpoint tie convention replaced arbitrary grid-edge MLE completions in exactly unobserved directions.

`initial_boundary_bounds/` preserves the earlier overly loose boundary-QP dual bounds and the falsely passing structural gate. The final solver uses the KKT multiplier interval at vertices; the final structural gate first rejects an overly wide numerical interval. Dependent stages were rerun using the same seeds. These old records are retained for auditability and must not be pooled with final rows.
