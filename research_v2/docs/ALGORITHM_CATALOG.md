# Algorithm catalog and execution trace

All methods share the supplied scenario and observation information. Reference masses, posterior masses and adverse decision weights have separate representations.
Objectives are implemented in `risk_envelopes.py`; convex optimization and independent bounds in `risk_center.py`; the direct Clarabel epigraph is in `native_conic.py`. Full specifications are frozen in `configs/selection.json` and protocol revisions.

| Algorithm | Functional / parameter | Conditional | Development rows | G1 rows | G2 rows | Confirm rows |
|---|---|---|---:|---:|---:|---:|
| A-Mean | mean; rho=1 | False | 24 | 240 | 1920 | 6480 |
| B-Mean | mean; rho=1 | True | 24 | 0 | 0 | 0 |
| A-Max | max; rho=0.5 | False | 24 | 240 | 1920 | 0 |
| B-Max | max; rho=0.5 | True | 24 | 240 | 1920 | 6480 |
| A-ATK:0.25 | count; rho=0.25 | False | 24 | 0 | 0 | 0 |
| B-ATK:0.25 | count; rho=0.25 | True | 24 | 0 | 0 | 0 |
| A-Tail-P:0.25 | tail; rho=0.25 | False | 24 | 240 | 1920 | 0 |
| A-Tail-U:0.25 | tail; rho=0.25 | False | 24 | 0 | 0 | 0 |
| B-Tail-P:0.25 | tail; rho=0.25 | True | 24 | 240 | 1920 | 6480 |
| B-Tail-U:0.25 | tail; rho=0.25 | True | 24 | 0 | 0 | 0 |
| A-Absolute:0.25 | tail; rho=0.25 | False | 24 | 0 | 0 | 0 |
| A-ATK:0.5 | count; rho=0.5 | False | 24 | 0 | 0 | 0 |
| B-ATK:0.5 | count; rho=0.5 | True | 24 | 0 | 0 | 0 |
| A-Tail-P:0.5 | tail; rho=0.5 | False | 24 | 240 | 1920 | 6480 |
| A-Tail-U:0.5 | tail; rho=0.5 | False | 24 | 240 | 1920 | 0 |
| B-Tail-P:0.5 | tail; rho=0.5 | True | 24 | 240 | 1920 | 6480 |
| B-Tail-U:0.5 | tail; rho=0.5 | True | 24 | 240 | 1920 | 0 |
| A-Absolute:0.5 | tail; rho=0.5 | False | 24 | 0 | 0 | 6480 |
| C-LP-Max:0.01 | max; lambda/u=0.01 | False | 24 | 240 | 1920 | 0 |
| C-LP-Tail:0.01 | tail; lambda/u=0.01 | False | 24 | 0 | 0 | 0 |
| C-Soft:0.01 | soft; tau/u=0.01 | False | 24 | 0 | 0 | 0 |
| C-Soft-Cond:0.01 | soft; tau/u=0.01 | True | 24 | 0 | 0 | 0 |
| C-LR:0.01 | lr; r=0.01 | False | 24 | 0 | 0 | 0 |
| C-LP-Max:0.1 | max; lambda/u=0.1 | False | 24 | 0 | 0 | 0 |
| C-LP-Tail:0.1 | tail; lambda/u=0.1 | False | 24 | 240 | 1920 | 0 |
| C-Soft:0.1 | soft; tau/u=0.1 | False | 24 | 240 | 1920 | 0 |
| C-Soft-Cond:0.1 | soft; tau/u=0.1 | True | 24 | 240 | 1920 | 0 |
| C-LR:0.1 | lr; r=0.1 | False | 24 | 0 | 0 | 0 |
| C-LP-Max:1.0 | max; lambda/u=1.0 | False | 24 | 0 | 0 | 0 |
| C-LP-Tail:1.0 | tail; lambda/u=1.0 | False | 24 | 0 | 0 | 0 |
| C-Soft:1.0 | soft; tau/u=1.0 | False | 24 | 0 | 0 | 0 |
| C-Soft-Cond:1.0 | soft; tau/u=1.0 | True | 24 | 0 | 0 | 0 |
| C-LR:1.0 | lr; r=1.0 | False | 24 | 240 | 1920 | 0 |
| C-Budget:0.25 | budget; budget fraction=0.25 | False | 24 | 240 | 1920 | 0 |
| C-Budget:0.5 | budget; budget fraction=0.5 | False | 24 | 0 | 0 | 0 |
| C-Budget:1.0 | budget; budget fraction=1.0 | False | 24 | 0 | 0 | 0 |
| H-RankThenP | naive; rho=0.5 | False | 24 | 0 | 0 | 0 |
| H-Conditional-RankThenP | naive; rho=0.5 | True | 24 | 0 | 0 | 0 |
| H-pR | pr; rho=0.5 | False | 24 | 0 | 0 | 0 |
| H-Conditional-pR | pr; rho=0.5 | True | 24 | 0 | 0 | 0 |
| H-FilterThenK | filter; rho=0.5 | False | 24 | 0 | 0 | 0 |
| H-Conditional-FilterThenK | filter; rho=0.5 | True | 24 | 240 | 1920 | 6480 |
| H-PostSampleK | sample; rho=0.5 | False | 24 | 0 | 0 | 0 |
| H-Conditional-PostSampleK | sample; rho=0.5 | True | 24 | 0 | 0 | 0 |

## Supplementary exploratory coverage

These rows do not enlarge the original confirmatory comparison family. The original confirmation had already been inspected when the selector was repaired.

| Method | Blend development | Repair screening | Reused test panel |
|---|---:|---:|---:|
| A-Blend | 24 | 2160 | 0 |
| B-Blend | 24 | 2160 | 0 |
| C-LP-Max:0.01 | 0 | 2160 | 6480 |
| H-PostSampleK | 0 | 2160 | 6480 |

## Aliases, triggered methods and additional comparators

- REF-MA is A-Mean, REF-MAX is A-Max. B-Mean is exactly A-Mean on a common joint posterior; both run in development as a correctness check.
- A-ATK and B-ATK coincide with reference tails only when the relevant reference atoms have equal mass and rho agrees with k/M. Under importance quadrature they are separate implemented methods.
- A-Blend and B-Blend are implemented. An erroneous worst-group selection criterion initially made the blend trigger unreachable. The corrected criterion activated both, and blend_development / repair_screen contain the supplementary runs. Original confirmation is kept separate.
- B-Plugin uses the outer node with largest discrete posterior mass and solves its conditional tail at rho=0.5. It is a grid-based plug-in comparator, not a continuous MAP optimizer.
- REF-EB learns a four-component prior mixture from independent development marginal likelihoods; it does not use true parameter labels.
- Equal Weight, a fixed diagonal EM shrinkage comparator, affine MLE, old LCRC (a fixed 24-instance subset), and four privileged finite-grid oracle priors are in baseline_audit. These do not replace same-information MA/EB comparisons.
- H-RankThenP and its conditional version use four fixed starts and derivative-free search. They have no global gap. Their maximum normalized start-value spreads in development were approximately 0.245u and 0.472u, respectively; this is substantial numerical instability, not merely absence of a convex gap. A future heuristic candidate must have all four finite starts within 1e-4u on the whole development panel. Neither qualifies.
- H-pR is a convex count Top-k objective applied to probability-times-regret. It has representation dependence, not an automatic nonconvexity problem.
- H-PostSampleK samples each finite posterior once and applies equal-count tails without multiplying the likelihood again. Additional fixed-action sampling convergence diagnostics are saved in posterior_sampling_tail.json.
- Conditional H variants preserve outer posterior group masses. Filtering first retains at least 95% posterior mass; subsequent integer k is ceil(rho*K), where K is retained count.
- B-Max main experiments maximize over supplied finite conditional scenarios. The high-resolution audit additionally uses exact hidden-box vertices for G2, keeping an approximate posterior outer marginal. Only that audit has exact hidden maximization.

## Evidence status

A nonzero row count means attempted execution, not a favorable result. Numerical exceptions, open gaps and heuristic statuses remain in the records. Performance eligibility and approximation sensitivity are discussed in REPORT_ZH.md. The separate boundary and embedded-information experiments are mechanism studies, not extra observations in the algorithm comparison.

## Explicit scope limits

No real A-share backtest, public push, DFL training, nonlinear-shrinkage campaign or LF-PQP comparison was performed in this round. Five-parameter cases are a fixed precision diagnostic, not a broad confirmation claim. General simultaneous outer/inner decay and continuous-nuisance optimality are open theoretical results.