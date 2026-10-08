# Experimental protocol

This is an exploratory CPU pilot dated 2026-10-08, not a preregistered study. `config.json` records implemented settings after execution. Algorithm code is unchanged from the corrected local research run; public-release additions provide documentation, plotting, and a non-destructive smoke check.

## Family and information access

The public affine covariance family has 20 assets and three parameters in [-1,1]^3. Seed 812 shuffles diagonal entries geometrically spaced between 0.65 and 2.8. One diagonal direction has operator norm 0.14; two cross-block directions each have norm 0.22. The covariance minimum eigenvalue is at least 0.07 throughout the standard box. Observation means are known zero, shared by all methods. Portfolios sum to one and have weights between zero and 0.2.

Each block contains two independent time points. A separate block observes the first ten assets at the first time and the other ten at the second. A joint block observes all twenty at the first time and nothing at the second. Each block uses twenty scalars. Changing the number of joint blocks therefore changes common information without changing marginal sample counts or the total scalar budget.

The evaluator draws continuous uniform parameters. Algorithm functions receive only the masked returns and public family. Full returns, true parameters, and oracle decisions are reserved for evaluation. Saved row-level records expose truth only for retrospective audit.

## Stages

| Stage | Design | Purpose |
|---|---|---|
| Gate 0 | Independent QP/conic checks, grouped likelihood, observation equivalence, off-grid retention, stability | Numerical and statistical implementation checks |
| S1 / Gate 1 | Dimensions 6 and 20; five seeds; four perturbation scales; three mechanisms | Hidden curvature, different boundary cones, multi-model compatibility; 120 records |
| Development | 24 risk-scale instances and 24 independent validation instances | Freeze risk unit and EM shrinkage |
| Pilot | Ten instances, all seven methods | CPU time and process peak memory |
| S2 | Block counts 64/256; joint counts 0/4/16/64; 20 paired repetitions per cell | Same-information method comparisons |
| S3 | Eight settings, 20 repetitions each | Off-grid cell budgets 8/32/128; covariance misspecification; t6 observations; wide box; zero/full joint information |
| Post-pilot audit | Repetitions 0 and 1 from all eight S2 cells | Finer 9^3 MLE grid and order-5/order-7 quadrature checks; no retuning |

The global generator seed is 20261008. Derived seeds and diagnostics are retained in every instance table. The S3 cell-budget variants reuse identical datasets, so they must not be treated as independent replications. Compatibility seeds permute an analytic construction; they are invariance checks. S1 numerical intervals must be tight before checking a theoretical coefficient.

## Seven paired methods

1. LCRC: delta=0.05, fixed uniform mixture over 3^3 anchors, default 32-cell conservative cover, all vertices retained.
2. Equal Weight: a feasible constant action.
3. EM + shrinkage: Gaussian conditional moments with known zero mean; 100 iterations, relative covariance tolerance 1e-6, eigenvalue floor 1e-7. The independent validation set selects from {0,0.1,0.25,0.5,0.75,1}; selection was 1, giving a diagonal estimator.
4. Affine-MLE + QP: exact masked likelihood, five starts selected from a fixed 3^3 grid, bounded L-BFGS-B. Exactly unobserved coordinates use the midpoint without changing likelihood. This is not a certified global MLE.
5. Full-set regret: relative robustness over the original parameter box vertices.
6. Likelihood-absolute: worst-case objective over the same outer set as LCRC, without subtracting scenario oracle values.
7. Model-average: fixed uniform continuous prior, five-point Gauss-Legendre product quadrature with volume weights, followed by a posterior-mean covariance QP. Its prior matches the truth generator. Refinement never changes its prior.

Structured methods share the same public family; EM does not have this structural information. An advantage over EM alone would not isolate an algorithmic contribution. DFL, LF-PQP, and a comprehensive nonlinear-shrinkage comparison are not implemented.

## Metrics and uncertainty

True covariance regret is divided by a fixed development risk unit (about 0.038575312). Epsilon is 5%, 10%, or 20% of this unit. Percentages of this unit are not investment returns. Results include mean/median/90th-percentile regret, the worst observed mean among four sign-based parameter subgroups, epsilon success, coverage, numerical gaps, runtime and execution failures.

Paired intervals use Student-t intervals over twenty independent repetitions within a setting. They are exploratory and unadjusted for multiple comparisons. Coverage uses exact Clopper-Pearson intervals; 20/20 has a 95% interval lower endpoint near 0.832, so it does not empirically establish 95% population coverage. Parameter subgroups do not provide a continuous-domain worst-case risk estimate.

The standardized t6 generator preserves covariance via a Gaussian row times sqrt(4/chi-square(6)). Covariance misspecification adds a symmetric perturbation of operator norm 0.12. Both departures invalidate the nominal Gaussian-family statistical proof, so coverage there is empirical only. The wide-box stress uses [-1.3,1.3]^3 and checks positive definiteness at vertices.

## Result interpretation

The top-level saved rows are final corrected runs. All covers retain the full parameter box, including the 128-cell setting. A post-pilot grid audit shows that the exact likelihood threshold does exclude some points; the cover's failure to shrink is therefore partly computational conservatism. The full-box upper bound is only about 1.853% of the risk unit, already smaller than the smallest epsilon. Perfect epsilon success is not evidence of learning.

The ten-instance timing pilot used one BLAS thread on a shared Windows desktop (Intel Core Ultra 5 125H). The median combined seven-method time was about 0.434 seconds, p95 0.583 seconds, and process peak working set about 162.6 MiB. These are local measurements, not cross-hardware guarantees. Cell refinement to 128 increased median LCRC time to about 7.60 seconds without a tighter cover.

The two archived result directories preserve superseded MLE ties and vertex-QP bound/gate issues. See `results/README.md` and `THEORY_AUDIT.md`. No seeds, tolerances, or truth distributions were changed to manufacture an improvement. Real A-share backtests have not been performed and no private data are shipped.
