# Theory audit and novelty boundaries

Date: 2026-10-08. Audit performed within this research task, not by an external referee. This audit refers to the parent manuscript, which is not included in this algorithm/experiment release. Algorithmic proofs are reproduced in `docs/ALGORITHM_DERIVATIONS.md`; references to manuscript sections below describe that earlier audit.

## Mathematical findings

| Item | Audit outcome | Manuscript treatment |
|---|---|---|
| T1 localization | Strong monotonicity bounds individual shifts by t times drive / perturbed curvature. The baseline action has O(t²) regret, which also localizes the minimax action. | Full inequalities in Appendix B. |
| Polyhedral expansion | Requires a fixed finite representation, fixed baseline point, bounded rescaled directions, and a sufficiently small common t. Domain tangent directions are included. | These conditions are explicit; no arbitrary continuous-family uniformity claim. |
| Individual oracle cone | Hoffman bound plus uniform Lipschitz quadratic terms makes the local cone penalty exact for sufficiently small t. | Proved before using oracle improvement coefficients. |
| Intersection-cone step | Projection need not reduce all original losses. Correct proof discards nonnegative a_j/t and only then compares bounded smooth terms within O(t). | This is stated explicitly in main text and proof. |
| O(t³) remainder | Follows by combining O(t) rescaled oracle and center errors with t². Constants depend on the fixed finite family and polyhedral error-bound constants. | No claim of uniformity over a changing family or vanishing strong convexity. |
| Dual and support | Relative Slater gives the epigraph dual. Projected Carathéodory gives span(C_intersection)+1 active gradients. | A certificate support bound, not a scenario-search or statistical-prior support bound. |
| Statistical equality | Equivalent observation laws imply one output law. Jensen gives lower bound Gamma; a constant center attains it. | Kept separate from deterministic geometry. |
| Multi-market construction | H_j w0=v_j, equal diagonal, SPD at small t; simplex circumradius and pair radius give stated constants. | Actual d-asset covariance example; Helly itself marked classical. |
| T2 | Strong convexity and the global feasible-gradient metric prove square-root stability. Common g is essential. | Full proof; center-only gradient difference is not substituted. |
| T3 | Dataset mixture likelihood ratio gives coverage; compatible-group sufficient bound follows from affinity and a union bound. | No general matching rate or optimal log M claim. |
| Testing lower bound | Disjoint epsilon-good sets yield a test; pairwise testing is insufficient for general multi-model incompatibility. | Both pair and equivalent-group failure bounds included. |
| T4 | Affinity in parameters makes oracle value concave and regret convex. A valid retained-cell union covers the original likelihood set. | Every vertex is retained; truth need not be a grid point. |
| Model approximation | Target regret stability alone is insufficient for likelihood coverage transfer; also need TV control for the whole observed experiment. | Both r and eta conditions stated, no empirical estimate falsely called certified. |
| T5 | Standard statistical decision minimax specialization, separate expected-risk criterion. | Appendix only; no experimental LF-PQP claim. |
| New computational bound | Zero-mean grouped Gaussian Hessian has a uniform norm bound using cell eigenvalue lower bounds and scatter trace. | Full derivation added; compared with first-derivative upper bound. |

No contradiction to T1–T4 was found under the narrowed explicit assumptions. This is not proof of priority or a guarantee that no proof detail needs refinement. The finite-sample group bound is elementary once compatibility is defined; its independent novelty is limited. The strongest candidate result remains the heterogeneous local geometry and its relation to indistinguishable markets. An external expert should examine whether it is a direct corollary of established epi-differentiability/minimax perturbation results.

## Primary-source comparison and reading depth

| Source | Material inspected | Relation and limitation |
|---|---|---|
| [Hauser–Krishnamurthy–Tütüncü](https://arxiv.org/html/1305.0144v2) | Full text, finite/polytopic sections and convexification argument | Regret center and vertex reduction are prior art; our contribution cannot be their renaming. |
| [Wang–Glynn–Ye](https://arxiv.org/html/1307.6279) | Full text framework and likelihood calibration sections | Likelihood-defined robustness predates this project. Our finite-sample ratio construction and incomplete experiment are more specific, not a new generic paradigm. |
| [Universal Inference](https://arxiv.org/html/1912.11436v4) | Full text coverage construction | Nonasymptotic likelihood-ratio calibration is a standard borrowed component. |
| [Yata](https://arxiv.org/html/2111.04926) | Setup and main-result sections | Binary welfare decision with known Gaussian experiment covariance; partial identification can already support optimal decisions. Not subsumed by our portfolio theorem. |
| [Christensen–Moon–Schorfheide](https://arxiv.org/html/2204.11748) | Main optimality statements and plug-in/quasi-Bayes comparison | Joint identification and estimation uncertainty is not a new observation here. Manuscript cites the verified preprint version instead of relying on an unverified journal status. |
| [Duchi–Ruan](https://arxiv.org/html/1612.05612) | Perturbation and local minimax theorem sections | Local decision sensitivity and statistical lower bounds are mature; hidden equivalent baselines and critical-cone intersection distinguish the present stated problem. |
| [Degenne–Koolen](https://arxiv.org/pdf/1902.03475) | PDF abstract and main framework | Multiple acceptable answers already change information complexity. This draft does not claim their adaptive complexity theorem as a consequence. |
| [PEAR](https://proceedings.mlr.press/v306/lee26q.html), [preprint full text](https://arxiv.org/html/2605.01361v1) | Official metadata and full preprint framework | Fixed-base, regular active-constraint sensitivity is a special geometry, not a baseline that should be incorrectly adapted to unknown-Q GMV. No PEAR training reproduction in this pilot. |
| [Lounici–Pacreau](https://arxiv.org/abs/2306.00752) | Primary metadata/abstract, prior reading of experiment section | Missing covariance estimation is an essential neighbor; not reproduced as a matched block-mask estimator in the current small pilot. |

This is a targeted nearest-neighbor review, not an exhaustive systematic novelty search. No claim of strict dominance over these papers is made. No theoretical containment, superior rates, empirical improvement, or robustness results are inferred from a desired research narrative.

## Implementation audit history

1. Initial Gate 0 found an open dual gap (~2.3e-7) despite successful optimizer termination. Normalizing the small dual objective fixed this; the audited case then closed to ~2.6e-11. Performance tests began after the corrected gate passed.
2. Both first-derivative and uniform-Hessian cell likelihood bounds are analytic upper bounds. Sampled checks are secondary. No local Hessian plug-in was passed off as a rigorous remainder.
3. The first performance run exposed an arbitrary MLE tie convention in unobserved coordinates. Final code uses the box midpoint only where all observed covariance derivatives are exactly zero. The observed likelihood is checked unchanged. All same-seed experiments were rerun; original rows remain archived.
4. Statistical calibration, cover conservatism, numerical optimality and misspecification are recorded separately. The outer-center dual lower bound is never used as the lower bound for the original likelihood set.
5. The native LaTeX compiler could not find its standard directories. The source remains available in the app; final PDF uses the pre-existing TeX Live installation with unmodified official style files.
6. PDF visual QA exposed a substantive numerical audit failure: the original boundary curve did not approach its claimed coefficient. At an all-bound vertex, selecting the budget multiplier from a median gradient gave a valid but extremely loose dual bound; the original Gate 1 allowed this large gap to mask coefficient failure. The corrected solver uses the KKT multiplier interval, Gate 0 includes a vertex regression case, and Gate 1 first requires a small rescaled gap. Final d=6/d=20 boundary ratios at the smallest t are approximately 0.24549/0.24846 versus theoretical 0.25. The incorrect pass and initial records are archived in `results/initial_boundary_bounds`, and all dependent performance stages are rerun. This is an implementation/audit correction, not a change to the theoretical coefficient.

## Findings that limit the current paper

- The continuous outer cover did not shrink in the registered pilot family, even with 128 cells in the stress setting.
- All tolerance-success events are trivial relative to the full-box ambiguity in this family. Their frequency does not demonstrate successful information acquisition.
- Model averaging is the strongest average-regret comparator here; its prior matches the uniform synthetic parameter distribution. LCRC has not demonstrated a favorable broad robust-versus-average trade-off.
- A 20-repeat stress study cannot validate 95% coverage. The model-misspecified rows deliberately have no theorem-level confidence interpretation.
- Theoretical and numerical certificates use double precision plus cushions, not machine-verified interval bounds.
- The final manuscript is a research draft with a genuine negative algorithmic result, not a submission-ready success narrative.
