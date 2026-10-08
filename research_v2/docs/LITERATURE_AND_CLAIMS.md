# Nearest-neighbor and claim audit

Primary sources checked on 2026-10-08. This is a targeted review, not proof of priority. The research notes distinguish exact statements under their assumptions from numerical demonstrations and unresolved extensions.

| Work / primary source | Relevant established contribution | What this project must distinguish |
|---|---|---|
| [Fan et al., Average Top-k, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6c524f9d5d7027454a783c841250ba71-Abstract.html) | Average largest-k loss and convex formulations | No novelty claim for replacing max by Top-k |
| [Rockafellar–Uryasev, CVaR](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf) | Threshold/hinge optimization of tail risk | Probability-mass tails and fractional threshold atoms are borrowed tools |
| [Wu–Zhu–Zhou, Bayesian Risk Optimization](https://arxiv.org/abs/1609.08665) | Risk functionals, including CVaR, applied to posterior model uncertainty; asymptotic results | Posterior CVaR by itself is prior art; boundary interaction with changing model probabilities needs a separate result |
| [Christensen–Moon–Schorfheide, partially identified payoffs](https://www.restud.com/wp-content/uploads/2026/02/MS33570manuscript.pdf) | Average treatment of identified parameters, conditional minimax for unidentified parameters, discrete decision asymptotics | B-Max is a continuous-portfolio analogue, not a transfer of the original optimality theorem; preserve outer estimation uncertainty |
| [Wang–Glynn–Ye, likelihood robustness](https://web.stanford.edu/~glynn/papers/2016/WangGY16.html) | Data-driven likelihood-based ambiguity | Evidence penalties and reverse-KL scenario envelopes are adaptations; scenario reference weights are not observed category counts |
| [Li et al., tilted losses](https://www.jmlr.org/papers/v24/21-1095.html) | Exponential tilting and smooth interpolation of loss aggregation | C-Soft is a mature component; its rare-model boundary scale is the relevant question here |
| [Shapiro–Zhou–Lin, Bayesian DRO](https://arxiv.org/abs/2112.08625) | Bayesian model uncertainty combined with distributional robustness | Do not claim that posterior learning plus robustness is new in general |
| [Conditional Distributionally Robust Functionals](https://optimization-online.org/2022/05/8922/) | Conditional risk/robust functionals and their construction | Fixed marginal and conditional ambiguity deserve direct comparison; conditional envelopes alone are not a new paradigm |
| [Hauser–Krishnamurthy–Tutuncu, relative robust portfolios](https://arxiv.org/abs/1305.0144) | Relative robustness and scenario regret centers | The weighted-QP dual and finite regret optimization remain established components |
| [PEAR](https://proceedings.mlr.press/v306/lee26q.html) | Fixed-base decision sensitivity tied to constrained geometry | Do not present a stable-face quadratic projection as the new result; changing probability scales and heterogeneous boundary cones are the proposed distinction |
| [Shyamalkumar–Wang, Robust Optimal Portfolio in a Mixture Setting with Partial Ambiguity](https://arxiv.org/abs/2603.00851) | Portfolio optimization with partly known mixture components, uncertainty in other components/weights, variance/CVaR objectives and numerical algorithms | Particularly close motivation: reliable and ambiguous risk components. Its mixture-of-return-distributions target differs from our model-regret posterior tail and local boundary-response theorem. This distinction is a research positioning argument, not proof of superiority or strict containment |

Reading depth: the supplied sources and their official abstracts/main formulations were checked; the Bayesian risk and partial-identification formulations were inspected in the primary full text. The additional 2026 partial-ambiguity portfolio paper's introduction, objective and algorithm summary were inspected. No claim is made that every proof in each paper has been independently reproduced.

Additional sensitivity/financial-risk neighbors, checked at their publisher abstracts:

| Work | Assumptions / target / known conclusion | Reduction or unresolved comparison |
|---|---|---|
| [Poliquin–Rockafellar, A Calculus of Epi-Derivatives Applicable to Optimization](https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/calculus-of-epiderivatives-applicable-to-optimization/1D23CEC59BECDC0A9E24ACBE5C2D6AD4) | Extended-real objectives including constraints; epi-derivatives describe sensitivity subproblems | The present finite-family proof uses this general style of variational analysis. An explicit reduction to a published calculus theorem remains to be checked before claiming a new sensitivity theorem |
| [Römisch–Wets, Stability of epsilon-approximate Solutions to Convex Stochastic Programs](https://epubs.siam.org/doi/abs/10.1137/060657716) | Perturbed probability laws and convex stochastic objectives; Lipschitz stability under integrand-generated distances | Our singular probability-versus-action scale is more specific; no general strengthening of their stability result is established |
| [Bayesian portfolio selection using VaR and CVaR](https://www.sciencedirect.com/science/article/pii/S0096300322002041) | Posterior predictive portfolio returns; quantile-based portfolio weights | Return-tail risk is different from a tail across model-specific regrets, though both use Bayesian uncertainty and portfolio decisions |

These abstracts establish relevant prior-art categories, not an exhaustive theorem-by-theorem exclusion of overlap. In particular, an elementary-looking application of an existing second-order epi-calculus could lower the novelty of our finite-family result even if its probability threshold is useful in this application.

## What could carry the contribution

1. A finite-family probability-scale limit with a hard-cone / linear-penalty / disappearance decomposition, with a complete epi-limit proof.
2. A conditional extension that correctly handles positive outer masses, and a separate vanishing-outer-group result whose penalty is itself a conditional tail.
3. A demonstrated contrast between CVaR and entropy: polynomial rarity may release a CVaR boundary but not a quadratic-temperature entropic boundary.
4. A complete-mask affinity bound linking posterior odds to informative observations, with a carefully delimited testing lower-bound subproblem.

These derivations still need a specialist priority review against general parametric convex optimization, second-order epi-differentiability and stochastic-programming sensitivity. The present statistical lower bound does not establish general portfolio minimax optimality, and the implemented conditional-tail algorithm has no automatic claim to attain every information lower bound.

## What would not carry it

Renaming CVaR, adding likelihood weights to an existing regret center, using the posterior mean covariance, combining Bayes with minimax without a new result, or showing an advantage only against an intentionally mismatched prior. Novelty and empirical performance are separate axes and are reported separately.
