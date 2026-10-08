# Algorithm derivations and computational certificates

These notes explain the implemented algorithms and their mathematical bounds. They are research derivations, not an externally refereed proof audit. The full local-geometry theorem belongs to the parent manuscript; this public package focuses on the computational and statistical steps used by the experiments.

## 1. Decision ambiguity

Let the common feasible set be a nonempty compact polytope. For a model parameter theta, define

$$f_\theta(w)=\tfrac12 w^\top Q_\theta w-\mu_\theta^\top w+g(w),\qquad V_\theta=\min_{w\in\mathcal W} f_\theta(w),$$

$$R_\theta(w)=f_\theta(w)-V_\theta,\qquad \Gamma(A)=\min_{w\in\mathcal W}\sup_{\theta\in A}R_\theta(w).$$

The penalty g is common and convex; it may include turnover. The pilot uses zero means, g=0, and capped long-only portfolios. Matrices are uniformly positive definite in the implementation. Then scenario oracles are unique, and compactness ensures the center exists for compact parameter sets with continuous coefficients.

Gamma is a **decision ambiguity**, not a claimed minimal sufficient statistic. Gamma(A) <= epsilon means that all models in A share a portfolio with regret at most epsilon; it does not require identifying which model is true.

## 2. Complete-mask likelihood

Assume independent zero-mean Gaussian return vectors. The observation masks are fixed, or exogenous with a parameter-independent distribution. Condition on the masks. For each distinct observed coordinate set S, let n_S be its row count and T_S the sum of observed outer products. Writing Q_S for the observed principal covariance submatrix gives

$$\ell(\theta)=-\tfrac12\sum_S\left[n_S\{|S|\log(2\pi)+\log\det Q_{\theta,S}\}+\operatorname{tr}(Q_{\theta,S}^{-1}T_S)\right].$$

**Derivation.** Multiply the marginal Gaussian densities of each observed row, take logarithms, and group rows with identical masks. Completely unobserved rows contribute one to the density and zero to the log likelihood. The within-row joint covariance is retained; pairwise sample counts alone are insufficient to specify this experiment.

Two models with identical covariance submatrices on every observed mask have identical likelihoods on every possible dataset. New joint observations can distinguish their cross-block covariances. `MaskLikelihood` implements the grouped expression and its affine-parameter gradient.

## 3. Finite-sample likelihood calibration

Choose anchor parameters and weights a_j before observing the inference data, with positive weights summing to one. Define a mixture of **whole-dataset densities**:

$$q(D)=\sum_j a_j p_j(D),\qquad C_\delta(D)=\{\theta:p_\theta(D)\geq\delta q(D)\}.$$

For any true model in the family, under domination,

$$\mathbb E_\theta\frac{q(D)}{p_\theta(D)}=1,\qquad
\mathbb P_\theta\{\theta\notin C_\delta(D)\}\leq\delta.$$

**Proof.** The expectation is the integral of the probability density q; Markov's inequality bounds the event q/p > 1/delta. In non-common-support experiments the expectation is at most one, which is still sufficient. Gaussian densities here are positive. The anchors need not contain the true parameter. A data-fitted numerator would require a separate valid construction, such as sample splitting; it cannot be substituted into this argument unchecked.

The implemented threshold is log(delta) + logsumexp(log(a_j)+ell_j). A product of per-row mixtures would describe a different model with a newly drawn latent parameter at every row.

On the coverage event, any data-dependent action with a verified upper bound U on its regret throughout C_delta satisfies true regret <= U. Consequently, the probability of issuing a false certificate R_true > U is at most delta under the stated model assumptions. This remains a valid statement if U is wide; coverage alone says nothing about usefulness.

## 4. QP oracle bounds and their direction

Consider the positive-definite QP with c=-mu:

$$V=\min_{Aw=a,\,Gw\leq h}\{\tfrac12w^\top Qw+c^\top w\}.$$

A feasible action gives the upper bound V_U=f(w). For any unrestricted equality multiplier y and nonnegative inequality multiplier s, completing the square in the Lagrangian gives

$$V_L=-\tfrac12(c+A^\top y+G^\top s)^\top Q^{-1}(c+A^\top y+G^\top s)-a^\top y-h^\top s\leq V.$$

Indeed the expression is the unconstrained infimum of the Lagrangian, which never exceeds any feasible objective. Thus V_L <= V <= V_U without requiring optimal multipliers.

For turnover kappa ||w-b||_1, use its support representation max over ||z||_infinity <= kappa of z'(w-b). Any such z yields the same lower bound with c replaced by c+z and an additional term -z'b.

**Direction matters:** an upper bound on regret is f(w)-V_L. Subtracting the approximate primal oracle objective V_U would underestimate regret.

For a capped simplex vertex there may be no free coordinate. With h_i=(Qw-mu)_i, the equality multiplier must satisfy

$$\max_{i:w_i=0}(-h_i)\leq y\leq\min_{i:w_i=u_i}(-h_i).$$

The final implementation uses this interval rather than a median gradient. The earlier choice produced loose valid lower bounds and a falsely passing structural diagnostic; archived records are labeled accordingly.

The formulas are exact in real arithmetic. `solve_qp` reports residuals and adds a floating-point cushion. General constrained solver outputs are approximately feasible, and the cushion is not a formal feasibility-repair theorem or interval-arithmetic guarantee. Numerical gaps and independent reference checks must be inspected.

## 5. Regret-center dual through ordinary QPs

For M scenarios, compact convex feasibility and convex losses allow convex minimax duality:

$$\Gamma=\max_{\pi\in\Delta_M}\left\{\min_{w\in\mathcal W}\sum_j\pi_j f_j(w)-\sum_j\pi_j V_j\right\}.$$

**Derivation.** The maximum of finitely many regrets equals the maximum of their simplex-weighted average. Interchanging min and max is justified by compact convex domains and continuity, convexity in w and affinity in pi. Since the penalty is common, the inner problem has averaged Q and mu and remains an ordinary QP (or the same QP with common turnover).

For any feasible w and simplex pi, scenario oracle bounds and an averaged-QP lower bound give

$$U(w)=\max_j\{f_j(w)-V_{j,L}\},\qquad L(\pi)=V_{\pi,L}-\sum_j\pi_j V_{j,U},$$

$$L(\pi)\leq\Gamma\leq U(w).$$

**Proof.** The right inequality uses feasibility and V_j >= V_j,L. For the left, the weighted minimum uses V_pi >= V_pi,L and subtracts V_j <= V_j,U. Neither inequality depends on successful optimizer termination. With exact oracles, a simplex supergradient is the vector of scenario regrets of the averaged-QP minimizer, by Danskin's theorem.

`center` uses normalized SLSQP on this concave dual, caches QPs, and retains the best primal upper and dual lower bounds independently. Approximate oracle objectives/gradients guide optimization; the independently evaluated bounds determine the reported gap. `direct_center` solves a separate conic epigraph problem for auditing; it uses approximate oracle constants and is a numerical reference, not an additional exact certificate.

## 6. Continuous outer covers

Let Q_theta=Q_0+sum_r theta_r E_r (and optionally affine mu) on a compact box. A cell B has midpoint c and half-widths h. It is discarded only when a valid upper bound on sup_B ell is below the likelihood threshold. Otherwise it is retained or subdivided. The retained union O then contains C_delta, including off-grid true parameters.

For each mask let s_S be a positive lower eigenvalue bound throughout the cell and define rho_S=sum_r h_r ||E_r,S||. Affinity and convexity of the PSD cone make the minimum over all cell vertices a valid eigenvalue lower bound throughout the cell.

Differentiation gives

$$D\ell_S[H]=\tfrac12\operatorname{tr}\{(Q_S^{-1}T_SQ_S^{-1}-n_SQ_S^{-1})H\}.$$

Using ||Q_S^-1|| <= 1/s_S and T_S positive semidefinite gives the first-derivative bound

$$\sup_B\ell\leq\ell(c)+\tfrac12\sum_S\rho_S\left(\frac{\operatorname{tr}T_S}{s_S^2}+\frac{n_S|S|}{s_S}\right).$$

A second derivative differentiates each inverse via D(Q^-1)[K]=-Q^-1 K Q^-1. The log-determinant contribution is bounded by n_S |S| ||H|| ||K||/(2 s_S^2). The two scatter terms together are bounded by tr(T_S) ||H|| ||K||/s_S^3. Therefore

$$|D^2\ell_S[H,K]|\leq\|H\|\|K\|\left(\frac{n_S|S|}{2s_S^2}+\frac{\operatorname{tr}T_S}{s_S^3}\right).$$

Taylor's theorem along the entire cell segment yields the alternative bound

$$\sup_B\ell\leq\ell(c)+\sum_r h_r|\partial_r\ell(c)|+\tfrac12\sum_S\rho_S^2\left(\frac{n_S|S|}{2s_S^2}+\frac{\operatorname{tr}T_S}{s_S^3}\right).$$

`cell_bound` takes the smaller of the two valid bounds and adds a numerical cushion. A Hessian evaluated only at c would not justify this remainder.

For fixed w, f_theta(w) is affine in theta, while V_theta is concave as an infimum of affine functions. Hence R_theta(w) is convex. Every cell point is a convex combination of its vertices, so its regret is at most the largest vertex regret. The continuous outer-set regret center therefore reduces exactly to **all** retained vertices. Dropping vertices arbitrarily invalidates the argument. A cell budget caps refinement effort, not the number of necessary vertices by deletion.

## 7. Separate the three gaps

Let I consist of points actually verified to meet the likelihood threshold. Then

$$I\subseteq C_\delta\subseteq O,\qquad \Gamma(I)\leq\Gamma(C_\delta)\leq\Gamma(O).$$

An inner-center dual bound L_I and outer-center primal bound U_O therefore sandwich Gamma(C_delta). By contrast, an outer-center dual bound L_O only lower-bounds Gamma(O); it need not lower-bound Gamma(C_delta).

- `numerical_gap = U_O-L_O` measures outer-center optimization accuracy.
- `refinement_gap = U_O-L_I` is a total sandwich width, including optimization error; it is not a pure discretization error.
- Statistical confidence comes from the likelihood ratio argument, not either optimization gap.

None of these deterministic optimization lower bounds is automatically a statistical minimax lower bound.

## 8. Stability for approximate model representations

Suppose objectives f and f' share g and are alpha-strongly convex on the affine span T of the feasible set. Define

$$r=\sup_{w\in\mathcal W}\|P_T[(Q-Q')w-(\mu-\mu')]\|.$$

Then for every feasible w,

$$|\sqrt{R_f(w)}-\sqrt{R_{f'}(w)}|\leq\frac{r}{\sqrt{2\alpha}}.$$

**Proof.** Write d=f'-f; its restriction to the feasible set is r-Lipschitz. Let x minimize f and x' minimize f'. Strong convexity gives ||w-x|| <= sqrt(2 R_f(w)/alpha). Also

$$f'(x)-f'(x')\leq-\tfrac\alpha2\|x-x'\|^2+r\|x-x'\|\leq\frac{r^2}{2\alpha}.$$

Decomposing R_f'(w) around x and applying these inequalities gives

$$R_{f'}(w)\leq R_f(w)+r\sqrt{2R_f(w)/\alpha}+r^2/(2\alpha)=(\sqrt{R_f(w)}+r/\sqrt{2\alpha})^2.$$

Swap f and f' to conclude. Thus a representation certificate U can be enlarged to (sqrt(U)+r/sqrt(2 alpha))^2. This transfers objective regret only. Transferring statistical coverage to a misspecified observation law additionally needs control of that law (for example, total variation <= eta for the whole experiment, losing at most eta in coverage). The S3 misspecification rows do not estimate such a theorem-level eta.

## 9. LCRC pseudocode and implemented limits

```text
input: masked data, fixed affine family, delta, cell budget, QP tolerance
form whole-dataset anchor-mixture likelihood threshold
start with the full parameter box
for each pending cell:
    compute a uniform likelihood upper bound
    record its midpoint as an inner point only if its likelihood passes
    discard the cell only if its upper bound fails
    otherwise split while budget permits, or retain it
collect and deduplicate every retained-cell vertex
check candidate inner points against the actual likelihood threshold
solve the outer finite-scenario regret center using QP dual iterations
solve the inner finite-scenario center for its lower bound
return weights, U_outer, L_outer, L_inner, residuals, gaps, cover diagnostics
```

The default pilot uses p=3 and 32 cells. It exhausts the chosen refinement budget rather than stopping early at epsilon, to compare fixed computational budgets. The finite-family variant applies the same threshold directly to scenario likelihoods. The pilot uses a public low-dimensional correctly specified Gaussian family; this is a substantial modeling assumption. A general high-dimensional covariance model would make cell enumeration expensive.

## Prior work and claim boundaries

- [Hauser, Krishnamurthy and Tutuncu: Relative Robust Portfolio Optimization](https://arxiv.org/abs/1305.0144): finite-scenario relative robustness and regret-center optimization are direct prior art.
- [Wang, Glynn and Ye: Likelihood Robust Optimization for Data-Driven Problems](https://arxiv.org/abs/1307.6279): likelihood-based robustness is established.
- [Wasserman, Ramdas and Balakrishnan: Universal Inference](https://arxiv.org/abs/1912.11436): finite-sample likelihood-ratio calibration is a borrowed statistical principle.

The research question concerns when incomplete observations suffice for a common near-optimal portfolio, and how hidden geometry affects that requirement. This package does not establish a generally minimax-optimal information rate, minimal statistical sufficiency, formal floating-point certification, or an empirical advantage for the present LCRC implementation.
