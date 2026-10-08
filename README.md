# LCRC: Portfolio Decisions from Incomplete Risk Information

Research code, algorithm derivations, and CPU synthetic experiments for **When Is Incomplete Risk Information Enough for Near-Optimal Portfolio Decisions?**

The central quantity is decision ambiguity: the smallest worst-case regret of one portfolio across the risk models that observations have not excluded. LCRC (likelihood-calibrated regret center) combines a calibrated likelihood set, a conservative continuous-parameter cover, and a finite-scenario regret center.

**Current finding:** the tested continuous cover is too conservative. It retains the entire parameter box, so LCRC coincides with full-set relative robustness. Its pooled S2 mean regret is approximately **74% higher** than likelihood-weighted model averaging. The preset success tolerances are attainable without data. These are preliminary synthetic results, not evidence of algorithmic superiority or a completed ICML submission.

## Read first

- [Algorithm derivations and certificate proofs](docs/ALGORITHM_DERIVATIONS.md)
- [Experimental protocol](docs/EXPERIMENTAL_PROTOCOL.md)
- [中文实验解读](REPORT_ZH.md)
- [Theory and implementation audit](THEORY_AUDIT.md)
- [Result inventory and superseded-run history](results/README.md)

![Same-information comparison](results/comparison.png)

The vertical axis is regret as a percentage of a fixed development risk unit, **not a return percentage**. Methods share each masked dataset. Model averaging's uniform prior matches the synthetic truth distribution.

## Reproduce

Python 3.12 is recommended. Create an environment and install the recorded dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_check.py
```

`smoke_check.py` runs numerical Gates 0 and 1 in a temporary output directory and checks the committed result inventory. It leaves the original experimental timings and records intact.

Run the complete exploratory protocol serially from the repository root:

```bash
python experiments.py gate0
python experiments.py s1
python experiments.py tune
python experiments.py pilot
python experiments.py S2
python experiments.py S3
python experiments.py summarize
python audit_extra.py
python plot_results.py
```

These commands overwrite the corresponding files under `results/`. Preserve a previous run if its original timing matters. The source limits BLAS to one thread. No GPU or external dataset is required. Recorded package and hardware versions are in [environment.json](environment.json); cross-platform installation and performance have not been validated.

## Code map

| File | Purpose |
|---|---|
| `core.py` | Mask-grouped Gaussian likelihood, bounded QP oracle, regret center, finite/continuous LCRC, MLE, model averaging, missing-data EM |
| `experiments.py` | Gates, structural constructions, development tuning, timing, paired S2/S3 comparisons, summaries |
| `audit_extra.py` | Fixed post-pilot fine-grid and quadrature checks; no retuning |
| `plot_results.py` | Recreate the three scientific figures from saved results |
| `smoke_check.py` | Non-destructive numerical and artifact checks |
| `config.json` | Implemented exploratory settings, recorded after execution; not a preregistration |

The evaluator holds the synthetic true covariance and oracle. Decision functions receive only masked observations and the public family. Saved evaluation rows include truth for auditing. The dataclasses in `core.py` document interface fields; the pilot functions also expose direct array-based APIs.

## Scope and provenance

This repository publishes the algorithm/experiment portion of a working research draft dated 2026-10-08. It includes new explanatory derivation notes and the original corrected experiment code and synthetic records. It excludes private market data, original conversation attachments, and the full manuscript. Real A-share evaluation and LF-PQP have not been run.

Finite-scenario min-max regret and likelihood robustness have direct prior work; neither is claimed as a new generic algorithm. See the [derivations](docs/ALGORITHM_DERIVATIONS.md#prior-work-and-claim-boundaries) for primary sources. Mathematical guarantees require the stated model assumptions. Floating-point bounds use numerical cushions and independent checks, not directed interval arithmetic. Archived pre-correction results are explicitly labeled and excluded from the final comparisons.
