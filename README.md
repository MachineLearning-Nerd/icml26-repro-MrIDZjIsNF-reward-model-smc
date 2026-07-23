# Claim-faithful CPU reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/blob/master/notebooks/reward_model_smc.py)

This repository reproduces all six theoretical claims selected from
[*On the Power of (Approximate) Reward Models for Inference-Time Scaling:
Sequential Monte Carlo and Beyond*](https://arxiv.org/abs/2602.01381).
The previous live judge score is still **0/12**; the results below are
reproduction verdicts and a forecast, not points awarded by the judge.

The strongest test implements the paper's actual resampling-pool
Metropolis–Hastings Algorithm 2. The paper target is conditional
`δTV ≤ 0.10`; across horizons `T=3,4,5,6,8`, the observed 99.9% simultaneous
TV upper bounds are `0.0082, 0.0092, 0.0121, 0.0185, 0.0367`. The exact
good-event probabilities are at least `0.999999999999965`, and a separate
110-state augmented-space enumeration checks detailed balance to
`2.17e-19`. Inverting the MH ratio raises TV to `0.16265`, so the negative
control fails as intended.

The other checks execute the Appendix-C oracle lower-bound construction,
evaluate the literal Theorem 5.1 particle threshold, and exhaust terminal
path laws. Results are **VERIFIED** for claims 1, 2, 3, 5, and 6. Claim 4 is
**FALSIFIED only as imported**: the paper's bound `TV ≤ 2Tε` is verified,
but the stronger judge-dataset inference “guidance fails once
`ε ≥ 1/(2T)`” has an assumption-satisfying counterexample with exact TV zero.

All experiments used local Apple CPU only. No GPU or Hugging Face CPU upgrade
was needed. The setup substitutes audited finite binary state spaces for
language-model inference; it tests the theorem mechanisms and exact
constructions, not an LLM benchmark or a machine-checked universal proof.

- [Illustrated claim-by-claim report](reports/reward-model-smc-reproduction/report.md)
- [Tutorial-style marimo notebook](notebooks/reward_model_smc.py)
- [Machine-readable claim contracts and evidence](.openresearch/artifacts/)
- [Reproduction command ledger](reports/reward-model-smc-reproduction/commands.md)

Run the notebook locally with `uv run marimo edit notebooks/reward_model_smc.py`
or `uv run marimo run notebooks/reward_model_smc.py`. It embeds the formal
results and does not rerun the expensive evidence suite.

## Experiment log

The command below is copied verbatim from each `orx exp status`; it is fixed
across the tree.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
| --- | --- | --- | --- | --- |
| `master` | Publication surface | Not run as an experiment (publication surface) | README, report, notebook, and exact published text mirror | — |
| [`orx/frozen-judged-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/tree/orx/frozen-judged-baseline) | Freeze the judged proxy baseline and uv lock | `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py` | Historical self-checks ran; 0/12 accepted live evidence | Local CPU, 31 s |
| [`orx/exact-finite-state-theorem-harness`](https://github.com/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/tree/orx/exact-finite-state-theorem-harness) | Exact laws, source audits, oracle construction, controls | `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py` | Claims 1,2,3,5 verified; imported claim 4 falsified; claim 6 blocked | Local CPU, 10 s |
| [`orx/statistical-scaling-stress-test`](https://github.com/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/tree/orx/statistical-scaling-stress-test) | Independent Monte Carlo, Wilson intervals, bootstraps | `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py` | Independently corroborated claims 1–5; claim 6 blocked | Local CPU, 10 s |
| [`orx/cumulative-evidence-and-resampling-pool-mh`](https://github.com/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/tree/orx/cumulative-evidence-and-resampling-pool-mh) | Implement Algorithm 2 and augmented-state checker | `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py` | Cumulative claims 1–6 pass their direct contracts | Local CPU, 30 s |
| [`orx/release-candidate-cumulative-evidence`](https://github.com/MachineLearning-Nerd/icml26-repro-MrIDZjIsNF-reward-model-smc/tree/orx/release-candidate-cumulative-evidence) | Log every raw metric; generate and validate release package | `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py` | 5 VERIFIED, 1 FALSIFIED; report/SVG/notebook/manifests valid | Local CPU, 50 s |

## Reproduce

```bash
uv sync --frozen
.venv/bin/python repro/src/verify_smc.py
```

The verifier exits nonzero if a claim contract, independent checker, negative
control, report render, notebook check, protected-Space subset check, text-only
allowlist, or secret scan fails.

---

# Original project note

OpenReview `MrIDZjIsNF`. arXiv `2602.01381`. Six claims / 12 possible judge
points. Owner: loop12pt.
