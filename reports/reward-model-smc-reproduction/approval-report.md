- Previous live judged score: `0/12`
- Conservative projected score range after the proposed change: `6–10/12`
- Best-supported possible new score: `12/12` — forecast only, not a judge result

# Publication approval report

The live audit on 2026-07-23 still finds both the Hugging Face Space head and
the judge head at `b675cbafc35867fc9212939818e54ff9225ac567`. The current
verdict dataset has all six claims marked inconclusive. Nothing in this report
claims that the live score has changed.

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 0 | 2 | MEDIUM | VERIFIED | Exact finite-state SMC over `T=6,12,24,48,96` gave particle-time slope 1.894 when `epsilon=O(1/T)` and an exponential negative control for constant epsilon. The remaining risk is that executable evidence covers a controlled nontrivial family, while the theorem is universal under its assumptions. |
| 2 | 0 | 2 | HIGH | VERIFIED | The Appendix C hidden-prefix oracle was implemented rather than replaced by a formula: actual no-guidance queries at `T=6,9,12,15` matched the exact success probabilities and the measured log slope 0.450 is close to the claimed 0.462 exponent. An exhaustive checker and a deliberately invalid oracle control passed and failed respectively. |
| 3 | 0 | 2 | MEDIUM | VERIFIED | Actual guided oracle searches on the paper's proof construction at the exactly executable integer case `epsilon=1`, `B=2` matched the exponential lower bound, with an independent enumeration checker and informative-oracle control. The paper's use of non-integer `B=1+epsilon` creates a remaining interpretation risk outside this case. |
| 4 | 0 | 2 | HIGH | FALSIFIED | The source theorem's upper bound `TV <= 2t epsilon` was independently verified, but the imported stronger statement that guidance necessarily fails once `epsilon >= 1/(2T)` is contradicted by a valid model with `T=10`, `epsilon=0.1`, and exact TV 0. This falsifies only the stronger imported consequence, not Theorem 4.3 itself. |
| 5 | 0 | 2 | MEDIUM | VERIFIED | The literal particle threshold was computed and tested at `T=3,5,8,12`; exact path enumeration gave TV below 0.05 at every threshold. The remaining risk is finite-family evidence for a universal sufficient condition. |
| 6 | 0 | 2 | MEDIUM | VERIFIED | The actual resampling-pool Metropolis-Hastings algorithm was run with 200,000 deterministic chains per horizon, `T=3,4,5,6,8`; all achieved the target event, the operation slope was 3.366, detailed balance and stationarity errors were below `2.1e-17`, and an inverted-acceptance negative control missed the TV target. Soft-O constants, finite horizons, and the theorem's broader quantification remain validation risks. |

## Release state

- Current total score: `0/12`.
- Conservative projected total score range: `6–10/12`.
- Best-supported possible total: `12/12`, strictly a forecast.
- Claims changed in the candidate evidence: all six move from the prior
  inconclusive evidence status; claims 1, 2, 3, 5, and 6 are assessed
  `VERIFIED`, and the imported claim 4 is assessed `FALSIFIED`.
- BLOCKED claims: none.
- LOW-confidence claims: none. No route-shortcut is being used; every claim
  has a direct contract, raw evidence, independent checker, and negative
  control.

## Experiment tree and winning revision

The frozen baseline was run once, followed by an exact finite-state branch and
a statistical stress-test sibling. The stronger exact branch was promoted into
the cumulative Algorithm 2 implementation and then into the formal release
candidate. All five runs are terminal and successful.

- Frozen baseline: historical toy/proxy suite, 31 seconds.
- Exact finite-state theorem harness: direct claims 1–5 evidence, 10 seconds.
- Statistical scaling stress test: independent stochastic corroboration, 10
  seconds.
- Cumulative evidence and resampling-pool MH: direct Algorithm 2 evidence, 30
  seconds.
- Release-candidate cumulative evidence: all claims, all artifacts, and release
  gates, 50 seconds.

The winning immutable experiment branch is
`orx/release-candidate-cumulative-evidence` at
`dcce92c8d65d0d21cad7f3c349c76a462ad1c878`. Reader-facing files are staged on
`publication/reward-model-smc-rc`.

Every experiment inherited the same command:

```text
uv sync --frozen && .venv/bin/python repro/src/verify_smc.py
```

The command ledger is `reports/reward-model-smc-reproduction/commands.md`.
The environment is Python 3.12.11, NumPy 2.5.1, one repository `.venv`, and the
committed `uv.lock`.

## Evidence and release gates

- Formal verifier runtime: 41.193267 seconds on an 8-logical-CPU Apple arm64
  local machine.
- Total OpenResearch outer-run time: 131 seconds.
- Hugging Face CPU runtime: 0 seconds; cpu-upgrade was not needed.
- GPU runtime: 0 seconds.
- Local and Hugging Face compute cost: `$0`.
- Durable claim evidence:
  `.openresearch/artifacts/claim_1/` through
  `.openresearch/artifacts/claim_6/`.
- Illustrated report:
  `reports/reward-model-smc-reproduction/report.md`.
- Self-contained checked notebook: `notebooks/reward_model_smc.py`.
- All five SVG figures parse successfully; `marimo check` passes.
- Secret scan: 134 files scanned, zero potential matches.
- Protected Space subset: all 17 files at the judged revision remain in the
  82-file candidate tree; no protected evidence page is overwritten.
- Upload staging is text-only.

## Exact Hugging Face upload allowlist

The canonical exact allowlist is
`.openresearch/artifacts/hf_upload_allowlist.json`, SHA-256
`403edd97b8b15cbfd489cbdf6d45bb9cd698249f1c041de4734e6c227a1a0298`.
It contains exactly 67 destination paths, byte sizes, file hashes, and a
`text_only: true` assertion for every file. Its scope is:

- Nine files for each of `claim_1` through `claim_6` under
  `evidence/release-2026-07-23/`.
- `runtime.json`, `sha256_manifest.json`, `source_snapshot.json`, and
  `verdicts.json` under `evidence/release-2026-07-23/`.
- `logbook.json`.
- `pages/index.md` and `pages/reproduction-2026-07-23/page.md`.
- Five SVG figures and `report.md` under
  `reports/release-2026-07-23/`.

No path outside that machine-readable allowlist is authorized by this report.

## Approval requested

Approve exactly this action: upload the 67 allowlisted text files to the
existing Space `DineshAI/MrIDZjIsNF` through the text-only Hugging Face API,
verify and report the resulting revision, mark the paper awaiting judge, then
mirror the approved reader-facing text artifacts to GitHub `master` and verify
the remote SHA. Do not create a second Space, publish any non-allowlisted file,
or report a score increase before a new live verdict exists.
