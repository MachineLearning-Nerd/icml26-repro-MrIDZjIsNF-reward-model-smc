# Claim-faithful CPU reproduction — 2026-07-23

This additive release answers the live judge's six criticisms with executable
finite-state evidence. Formal evidence commit: `dcce92c8d65d0d21cad7f3c349c76a462ad1c878`. Fixed command:
`uv sync --frozen && .venv/bin/python repro/src/verify_smc.py`.

| Claim | Reproduction verdict | Evidence |
| --- | --- | --- |
| 1 | VERIFIED | Literal Theorem 5.1 sizing, T=6…96, operation slope 1.894 |
| 2 | VERIFIED | Actual Appendix-C no-guess oracle queries |
| 3 | VERIFIED | Exact integer proof instance B=2, ε=1 |
| 4 | FALSIFIED | Theorem bound holds; imported threshold consequence has a valid counterexample |
| 5 | VERIFIED | Literal sufficient N bound and independent path enumeration |
| 6 | VERIFIED | Actual Algorithm 2, conditional TV certification, augmented-state detailed balance |

These are evidence verdicts, not live judge points. The old pages remain
unchanged and reachable. Detailed text artifacts are under
`evidence/release-2026-07-23/`; the illustrated report is
`reports/release-2026-07-23/report.md`.
