# Claim 1 method

The fixed cumulative verifier recomputes the construction from source, audits
the required assumptions, compares the observed law with an independently
computed reference law, and runs a negative control designed to violate a
specific premise.  It exits nonzero if the claim contract or control behavior
changes.

Formal run command: `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py`.
