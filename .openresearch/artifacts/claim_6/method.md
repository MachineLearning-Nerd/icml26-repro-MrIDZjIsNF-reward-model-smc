# Claim 6 method

The verifier implements Algorithm 2 literally on the audited product potential
`V(s_1:t)=r^(sum s_i)` and a fair binary reference.  It uses the exact
binomial sufficient statistic for each `M`-candidate pool, so no candidate-level
approximation is introduced.  Across 200,000 independent chains per horizon it:

1. records the Appendix-F good event for every proposal and every time step;
2. conditions the output law on that event;
3. attaches a 99.9% simultaneous multinomial TV radius;
4. evaluates the exact binomial probability of the full good event;
5. records literal `M*T*H` operations and the theorem's complexity scale; and
6. exhaustively enumerates a separate tiny augmented state space to verify
   detailed balance, stationarity, and the target path marginal.

Formal run command: `uv sync --frozen && .venv/bin/python repro/src/verify_smc.py`.
