# Limitations and deviations

- The stochastic sweep reaches T=8 because pathwise simultaneous TV certification has 2^T categories; it is a finite-state theorem reproduction, not a language-model benchmark.
- Soft-O hides constants and polylogarithms, so operation counts and their normalized scale are reported rather than fitting wall-clock time alone.
- Experiments cannot replace the paper's universal proof; Appendix-F inequalities are source-audited and the implementation is independently exhausted on a small augmented space.
