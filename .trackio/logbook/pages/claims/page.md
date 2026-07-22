# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4f44957689b2", "created_at": "2026-07-22T03:04:57+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Under Assumption 3.2 (uniform Bellman error bound ε), if ε = O(1/T), SMC-based inference-time scaling attains a target TV error with particle/time complexity that is polynomial rather than exponential in the horizon T (Section 5, Theorem 5.1, Corollary 5.2).
2. Without reward guidance, the number of samples needed to hit the target region grows as Ω(L^(2T/3)), an exponential lower bound in T (Section 4, Theorem 4.1).
3. Even with a Bellman-error-bounded reward model, sampling complexity is lower-bounded by Ω((1+ε)^(2T/3)), showing guidance alone cannot remove exponential dependence unless ε shrinks with T (Section 4, Corollary 4.2).
4. For single-particle guided SMC, the total-variation error is bounded by 2Tε, so guidance fails to control error once ε ≥ 1/(2T) (Section 4, Theorem 4.3).
5. Theorem 5.1 establishes a particle complexity bound N ≥ L^6 T(1+ε)^(6(T-1))/(2δ_TV) for SMC to achieve TV error δ_TV (Section 5, Theorem 5.1).
6. A resampling-pool Metropolis-Hastings chain-based approach achieves the target accuracy with time complexity Õ(L T^3 log(1/δ) log(1/δ_TV)) (Section 6, Theorem 6.1).
