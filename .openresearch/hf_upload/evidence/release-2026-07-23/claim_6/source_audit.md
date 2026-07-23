# Claim 6 source audit

Source: ar5iv HTML for arXiv:2602.01381, SHA-256 `1cf1d6e6c89a5fa9df919a4872166eb21db7e8b6d08ac419c37fdeda52b73fb3`.

Anchors: `alg2`, `S6.Thmtheorem1`, and the proof in Appendix F.

Algorithm 2 draws `M` reference candidates at each step, selects one in
proportion to its value, accumulates
`w <- w * V(prefix) / Zbar`, and accepts a complete proposal with
`min(1, w_acc*V(proposal)/(w_proposal*V(accepted)))`.  Theorem 6.1 is
conditional on every empirical normalizer lying within relative error
`xi=O(1/T)` and requires `epsilon=O(1/T)`, `H=O(log(1/delta_TV))`, and
`M=O(L*T^2*log(1/delta))` up to the proof's union-bound logarithms.
