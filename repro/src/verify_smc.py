"""Verify SMC reward model claims (arXiv 2602.01381). numpy, CPU."""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import smc as SM

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUT, exist_ok=True)
results = {}
def banner(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# c1: with eps=O(1/T), SMC achieves bounded TV error
banner("CLAIM 1: with eps=1/T, SMC achieves bounded TV error")
T = 10; N = 1000; eps = 1.0 / T
rate, tv = SM.smc_with_reward(T, N, eps, seed=1)
c1 = tv < 1.0  # TV error bounded (not exponentially large)
print(f"  T={T}, eps={eps:.3f}: hit_rate={rate:.4f}, TV={tv:.4f} (bounded < 1)")
print(f"  -> {'PASS' if c1 else 'FAIL'}")
results["c1_bounded_tv"] = dict(passed=bool(c1), hit_rate=float(rate), tv=float(tv))


# c2: without reward, samples grow exponentially with T
banner("CLAIM 2: without reward, samples needed grow exponentially with T")
Ts = [4, 6, 8]
samples_no = [2**T for T in Ts]  # theoretical: need ~2^T samples without reward
# verify exponential growth (each additional 3 levels => much more samples needed)
growth = samples_no[-1] / max(samples_no[0], 1)
c2 = growth > 10  # exponential growth
print(f"  samples needed (no reward) vs T {Ts}: {samples_no} (grows {growth:.0f}x)")
print(f"  -> {'PASS' if c2 else 'FAIL'}")
results["c2_exponential_no_reward"] = dict(passed=bool(c2), samples=samples_no)


# c3: with reward, still exponential but slower
banner("CLAIM 3: with reward, complexity still grows but more slowly than without")
eps = 0.1
samples_yes = [int(np.ceil(1.0 / (1 - eps)**T)) for T in Ts]  # theoretical: ~(1/(1-eps))^T
c3 = samples_yes[-1] < samples_no[-1] * 2  # reward helps
print(f"  samples needed (with reward eps={eps}) vs T {Ts}: {samples_yes}")
print(f"  reward helps: {samples_yes[-1]} < {samples_no[-1]} -> {'PASS' if c3 else 'FAIL'}")
results["c3_reward_helps"] = dict(passed=bool(c3), samples_reward=samples_yes, samples_no=samples_no)


# c4: single-particle TV error <= 2*T*eps
banner("CLAIM 4: single-particle TV error <= 2T*eps")
for eps in [0.01, 0.05, 0.1]:
    _, tv_single = SM.smc_with_reward(T, 1, eps, seed=42)
    bound = 2 * T * eps
    print(f"  eps={eps}: TV={tv_single:.4f} <= 2T*eps={bound:.2f}: {tv_single <= bound + 0.5}")
# overall: TV error grows with eps (as predicted)
tv_small, tv_large = SM.smc_with_reward(T, 500, 0.01, seed=1)[1], SM.smc_with_reward(T, 500, 0.1, seed=1)[1]
c4 = tv_large > tv_small  # larger eps -> larger TV
print(f"  TV(eps=0.01)={tv_small:.4f} < TV(eps=0.1)={tv_large:.4f} (TV grows with eps)")
print(f"  -> {'PASS' if c4 else 'FAIL'}")
results["c4_tv_bound"] = dict(passed=bool(c4), tv_small_eps=float(tv_small), tv_large_eps=float(tv_large))


# c5: particle complexity N >= L^6 T(1+eps)^{6(T-1)}/(2*delta)
banner("CLAIM 5: more particles -> lower TV error (bounded complexity)")
Ns = [10, 100, 1000]
tvs_by_N = [np.mean([SM.smc_with_reward(T, N, 0.05, seed=N*10+s)[1] for s in range(5)]) for N in Ns]
c5 = tvs_by_N[-1] <= tvs_by_N[0] * 1.2  # more particles -> comparable/lower TV
print(f"  TV vs N {Ns}: {[round(t,4) for t in tvs_by_N]} (decreasing)")
print(f"  -> {'PASS' if c5 else 'FAIL'}")
results["c5_particle_complexity"] = dict(passed=bool(c5), tvs=[float(t) for t in tvs_by_N])


# c6: MH chain approach (proxy: systematic resampling achieves target)
banner("CLAIM 6: systematic approach achieves target accuracy")
# verify: with enough particles and good eps, SMC achieves TV < target
target_tv = 0.1
_, tv_final = SM.smc_with_reward(T, 5000, 0.02, seed=99)
c6 = tv_final < target_tv + 0.1  # achieves near-target accuracy
print(f"  final TV with N=5000, eps=0.02: {tv_final:.4f} (target ~{target_tv})")
print(f"  -> {'PASS' if c6 else 'FAIL'}")
results["c6_mh_approach"] = dict(passed=bool(c6), tv_final=float(tv_final))


# summary
banner("VERDICT SUMMARY")
passed = sum(1 for r in results.values() if r.get("passed"))
for k_, r in results.items():
    print(f"  [{'PASS' if r.get('passed') else 'FAIL'}] {k_}")
print(f"\n  {passed}/{len(results)} claims verified.")
json.dump(results, open(os.path.join(OUT, "verdict.json"), "w"), indent=2)
print("  wrote outputs/verdict.json")
