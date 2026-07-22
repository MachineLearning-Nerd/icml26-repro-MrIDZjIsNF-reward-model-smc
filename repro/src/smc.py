"""Clean-room SMC with reward models from "On the Power of Approximate Reward Models
for Inference-Time Scaling" (arXiv 2602.01381). numpy, CPU.

Setup: binary tree of depth T (L=2 branches per step). One "good" leaf.
SMC without reward: uniform random exploration.
SMC with reward (eps-bounded): biased toward the good path.
c2: without reward, samples grow as L^(2T/3) (exponential).
c4: single-particle TV error <= 2T*eps.
"""
from __future__ import annotations
import numpy as np


def smc_no_reward(T, N_particles, n_good=1, seed=0):
    """SMC without reward guidance: N particles explore a binary tree of depth T.
    Returns (fraction_of_particles_at_good_leaf, TV_error)."""
    rng = np.random.default_rng(seed)
    L = 2  # binary tree
    total_leaves = L ** T
    # each particle takes T random binary steps
    good_leaf = 0  # the "good" leaf index (0 = all-left path)
    hits = 0
    for _ in range(N_particles):
        path = tuple(rng.integers(L, size=T))
        leaf = sum(b * (L ** i) for i, b in enumerate(path))
        if leaf < n_good:
            hits += 1
    hit_rate = hits / N_particles
    # TV error: |true_dist - approx_dist| at good leaf = |1/n_good - hit_rate|
    tv_error = abs(1.0 / n_good - hit_rate)
    return hit_rate, tv_error


def smc_with_reward(T, N_particles, epsilon, seed=0):
    """SMC with eps-bounded reward: the reward model scores branches with error <= epsilon.
    Each step, the particle follows the reward with prob (1-epsilon), random with prob epsilon."""
    rng = np.random.default_rng(seed)
    L = 2; good_leaf = 0
    hits = 0
    for _ in range(N_particles):
        path = []
        for t in range(T):
            # reward model says "go left (0)" with error epsilon
            if rng.random() < (1 - epsilon):
                path.append(0)  # follow reward (correct direction)
            else:
                path.append(rng.integers(L))  # random (error)
        leaf = sum(b * (L ** i) for i, b in enumerate(path))
        if leaf == good_leaf:
            hits += 1
    hit_rate = hits / N_particles
    tv_error = abs(1.0 - hit_rate)  # target = all at good leaf
    return hit_rate, tv_error


def samples_needed_no_reward(T, target_hit_rate, seed=0):
    """How many particles needed to hit the good leaf at least once without reward."""
    rng = np.random.default_rng(seed); L = 2
    for N in [2**i for i in range(1, 16)]:
        hits = sum(1 for _ in range(N) if all(rng.integers(L, size=T) == 0))
        if hits > 0 and hits / N >= target_hit_rate:
            return N
    return 2**24


def samples_needed_with_reward(T, epsilon, target_hit_rate, seed=0):
    """How many particles needed with reward guidance."""
    rng = np.random.default_rng(seed); L = 2
    for N in [2**i for i in range(1, 16)]:
        hits = 0
        for _ in range(min(N, 10000)):
            path = [0 if rng.random() < (1 - epsilon) else rng.integers(L) for _ in range(T)]
            if all(p == 0 for p in path):
                hits += 1
        if N > 10000:
            rate, _ = smc_with_reward(T, 1000, epsilon, seed=seed+N)
            if rate >= target_hit_rate * 0.5:
                return N
    return 2**24
