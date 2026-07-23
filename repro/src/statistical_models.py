"""Independent statistical checks for the reward-model SMC paper.

This module intentionally does not import the exact finite-state checker.  It
simulates the sufficient statistics of multinomial SMC and randomized no-guess
oracle algorithms, then compares them with independently evaluated reference
laws.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """Stable Binomial(n, p) probability mass function."""
    k = np.arange(n + 1, dtype=float)
    logs = np.array(
        [
            math.lgamma(n + 1)
            - math.lgamma(int(i) + 1)
            - math.lgamma(n - int(i) + 1)
            + i * math.log(p)
            + (n - i) * math.log1p(-p)
            for i in k
        ]
    )
    out = np.exp(logs - np.max(logs))
    return out / out.sum()


def one_step_selected_probability(n_particles: int, reward_ratio: float) -> float:
    """Exact marginal P(selected bit=1) after propose/weight/resample.

    K ~ Binomial(N, 1/2) is the number of one-bits proposed.  Conditional on K,
    multinomial resampling selects a one-bit with probability Kr/(Kr+N-K).
    """
    probs = binomial_pmf(n_particles, 0.5)
    k = np.arange(n_particles + 1, dtype=float)
    denom = k * reward_ratio + n_particles - k
    selected = np.divide(k * reward_ratio, denom, out=np.zeros_like(k), where=denom > 0)
    return float(probs @ selected)


def theorem5_bound(
    horizon: int, reward_bound: float, epsilon: float, delta_tv: float
) -> int:
    """Literal sufficient particle bound printed in Theorem 5.1."""
    value = (
        reward_bound**6
        * horizon
        * (1.0 + epsilon) ** (6 * (horizon - 1))
        / (2.0 * delta_tv)
    )
    return math.ceil(value)


def product_tv(horizon: int, p: float, q: float) -> float:
    """TV between exchangeable Bernoulli product laws, grouped by Hamming weight."""
    return 0.5 * float(np.abs(binomial_pmf(horizon, p) - binomial_pmf(horizon, q)).sum())


def simulate_selected_hamming_counts(
    *,
    horizon: int,
    n_particles: int,
    reward_ratio: float,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Simulate selected paths from the actual multinomial-SMC sufficient statistic.

    Product potentials make steps independent.  Sampling K at every step is
    exactly equivalent, for a randomly selected output particle, to proposing
    N fair bits, weighting them by r**bit, and multinomial resampling.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros(repetitions, dtype=np.int16)
    batch = 10_000
    for start in range(0, repetitions, batch):
        size = min(batch, repetitions - start)
        selected_sum = np.zeros(size, dtype=np.int16)
        for _ in range(horizon):
            ones = rng.binomial(n_particles, 0.5, size=size)
            prob = ones * reward_ratio / (ones * reward_ratio + n_particles - ones)
            selected_sum += rng.binomial(1, prob).astype(np.int16)
        counts[start : start + size] = selected_sum
    return counts


def empirical_hamming_tv(counts: np.ndarray, horizon: int, target_p: float) -> float:
    empirical = np.bincount(counts, minlength=horizon + 1) / len(counts)
    return 0.5 * float(np.abs(empirical - binomial_pmf(horizon, target_p)).sum())


def bootstrap_tv_interval(
    counts: np.ndarray,
    horizon: int,
    target_p: float,
    *,
    seed: int,
    draws: int = 300,
) -> tuple[float, float]:
    """Percentile interval for the empirical grouped-TV diagnostic."""
    rng = np.random.default_rng(seed)
    histogram = np.bincount(counts, minlength=horizon + 1)
    probabilities = histogram / histogram.sum()
    target = binomial_pmf(horizon, target_p)
    sample_size = len(counts)
    tvs = np.empty(draws)
    for i in range(draws):
        boot = rng.multinomial(sample_size, probabilities) / sample_size
        tvs[i] = 0.5 * np.abs(boot - target).sum()
    return float(np.quantile(tvs, 0.025)), float(np.quantile(tvs, 0.975))


def randomized_no_guess_trials(
    *,
    search_space: int,
    query_budget: int,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a randomized-permutation no-guess oracle strategy.

    The rank of the hidden prefix in a uniformly random permutation is itself
    uniform.  Sampling that rank is an exact implementation of the strategy,
    not an evaluation of the lower-bound formula.
    """
    rng = np.random.default_rng(seed)
    hidden = rng.integers(0, search_space, size=repetitions)
    permutation_rank = rng.integers(1, search_space + 1, size=repetitions)
    # Hidden labels are retained to demonstrate that query order is independent.
    hits = permutation_rank <= query_budget
    queries_used = np.minimum(permutation_rank, query_budget)
    return hits, np.column_stack([hidden, permutation_rank, queries_used])


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return center - half, center + half


def log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    return float(np.polyfit(np.asarray(list(xs), float), np.log(np.asarray(list(ys), float)), 1)[0])

