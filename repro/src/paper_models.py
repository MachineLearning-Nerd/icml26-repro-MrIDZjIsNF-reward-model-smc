"""Exact finite-state models used to audit arXiv:2602.01381.

The functions here mirror the paper's definitions.  They deliberately avoid
using the claimed bounds as simulated observations: target laws, guided laws,
assumption constants, and finite-N SMC bias are computed independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, exp, floor, lgamma, log
from typing import Iterable

import numpy as np


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("TV inputs must have the same shape")
    if not np.isclose(p.sum(), 1.0, atol=1e-11):
        raise ValueError("first TV input is not normalized")
    if not np.isclose(q.sum(), 1.0, atol=1e-11):
        raise ValueError("second TV input is not normalized")
    return float(0.5 * np.abs(p - q).sum())


def _binomial_half_pmf(n: int) -> np.ndarray:
    """Stable Binomial(n, 1/2) PMF, built outwards from its mode."""
    if n < 1:
        raise ValueError("n must be positive")
    mode = floor((n + 1) / 2)
    pmf = np.zeros(n + 1, dtype=float)
    pmf[mode] = exp(
        lgamma(n + 1)
        - lgamma(mode + 1)
        - lgamma(n - mode + 1)
        - n * log(2.0)
    )
    for k in range(mode, 0, -1):
        pmf[k - 1] = pmf[k] * k / (n - k + 1)
    for k in range(mode, n):
        pmf[k + 1] = pmf[k] * (n - k) / (k + 1)
    pmf /= pmf.sum()
    return pmf


def exact_resampled_bit_probability(n_particles: int, reward_ratio: float) -> float:
    """Marginal probability of bit 1 after one naive SMC resampling step.

    K ~ Binomial(N, 1/2) propagated particles have bit 1.  Conditional on K,
    multinomial resampling selects bit 1 with probability K*r/(K*r+N-K).
    Taking the expectation gives the expected empirical output law exactly.
    """
    if reward_ratio <= 0:
        raise ValueError("reward_ratio must be positive")
    k = np.arange(n_particles + 1, dtype=float)
    pmf = _binomial_half_pmf(n_particles)
    denominator = k * reward_ratio + n_particles - k
    conditional = np.divide(
        k * reward_ratio,
        denominator,
        out=np.zeros_like(k),
        where=denominator > 0,
    )
    return float(pmf @ conditional)


def product_bernoulli_tv(horizon: int, p: float, q: float) -> float:
    """TV between iid Bernoulli product laws, reduced exactly by Hamming weight."""
    if not (0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError("probabilities must lie in [0, 1]")
    terms = []
    for k in range(horizon + 1):
        multiplicity = comb(horizon, k)
        pk = p**k * (1 - p) ** (horizon - k)
        qk = q**k * (1 - q) ** (horizon - k)
        terms.append(multiplicity * abs(pk - qk))
    return 0.5 * float(sum(terms))


def product_bernoulli_tv_by_paths(horizon: int, p: float, q: float) -> float:
    """Independent checker: enumerate every path instead of grouping by weight."""
    if horizon > 20:
        raise ValueError("path enumeration is intentionally capped at T=20")
    total = 0.0
    for path_id in range(1 << horizon):
        ones = path_id.bit_count()
        pp = p**ones * (1 - p) ** (horizon - ones)
        qq = q**ones * (1 - q) ** (horizon - ones)
        total += abs(pp - qq)
    return 0.5 * total


@dataclass(frozen=True)
class ProductAudit:
    horizon: int
    reward_ratio: float
    ratio_bound_l: float
    bellman_epsilon: float
    target_bit_probability: float


def audit_product_model(horizon: int, reward_ratio: float) -> ProductAudit:
    """Audit V(prefix)=r**(#ones) under a uniform binary reference model."""
    if horizon < 1 or reward_ratio < 1:
        raise ValueError("requires T>=1 and r>=1")
    expected_ratio = (1.0 + reward_ratio) / 2.0
    epsilon = max(expected_ratio, 1.0 / expected_ratio) - 1.0
    return ProductAudit(
        horizon=horizon,
        reward_ratio=reward_ratio,
        ratio_bound_l=reward_ratio,
        bellman_epsilon=epsilon,
        target_bit_probability=reward_ratio / (1.0 + reward_ratio),
    )


def theorem_5_1_particle_bound(
    horizon: int, ratio_bound_l: float, epsilon: float, delta_tv: float
) -> float:
    if horizon < 2 or ratio_bound_l <= 0 or epsilon <= 0:
        raise ValueError("Theorem 5.1 requires T>=2 and L, epsilon > 0")
    if not 0 < delta_tv < 1:
        raise ValueError("delta_tv must be in (0,1)")
    return (
        ratio_bound_l**6
        * horizon
        * (1.0 + epsilon) ** (6 * (horizon - 1))
        / (2.0 * delta_tv)
    )


def exact_product_smc_tv(
    horizon: int, n_particles: int, reward_ratio: float
) -> tuple[float, float, float]:
    audit = audit_product_model(horizon, reward_ratio)
    q = exact_resampled_bit_probability(n_particles, reward_ratio)
    tv = product_bernoulli_tv(horizon, audit.target_bit_probability, q)
    return audit.target_bit_probability, q, tv


@dataclass(frozen=True)
class PrefixTree:
    horizon: int
    levels: tuple[np.ndarray, ...]
    epsilon: float
    ratio_bound_l: float


def build_prefix_tree(horizon: int, epsilon: float) -> PrefixTree:
    """Construct a nontrivial full binary reward tree with exact Bellman audit.

    Terminal rewards alternate smoothly.  Internal values are a child mean times
    alternating factors at the two extrema allowed by Assumption 3.2.
    """
    if horizon < 2 or not 0 < epsilon < 0.25:
        raise ValueError("requires T>=2 and epsilon in (0, .25)")
    terminal = np.array(
        [1.0 + 0.15 * ((path_id.bit_count() % 3) - 1) for path_id in range(1 << horizon)],
        dtype=float,
    )
    levels: list[np.ndarray] = [np.array([]) for _ in range(horizon + 1)]
    levels[horizon] = terminal
    for t in range(horizon - 1, -1, -1):
        children = levels[t + 1].reshape(-1, 2)
        mean = children.mean(axis=1)
        index = np.arange(mean.size)
        factor = np.where(index % 2 == 0, 1.0 + epsilon, 1.0 / (1.0 + epsilon))
        levels[t] = mean * factor

    max_ratio = 1.0
    max_bellman = 1.0
    for t in range(horizon):
        parent = levels[t]
        children = levels[t + 1].reshape(-1, 2)
        mean = children.mean(axis=1)
        max_ratio = max(
            max_ratio,
            float(np.max(children / parent[:, None])),
            float(np.max(parent[:, None] / children)),
        )
        max_bellman = max(
            max_bellman,
            float(np.max(parent / mean)),
            float(np.max(mean / parent)),
        )
    if max_bellman > 1.0 + epsilon + 1e-12:
        raise AssertionError("constructed tree violates its Bellman contract")
    return PrefixTree(
        horizon=horizon,
        levels=tuple(levels),
        epsilon=max_bellman - 1.0,
        ratio_bound_l=max_ratio,
    )


def target_prefix_law(tree: PrefixTree, t: int) -> np.ndarray:
    values = tree.levels[t]
    law = values / values.sum()
    return law


def sp_guided_laws(tree: PrefixTree) -> tuple[np.ndarray, ...]:
    laws: list[np.ndarray] = [np.array([1.0])]
    for t in range(1, tree.horizon + 1):
        parent_law = laws[-1]
        child_values = tree.levels[t].reshape(-1, 2)
        conditional = child_values / child_values.sum(axis=1, keepdims=True)
        laws.append((parent_law[:, None] * conditional).reshape(-1))
    return tuple(laws)


def sp_tv_curve(tree: PrefixTree) -> list[float]:
    laws = sp_guided_laws(tree)
    return [
        total_variation(target_prefix_law(tree, t), laws[t])
        for t in range(tree.horizon + 1)
    ]


def product_sp_tv(horizon: int, reward_ratio: float) -> float:
    """SP-gSMC is exactly the product target for multiplicative V."""
    audit = audit_product_model(horizon, reward_ratio)
    guided_bit_probability = reward_ratio / (1.0 + reward_ratio)
    return product_bernoulli_tv(
        horizon, audit.target_bit_probability, guided_bit_probability
    )


@dataclass(frozen=True)
class HardFamilyCertificate:
    horizon: int
    m: int
    branching: int
    reward_ratio: float
    hidden_prefixes: int
    target_region_mass: float
    tv_forced_hit_probability: float
    query_lower_bound: float
    ratio_bound_l: float
    bellman_epsilon: float


def hard_family_certificate(
    horizon: int, branching: int, reward_ratio: float
) -> HardFamilyCertificate:
    """Executable certificate for Appendix C equations (6)--(14)."""
    if horizon % 3:
        raise ValueError("Appendix C construction writes T=3m")
    if branching < 2 or reward_ratio <= 1:
        raise ValueError("requires integer B>=2 and reward ratio >1")
    m = horizon // 3
    hidden = branching ** (2 * m)
    good_weight = reward_ratio**m
    bad_weight = reward_ratio ** (-m)
    target_mass = good_weight / (good_weight + (hidden - 1) * bad_weight)
    forced = target_mass - 1.0 / 3.0
    return HardFamilyCertificate(
        horizon=horizon,
        m=m,
        branching=branching,
        reward_ratio=reward_ratio,
        hidden_prefixes=hidden,
        target_region_mass=target_mass,
        tv_forced_hit_probability=forced,
        query_lower_bound=forced * hidden,
        ratio_bound_l=max(reward_ratio, 1.0 / reward_ratio),
        bellman_epsilon=reward_ratio - 1.0,
    )


def empirical_sequential_query_hits(
    hidden_prefixes: int, queries: int, trials: int, seed: int
) -> tuple[int, float]:
    """Run a no-guess oracle algorithm that queries prefixes 0,1,...,Q-1."""
    if not 0 <= queries <= hidden_prefixes:
        raise ValueError("queries must lie in [0, hidden_prefixes]")
    rng = np.random.default_rng(seed)
    hidden_u = rng.integers(0, hidden_prefixes, size=trials)
    hits = int(np.count_nonzero(hidden_u < queries))
    return hits, hits / trials


def wilson_interval(successes: int, trials: int, z: float = 3.290526731) -> tuple[float, float]:
    """Two-sided Wilson interval; default z gives approximately 99.9% coverage."""
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (
        z
        * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return float(center - radius), float(center + radius)


def log_log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.log(np.asarray(tuple(xs), dtype=float))
    y = np.log(np.asarray(tuple(ys), dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def log_linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(tuple(xs), dtype=float)
    y = np.log(np.asarray(tuple(ys), dtype=float))
    return float(np.polyfit(x, y, 1)[0])
