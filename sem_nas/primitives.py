"""SEM-NAS primitives.

Four primitives compose the SEM-NAS loop:

* Rank-based RTS insertion (Algorithm 2): standard sampled-window rank
  replacement of Harik (1995) with no niche term. Used both for child
  insertion in the reproduction phase and for local-refinement write-back
  (LWB) of locally improved architectures.
* Rank-plus-distance LS-target selection (Algorithm 3): the first target
  is the current best proxy-ranked member; subsequent targets greedily
  maximize ``proxy_rank + min_Hamming_distance_to_selected / N_EDGES``.
* Edge-wise entropy-guided mutation: shifts mutation pressure toward
  low-entropy (converged) edges and under-represented operations.
* Hamming distance: the metric used by the LS-target non-redundancy term
  and by the diagnostic archive-coverage reporting.
"""
from __future__ import annotations

import numpy as np

from .encoding import N_EDGES, N_OPS
from .operators import MUTATION_PROB


# ---------------------------------------------------------------------------
# Hamming distance over the length-6 op vector
# ---------------------------------------------------------------------------

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.asarray(a) != np.asarray(b)).sum())


# ---------------------------------------------------------------------------
# Algorithm 2: rank-based Restricted Tournament Replacement (no niche term)
# ---------------------------------------------------------------------------

def rts_insert(pop: np.ndarray, fit: np.ndarray, child: np.ndarray,
               child_fit: float, W: int) -> bool:
    """Sampled-window rank replacement.

    Sample ``W`` indices from the population without replacement, identify
    the lowest-fitness member of the window, and replace it with ``child``
    if and only if ``child_fit > fit_of_window_min``. The population size
    is preserved.

    Returns True if the child was accepted.
    """
    n = len(pop)
    w_eff = min(int(W), n)
    if w_eff >= n:
        sample = np.arange(n)
    else:
        sample = np.random.choice(n, size=w_eff, replace=False)

    sample_fits = np.asarray(fit, dtype=float)[sample]
    chosen = int(sample[int(np.argmin(sample_fits))])
    if child_fit > fit[chosen]:
        pop[chosen] = child
        fit[chosen] = child_fit
        return True
    return False


# ---------------------------------------------------------------------------
# Algorithm 3: rank-plus-distance LS-target selection
# ---------------------------------------------------------------------------

def _rank_normalized(values: np.ndarray) -> np.ndarray:
    """Map ``values`` to ranks in ``[0, 1]`` (worst -> 0, best -> 1)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return np.ones(n, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


def rank_plus_distance_targets(pop: np.ndarray, fit: np.ndarray,
                               K: int) -> list[tuple[np.ndarray, float]]:
    """Greedy LS-target selection by proxy rank plus Hamming non-redundancy.

    The first target is the highest-fitness population member. Subsequent
    targets greedily maximize

        s(a) = rank_in_[0,1](psi(a)) + min_{b in T} d_H(a, b) / N_EDGES,

    where T is the set of targets already selected. This scoring keeps the
    target rule single-proxy while reducing redundant local search around
    near-identical population members.

    Returns a list of ``(chrom_copy, fitness)`` pairs of length
    ``min(K, |pop|)``.
    """
    if K <= 0:
        return []
    n = len(pop)
    k_eff = min(int(K), n)
    proxy_rank = _rank_normalized(np.asarray(fit, dtype=float))

    remaining = set(range(n))
    chosen: list[int] = []
    while len(chosen) < k_eff and remaining:
        if not chosen:
            best = max(remaining, key=lambda i: (float(proxy_rank[i]),
                                                 float(fit[i]), -i))
        else:
            def score(i: int) -> tuple[float, float, int]:
                min_d = min(hamming(pop[i], pop[j]) for j in chosen)
                primary = float(proxy_rank[i]) + min_d / float(N_EDGES)
                return (primary, float(fit[i]), -i)
            best = max(remaining, key=score)
        chosen.append(int(best))
        remaining.remove(best)
    return [(pop[i].copy(), float(fit[i])) for i in chosen]


# ---------------------------------------------------------------------------
# Edge-wise entropy-guided mutation
# ---------------------------------------------------------------------------

def _edge_op_counts(population: np.ndarray) -> np.ndarray:
    """Return a (N_EDGES, N_OPS) integer count matrix over the population."""
    counts = np.zeros((N_EDGES, N_OPS), dtype=float)
    arr = np.asarray(population)
    for edge in range(N_EDGES):
        counts[edge] = np.bincount(arr[:, edge], minlength=N_OPS).astype(float)
    return counts


def _edge_entropy_deficit(counts: np.ndarray) -> np.ndarray:
    """Per-edge ``D_e = 1 - H_e`` with ``H_e`` normalized Shannon entropy.

    Uses the convention ``0 log 0 = 0``. ``H_e`` is in ``[0, 1]``.
    Edges fully concentrated on one operation have ``D_e = 1``; uniform
    edges have ``D_e = 0``.
    """
    deficits = np.zeros(N_EDGES, dtype=float)
    for edge in range(N_EDGES):
        total = max(float(counts[edge].sum()), 1.0)
        probs = counts[edge] / total
        nz = probs > 0
        H = -float(np.sum(probs[nz] * np.log(probs[nz]))) / np.log(N_OPS)
        deficits[edge] = max(0.0, 1.0 - H)
    return deficits


def entropy_guided_mutation(individual: np.ndarray, population: np.ndarray,
                            mutation_prob: float = MUTATION_PROB) -> np.ndarray:
    """Mutate low-entropy edges toward under-represented operations.

    The expected number of mutated edges remains ``L * mutation_prob`` (i.e.,
    1 when ``mutation_prob = 1/L``), but probability mass is shifted toward
    edges with low Shannon entropy in the current population. If the
    Bernoulli edge mask is empty, one edge is forced to mutate so the child
    differs from the parent.
    """
    child = individual.copy()
    counts = _edge_op_counts(population)
    deficits = _edge_entropy_deficit(counts)

    # Floor to keep every edge reachable, then normalize to a probability vector.
    edge_weights = deficits + 0.05
    edge_probs = edge_weights / edge_weights.sum()

    expected_flips = N_EDGES * float(mutation_prob)  # 1 when mutation_prob = 1/L
    per_edge_prob = np.minimum(0.75, edge_probs * expected_flips)

    mask = np.random.random(N_EDGES) < per_edge_prob
    if not np.any(mask):
        forced = int(np.random.choice(np.arange(N_EDGES), p=edge_probs))
        mask[forced] = True

    for edge in np.flatnonzero(mask):
        current = int(child[edge])
        ops = [op for op in range(N_OPS) if op != current]
        # Prefer operations that are rare on this edge (rare-op weighting).
        weights = np.array([1.0 / (counts[edge, op] + 1.0) for op in ops])
        weights = weights / weights.sum()
        child[edge] = int(np.random.choice(ops, p=weights))
    return child


# ---------------------------------------------------------------------------
# Diagnostic: population-mean per-edge Shannon entropy in [0, 1]
# ---------------------------------------------------------------------------

def population_entropy(population: np.ndarray) -> float:
    """Mean per-edge normalized Shannon entropy of the population.

    Used as an archive-diversity diagnostic; not invoked by the search loop.
    """
    counts = _edge_op_counts(population)
    H_sum = 0.0
    n = max(len(population), 1)
    for edge in range(N_EDGES):
        probs = counts[edge] / n
        nz = probs > 0
        H_sum += -float(np.sum(probs[nz] * np.log(probs[nz])))
    return (H_sum / N_EDGES) / np.log(N_OPS)
