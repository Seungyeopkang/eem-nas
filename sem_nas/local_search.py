"""1-flip first-improvement local search and per-call FFC view.

The local-search phase of SEM-NAS uses a bounded 1-flip first-improvement
sweep over the Hamming-1 neighborhood of the LS target. The neighborhood
contains exactly ``N_EDGES * (N_OPS - 1) = 24`` architectures on NAS-Bench-201
TSS, and each call is capped at ``b_LS`` newly-evaluated neighbors.

Within a single LS call, the seed architecture is preseeded into a local cache
so re-evaluating it is free. Repeated neighbors inside the same call are also
free. New cache misses still consume one FFC each on the underlying
``FitnessEvaluator``.
"""
from __future__ import annotations

import numpy as np

from .encoding import N_EDGES, N_OPS, encoding_to_index


class LSView:
    """Per-LS-call FFC accounting wrapper around the main evaluator.

    Re-evaluations of an architecture that was already seen *inside this LS
    call* cost zero FFC. Cache misses delegate to ``main_eval.evaluate``,
    which charges one FFC against the global budget.
    """

    def __init__(self, main_eval, seed_chrom: np.ndarray | None = None,
                 seed_fit: float | None = None):
        self.main_eval = main_eval
        self.ffc = 0  # newly-evaluated neighbors inside this LS call
        self._cache: dict[int, float] = {}
        if seed_chrom is not None and seed_fit is not None:
            self._cache[encoding_to_index(seed_chrom)] = float(seed_fit)

    def evaluate(self, chrom: np.ndarray) -> float:
        idx = encoding_to_index(chrom)
        if idx in self._cache:
            return self._cache[idx]
        score = self.main_eval.evaluate(chrom)
        self._cache[idx] = score
        self.ffc += 1
        return score


def _one_flip_neighbors(chrom: np.ndarray, rng: np.random.Generator):
    """Yield the 24 Hamming-1 neighbors of ``chrom`` in a random order.

    The deterministic seed of ``rng`` (per-LS-call) gives the bit-exact
    permutation reported in the paper.
    """
    neighbors = []
    for edge in range(N_EDGES):
        current = int(chrom[edge])
        for op in range(N_OPS):
            if op == current:
                continue
            nb = chrom.copy()
            nb[edge] = op
            neighbors.append((edge, op, nb))
    perm = rng.permutation(len(neighbors))
    return [neighbors[i] for i in perm]


def first_improvement_1flip(chromosome: np.ndarray, evaluator: LSView,
                            ffc_budget: int, rng: np.random.Generator) -> dict:
    """First-improvement 1-flip LS, FFC-bounded by ``ffc_budget``.

    Sweeps the 24 Hamming-1 neighbors in a random order, accepts the first
    strictly improving neighbor, and restarts the sweep from the new
    incumbent. Stops at a local optimum or when the FFC budget is reached.

    Returns:
        dict with the locally improved (or unchanged) chromosome, its
        fitness, the number of newly-evaluated neighbors used, and the
        improvement delta.
    """
    current = chromosome.copy()
    current_fit = evaluator.evaluate(current)
    starting_fit = current_fit
    best_chrom, best_fit = current.copy(), current_fit
    ffc_start = evaluator.ffc
    steps = 0

    def can_afford_new_eval() -> bool:
        return (evaluator.ffc - ffc_start) < int(ffc_budget)

    while can_afford_new_eval():
        neighbors = _one_flip_neighbors(current, rng)
        improved_in_sweep = False
        for _, _, nb in neighbors:
            if not can_afford_new_eval() and encoding_to_index(nb) not in evaluator._cache:
                break
            score = evaluator.evaluate(nb)
            if score > current_fit:
                current, current_fit = nb, score
                if score > best_fit:
                    best_chrom, best_fit = nb.copy(), score
                steps += 1
                improved_in_sweep = True
                break
        if not improved_in_sweep:
            break  # local optimum or budget exhausted mid-sweep

    improved = best_fit > starting_fit
    return {
        "final_chromosome": best_chrom,
        "final_fitness": float(best_fit),
        "starting_fitness": float(starting_fit),
        "ffc_used": int(evaluator.ffc - ffc_start),
        "steps_taken": int(steps),
        "improved": bool(improved),
        "improvement_delta": float(best_fit - starting_fit) if improved else 0.0,
    }
