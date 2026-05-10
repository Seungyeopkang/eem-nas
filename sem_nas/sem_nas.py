"""Sample-Efficient Memetic NAS (SEM-NAS), the proposed method.

Algorithm 1 of the paper. SEM-NAS is a single-population memetic procedure
that integrates four primitives under a strict FFC budget:

1. Standard rank-based RTS replacement (Harik 1995, no niche term) for child
   insertion.
2. Rank-plus-distance LS-target selection: every K_gen generations, pick K_LS
   targets that maximize proxy rank plus normalized Hamming distance to the
   already-selected targets.
3. Bounded 1-flip first-improvement local search on each target, FFC-bounded
   by b_LS.
4. Local-refinement write-back (LWB): the locally improved architecture is
   reinserted through the same RTS rule used for ordinary offspring.

Edge-wise entropy-guided mutation acts as a proposal-side diversity pressure
on the reproduction phase. The search remains single-proxy: no test accuracy,
proxy fusion, learned predictor, or supernet is used inside the loop.
"""
from __future__ import annotations

import numpy as np

from .encoding import N_EDGES, random_individual
from .evaluator import FitnessEvaluator
from .local_search import LSView, first_improvement_1flip
from .operators import (
    MUTATION_PROB,
    tournament_selection,
    uniform_crossover,
)
from .primitives import (
    entropy_guided_mutation,
    rank_plus_distance_targets,
    rts_insert,
)


def _run_one_ls_target(pop, fit, target_chrom, target_fit, evaluator, *,
                       gen: int, rank: int, b_LS: int, W: int) -> None:
    """Run 1-flip first-improvement LS on one target and write back via RTS.

    The LS RNG is seeded as ``gen * 10007 + rank * 131 + 7`` so reruns with
    the same global seed are bit-exact (this matches the seed convention
    reported in the paper).
    """
    if evaluator.budget_exhausted():
        return
    remaining = max(0, evaluator.max_evals - evaluator.ffc)
    effective_budget = min(int(b_LS), remaining)
    if effective_budget <= 0:
        return

    ls_view = LSView(evaluator, seed_chrom=target_chrom, seed_fit=target_fit)
    ls_rng = np.random.default_rng(gen * 10007 + rank * 131 + 7)
    result = first_improvement_1flip(target_chrom, ls_view, effective_budget, ls_rng)

    # Local-refinement write-back through the same RTS rule used for offspring.
    rts_insert(pop, fit, result["final_chromosome"],
               float(result["final_fitness"]), W)


def run(evaluator: FitnessEvaluator,
        *,
        pop_size: int = 10,
        K_gen: int = 4,
        K_LS: int = 3,
        W: int = 3,
        b_LS: int = 25,
        mutation_prob: float = MUTATION_PROB) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SEM-NAS until the FFC budget is exhausted.

    Args:
        evaluator: ``FitnessEvaluator`` carrying the proxy/test arrays and
            the FFC budget.
        pop_size: population size ``|P|``.
        K_gen: LS trigger period (every K_gen generations).
        K_LS: number of LS targets per trigger.
        W: RTS window size.
        b_LS: per-LS-call FFC cap (number of newly-evaluated neighbors).
        mutation_prob: per-edge mutation probability used by the entropy-
            guided mutation rule. With the paper-final value ``1/N_EDGES``,
            the expected number of mutated edges per child equals 1.

    Returns:
        ``(best_chrom, population, fitness)`` at the FFC budget.
    """
    # Cap initialization so the random-init phase never overspends the budget.
    pop_cap = min(int(pop_size), max(0, evaluator.max_evals - evaluator.ffc))
    if pop_cap < 1:
        empty = np.zeros(N_EDGES, dtype=int)
        return empty, np.zeros((0, N_EDGES), dtype=int), np.zeros(0, dtype=float)

    pop = np.array([random_individual() for _ in range(pop_cap)])
    fit = evaluator.evaluate_batch(pop)

    gen = 0
    while not evaluator.budget_exhausted():
        gen += 1

        # Reproduction phase: |P| children per generation.
        for _ in range(len(pop)):
            if evaluator.budget_exhausted():
                break
            p1 = tournament_selection(pop, fit)
            p2 = tournament_selection(pop, fit)
            child, _ = uniform_crossover(p1, p2)
            child = entropy_guided_mutation(child, pop, mutation_prob)
            child_fit = evaluator.evaluate(child)
            rts_insert(pop, fit, child, float(child_fit), W)

        # Periodic LS phase with rank-plus-distance target selection and LWB.
        if gen % int(K_gen) != 0 or evaluator.budget_exhausted():
            continue
        targets = rank_plus_distance_targets(pop, fit, K=int(K_LS))
        for rank, (target_chrom, target_fit) in enumerate(targets):
            if evaluator.budget_exhausted():
                break
            _run_one_ls_target(
                pop, fit, target_chrom, target_fit, evaluator,
                gen=gen, rank=rank, b_LS=int(b_LS), W=int(W),
            )

    best_idx = int(np.argmax(fit))
    return pop[best_idx].copy(), pop, fit
