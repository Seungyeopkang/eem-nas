"""Generic GA operators.

These operators are shared between EEM-NAS and the evolutionary baselines.
They use NumPy global RNG via the seed set in the run driver, which is the
convention used by the paper experiments for bit-exact reruns.
"""
from __future__ import annotations

import numpy as np

from .encoding import N_EDGES, N_OPS

# Default hyperparameters (paper-final).
TOURNAMENT_SIZE = 5
CROSSOVER_PROB = 0.9
MUTATION_PROB = 1.0 / N_EDGES  # per-edge mutation rate (~0.167)


def tournament_selection(population: np.ndarray, fitness: np.ndarray,
                         k: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Sample k individuals without replacement and return the fittest one."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best = indices[int(np.argmax(np.asarray(fitness)[indices]))]
    return population[best]


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray,
                      prob: float = CROSSOVER_PROB):
    """Per-edge uniform crossover with crossover probability ``prob``."""
    if np.random.random() > prob:
        return parent1.copy(), parent2.copy()
    mask = np.random.randint(0, 2, size=N_EDGES)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def block_mutation(individual: np.ndarray,
                   prob: float = MUTATION_PROB) -> np.ndarray:
    """Per-edge mutation: each edge independently mutates with probability ``prob``.

    A mutated edge picks a new operation uniformly from the remaining N_OPS-1
    operations. The expected number of mutated edges is ``N_EDGES * prob``.
    With ``prob = 1/L``, this expected count equals 1.
    """
    child = individual.copy()
    for i in range(N_EDGES):
        if np.random.random() < prob:
            current = int(child[i])
            candidates = [v for v in range(N_OPS) if v != current]
            child[i] = int(np.random.choice(candidates))
    return child


def forced_block_mutation(individual: np.ndarray,
                          prob: float = MUTATION_PROB) -> np.ndarray:
    """Block mutation with an at-least-one-edit fallback.

    When the per-edge Bernoulli mask happens to be all zero, one edge is
    forced to mutate. Used by EEM-NAS for the entropy-mutation primitive
    and by the forced-edit fairness-diagnostic baselines.
    """
    child = block_mutation(individual, prob=prob)
    if np.array_equal(child, individual):
        edge = int(np.random.choice(np.arange(N_EDGES)))
        current = int(child[edge])
        candidates = [v for v in range(N_OPS) if v != current]
        child[edge] = int(np.random.choice(candidates))
    return child
