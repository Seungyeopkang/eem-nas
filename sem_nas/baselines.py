"""Budgeted fixed-proxy search baselines.

Four baselines compared against SEM-NAS in the paper:

* :func:`random_search` — i.i.d. uniform sampling.
* :func:`aging_evolution` — Aging (Regularized) Evolution of Real et al. (2019).
* :func:`simple_mutation` — steady-state mutation-only search.
* :func:`generic_ga` — uniform-crossover GA with elitism, tournament selection,
  and block mutation.

Two forced-edit controls are also exposed for the §4 fairness diagnostic:
:func:`simple_mutation_forced` and :func:`generic_ga_forced`. They differ from
the originals only in using :func:`forced_block_mutation`, which guarantees
at least one edit per mutated offspring.

All baselines share population size with SEM-NAS (``pop_size = 10``) and use
the same operator hyperparameters. They optimize a single fixed proxy under
identical FFC accounting, so the comparison isolates the effect of the
SEM-NAS search loop.
"""
from __future__ import annotations

import numpy as np

from .encoding import random_individual
from .evaluator import FitnessEvaluator
from .operators import (
    block_mutation,
    forced_block_mutation,
    tournament_selection,
    uniform_crossover,
)

DEFAULT_POP_SIZE = 10
DEFAULT_TOURNAMENT_SAMPLE = 25  # Aging Evolution sample size; clipped to |P|.
ELITISM_SIZE = 1


def _best(pop: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return pop[int(np.argmax(fit))].copy()


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------

def random_search(evaluator: FitnessEvaluator,
                  pop_size: int = DEFAULT_POP_SIZE) -> np.ndarray:
    """Uniform random sampling until the FFC budget is exhausted."""
    best_score = -np.inf
    best_ind = None
    while not evaluator.budget_exhausted():
        batch_size = min(pop_size, evaluator.max_evals - evaluator.ffc)
        batch = np.array([random_individual() for _ in range(batch_size)])
        batch_fit = evaluator.evaluate_batch(batch)
        i = int(np.argmax(batch_fit))
        if batch_fit[i] > best_score:
            best_score = float(batch_fit[i])
            best_ind = batch[i].copy()
    return best_ind if best_ind is not None else random_individual()


# ---------------------------------------------------------------------------
# Aging (Regularized) Evolution — Real et al. 2019
# ---------------------------------------------------------------------------

def aging_evolution(evaluator: FitnessEvaluator,
                    pop_size: int = DEFAULT_POP_SIZE,
                    sample_size: int = DEFAULT_TOURNAMENT_SAMPLE) -> np.ndarray:
    """Aging Evolution (Real et al., 2019) under FFC accounting."""
    population = []
    for _ in range(pop_size):
        if evaluator.budget_exhausted():
            break
        ind = random_individual()
        fit = evaluator.evaluate(ind)
        population.append({"genes": ind, "fitness": fit, "age": 0})

    best_score = max((p["fitness"] for p in population), default=-np.inf)
    best_ind = max(population, key=lambda p: p["fitness"])["genes"].copy() \
        if population else random_individual()

    while not evaluator.budget_exhausted():
        s = min(sample_size, len(population))
        idx = np.random.choice(len(population), size=s, replace=False)
        parent = max((population[i] for i in idx), key=lambda p: p["fitness"])
        child = block_mutation(parent["genes"])
        child_fit = evaluator.evaluate(child)
        for p in population:
            p["age"] += 1
        population.append({"genes": child, "fitness": child_fit, "age": 0})
        oldest = max(range(len(population)), key=lambda i: population[i]["age"])
        population.pop(oldest)
        if child_fit > best_score:
            best_score = float(child_fit)
            best_ind = child.copy()
    return best_ind


# ---------------------------------------------------------------------------
# Simple Mutation (steady-state)
# ---------------------------------------------------------------------------

def _simple_mutation_inner(evaluator: FitnessEvaluator, mutate_fn,
                           pop_size: int) -> np.ndarray:
    population = np.array([random_individual() for _ in range(pop_size)])
    fitness = evaluator.evaluate_batch(population)
    best_score = float(np.max(fitness))
    best_ind = population[int(np.argmax(fitness))].copy()
    while not evaluator.budget_exhausted():
        parent = population[np.random.randint(len(population))]
        child = mutate_fn(parent)
        child_fit = float(evaluator.evaluate(child))
        worst = int(np.argmin(fitness))
        if child_fit > fitness[worst]:
            population[worst] = child
            fitness[worst] = child_fit
        if child_fit > best_score:
            best_score = child_fit
            best_ind = child.copy()
    return best_ind


def simple_mutation(evaluator: FitnessEvaluator,
                    pop_size: int = DEFAULT_POP_SIZE) -> np.ndarray:
    """Steady-state mutation-only search."""
    return _simple_mutation_inner(evaluator, block_mutation, pop_size)


def simple_mutation_forced(evaluator: FitnessEvaluator,
                           pop_size: int = DEFAULT_POP_SIZE) -> np.ndarray:
    """Forced-edit Simple Mutation control (paper §4 fairness diagnostic)."""
    return _simple_mutation_inner(evaluator, forced_block_mutation, pop_size)


# ---------------------------------------------------------------------------
# Generic GA — uniform crossover, tournament selection, block mutation, elitism
# ---------------------------------------------------------------------------

def _generic_ga_inner(evaluator: FitnessEvaluator, mutate_fn,
                      pop_size: int) -> np.ndarray:
    population = np.array([random_individual() for _ in range(pop_size)])
    fitness = evaluator.evaluate_batch(population)
    while not evaluator.budget_exhausted():
        elite_idx = np.argsort(fitness)[-ELITISM_SIZE:]
        new_pop = [population[i].copy() for i in elite_idx]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            c1, c2 = uniform_crossover(p1, p2)
            c1 = mutate_fn(c1)
            c2 = mutate_fn(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = np.array(new_pop[:pop_size])
        fitness = evaluator.evaluate_batch(population)
    return _best(population, fitness)


def generic_ga(evaluator: FitnessEvaluator,
               pop_size: int = DEFAULT_POP_SIZE) -> np.ndarray:
    """Generic GA with uniform crossover + tournament + block mutation + elitism."""
    return _generic_ga_inner(evaluator, block_mutation, pop_size)


def generic_ga_forced(evaluator: FitnessEvaluator,
                      pop_size: int = DEFAULT_POP_SIZE) -> np.ndarray:
    """Forced-edit Generic GA control (paper §4 fairness diagnostic)."""
    return _generic_ga_inner(evaluator, forced_block_mutation, pop_size)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    "random": random_search,
    "aging_evolution": aging_evolution,
    "simple_mutation": simple_mutation,
    "generic_ga": generic_ga,
    # Forced-edit controls (fairness diagnostic).
    "simple_mutation_forced": simple_mutation_forced,
    "generic_ga_forced": generic_ga_forced,
}
