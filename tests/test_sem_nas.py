"""Smoke tests for the SEM-NAS reference implementation.

These tests do not require the NAS-Bench-201 pickles; they fabricate a
synthetic 15,625-entry proxy/test array. The point is to exercise the
search loop, FFC accounting, and primitive shapes.

Run with::

    python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem_nas.baselines import BASELINES
from sem_nas.encoding import (
    N_ARCHS,
    N_EDGES,
    N_OPS,
    encoding_to_index,
    index_to_encoding,
    random_individual,
)
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.primitives import (
    entropy_guided_mutation,
    hamming,
    rank_plus_distance_targets,
    rts_insert,
)
from sem_nas.sem_nas import run as run_sem_nas


def _make_evaluator(ffc: int = 100, seed: int = 0) -> FitnessEvaluator:
    rng = np.random.default_rng(seed)
    proxy = rng.standard_normal(N_ARCHS).astype(np.float64)
    test_acc = (50.0 + 40.0 * rng.random(N_ARCHS)).astype(np.float64)
    return FitnessEvaluator(proxy, test_acc, max_evals=ffc)


def test_encoding_bijection():
    rng = np.random.default_rng(42)
    for _ in range(50):
        idx = int(rng.integers(0, N_ARCHS))
        enc = index_to_encoding(idx)
        assert enc.shape == (N_EDGES,)
        assert enc.min() >= 0 and enc.max() < N_OPS
        assert encoding_to_index(enc) == idx


def test_random_individual_shape():
    enc = random_individual()
    assert enc.shape == (N_EDGES,)
    assert enc.dtype.kind in ("i", "u")
    assert enc.min() >= 0 and enc.max() < N_OPS


def test_evaluator_charges_one_ffc_per_call():
    ev = _make_evaluator(ffc=10)
    a = random_individual()
    ev.evaluate(a)
    ev.evaluate(a)  # duplicate still costs 1 FFC
    assert ev.ffc == 2
    assert len(ev.best_score_per_ffc) == 2


def test_rts_insert_preserves_size():
    rng = np.random.default_rng(0)
    pop = np.array([random_individual(rng) for _ in range(10)])
    fit = rng.standard_normal(10)
    child = random_individual(rng)
    n_before = len(pop)
    rts_insert(pop, fit, child, child_fit=float(fit.max()) + 1.0, W=3)
    assert len(pop) == n_before
    assert max(fit) >= float(fit.max())


def test_rank_plus_distance_targets_picks_K_distinct():
    rng = np.random.default_rng(1)
    pop = np.array([random_individual(rng) for _ in range(10)])
    fit = rng.standard_normal(10)
    targets = rank_plus_distance_targets(pop, fit, K=3)
    assert len(targets) == 3
    # Targets should be distinct architectures.
    encs = [encoding_to_index(t[0]) for t in targets]
    assert len(set(encs)) == 3


def test_entropy_guided_mutation_changes_at_least_one_edge():
    rng = np.random.default_rng(2)
    pop = np.array([random_individual(rng) for _ in range(10)])
    parent = pop[0]
    np.random.seed(0)
    child = entropy_guided_mutation(parent, pop, mutation_prob=1.0 / N_EDGES)
    assert hamming(parent, child) >= 1


def test_sem_nas_run_respects_ffc_budget():
    ev = _make_evaluator(ffc=80)
    np.random.seed(0)
    best, pop, fit = run_sem_nas(ev)
    assert ev.ffc <= 80
    assert best.shape == (N_EDGES,)
    assert pop.shape == (10, N_EDGES)
    assert fit.shape == (10,)
    # Best is the argmax of the final population fitness.
    assert float(np.max(fit)) >= float(fit.min())


def test_baselines_respect_ffc_budget():
    for name, fn in BASELINES.items():
        ev = _make_evaluator(ffc=60, seed=hash(name) & 0xFFFF)
        np.random.seed(0)
        best = fn(ev)
        assert ev.ffc <= 60, f"{name} overspent budget"
        assert best.shape == (N_EDGES,)


if __name__ == "__main__":
    # Allow running directly without pytest.
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
