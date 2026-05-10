"""Smoke tests for the SEM-NAS reference implementation.

These tests do not require the NAS-Bench-201 pickles or downloaded data.
The precomputed-backend tests fabricate a synthetic 15,625-entry proxy
array; the online-backend tests use random Gaussian batches and a small
NB-201 network.

Run with::

    python -m pytest tests/ -q

Tests for the online backend require PyTorch; if torch is not installed,
those tests are skipped automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
from sem_nas.proxy import PrecomputedProxyBackend
from sem_nas.sem_nas import run as run_sem_nas


def _make_evaluator(ffc: int = 100, seed: int = 0) -> FitnessEvaluator:
    rng = np.random.default_rng(seed)
    proxy = rng.standard_normal(N_ARCHS).astype(np.float64)
    test_acc = (50.0 + 40.0 * rng.random(N_ARCHS)).astype(np.float64)
    return FitnessEvaluator.from_arrays(proxy, test_acc, max_evals=ffc)


# ---------------------------------------------------------------------------
# Encoding / primitives
# ---------------------------------------------------------------------------


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


def test_rank_plus_distance_targets_picks_K_distinct():
    rng = np.random.default_rng(1)
    pop = np.array([random_individual(rng) for _ in range(10)])
    fit = rng.standard_normal(10)
    targets = rank_plus_distance_targets(pop, fit, K=3)
    assert len(targets) == 3
    encs = [encoding_to_index(t[0]) for t in targets]
    assert len(set(encs)) == 3


def test_entropy_guided_mutation_changes_at_least_one_edge():
    rng = np.random.default_rng(2)
    pop = np.array([random_individual(rng) for _ in range(10)])
    parent = pop[0]
    np.random.seed(0)
    child = entropy_guided_mutation(parent, pop, mutation_prob=1.0 / N_EDGES)
    assert hamming(parent, child) >= 1


# ---------------------------------------------------------------------------
# Search loops with the precomputed backend
# ---------------------------------------------------------------------------


def test_sem_nas_run_respects_ffc_budget():
    ev = _make_evaluator(ffc=80)
    np.random.seed(0)
    best, pop, fit = run_sem_nas(ev)
    assert ev.ffc <= 80
    assert best.shape == (N_EDGES,)
    assert pop.shape == (10, N_EDGES)
    assert fit.shape == (10,)


def test_baselines_respect_ffc_budget():
    for name, fn in BASELINES.items():
        ev = _make_evaluator(ffc=60, seed=hash(name) & 0xFFFF)
        np.random.seed(0)
        best = fn(ev)
        assert ev.ffc <= 60, f"{name} overspent budget"
        assert best.shape == (N_EDGES,)


# ---------------------------------------------------------------------------
# Online backend (PyTorch). Skipped if torch is unavailable.
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_online_backend_proxy_finite():
    """Build NB-201 once and confirm each proxy returns a finite scalar."""
    from sem_nas.proxy import OnlineProxyBackend
    from sem_nas.proxy.proxies import PROXY_NAMES

    enc = np.array([3, 1, 3, 1, 3, 1], dtype=int)  # mixed conv/skip cell
    for proxy_name in PROXY_NAMES:
        backend = OnlineProxyBackend(
            proxy_name=proxy_name,
            dataset="cifar10",
            batch_size=8,
            n_batches=2 if proxy_name == "zico" else 1,
            cells_per_stage=1,
            data_source="random",
            init_seed=0,
            cell_seed=0,
        )
        score = backend.evaluate(enc)
        assert np.isfinite(score), f"{proxy_name} returned non-finite score"


def test_sem_nas_with_online_backend_smoke():
    """Run SEM-NAS for a tiny FFC budget with the online backend."""
    from sem_nas.proxy import OnlineProxyBackend

    backend = OnlineProxyBackend(
        proxy_name="synflow",  # data-free, fastest of the seven
        dataset="cifar10",
        batch_size=8,
        n_batches=1,
        cells_per_stage=1,
        data_source="random",
        init_seed=0,
        cell_seed=0,
    )
    ev = FitnessEvaluator(backend, max_evals=12)
    np.random.seed(0)
    best, pop, fit = run_sem_nas(ev, pop_size=6, K_gen=2, K_LS=1, W=2, b_LS=2)
    assert ev.ffc <= 12
    assert best.shape == (N_EDGES,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
