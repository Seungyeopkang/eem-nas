"""Run the 84-cell main comparison matrix at FFC=100.

The matrix is

    methods (5)  x  proxies (7)  x  datasets (3)  x  seeds (N)

where the methods are SEM-NAS plus four baselines (random, aging_evolution,
simple_mutation, generic_ga). Each cell writes one pickle per seed compatible
with ``scripts/run_one.py``.

Example::

    python -m scripts.run_main_matrix --workers 8 --seeds 100
    python -m scripts.run_main_matrix --workers 8 --seeds 100 \
        --methods sem_nas,generic_ga --proxies zico,nwot
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem_nas.baselines import BASELINES
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.sem_nas import run as run_sem_nas
from scripts.load_proxy_pickle import DATASETS, load_proxy

DEFAULT_METHODS = ("sem_nas", "random", "aging_evolution",
                   "simple_mutation", "generic_ga")
DEFAULT_PROXIES = ("zico", "nwot", "synflow", "jacov", "snip",
                   "grad_norm", "fisher")
DEFAULT_FFC = 100
DEFAULT_SEEDS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str,
                        default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", type=str,
                        default=",".join(DATASETS))
    parser.add_argument("--proxies", type=str,
                        default=",".join(DEFAULT_PROXIES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="number of seeds per cell (0..seeds-1)")
    parser.add_argument("--ffc", type=int, default=DEFAULT_FFC)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "results" / "main_matrix"))
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run cells whose result pickle already exists")
    return parser.parse_args()


def _run_cell(args: tuple) -> tuple[str, float]:
    method, dataset, proxy, seed, ffc, data_dir, out_dir, overwrite = args
    out_path = Path(out_dir) / (
        f"{method}_{dataset}_{proxy}_seed{seed}_ffc{ffc}.pkl"
    )
    if out_path.exists() and not overwrite:
        return str(out_path), 0.0

    proxy_scores, test_accuracy = load_proxy(dataset, proxy, data_dir=data_dir)
    np.random.seed(int(seed))
    evaluator = FitnessEvaluator(proxy_scores, test_accuracy, max_evals=int(ffc))
    t0 = time.time()
    if method == "sem_nas":
        best, _, _ = run_sem_nas(evaluator)
    else:
        best = BASELINES[method](evaluator)
    elapsed = time.time() - t0

    best_score = float(np.max(evaluator.best_score_per_ffc)) \
        if evaluator.best_score_per_ffc else float("-inf")
    out = {
        "method": method,
        "dataset": dataset,
        "proxy": proxy,
        "seed": int(seed),
        "ffc_budget": int(ffc),
        "ffc_used": int(evaluator.ffc),
        "best_arch_encoding": np.asarray(best, dtype=int),
        "best_proxy_score": best_score,
        "best_test_accuracy": float(evaluator.get_test_accuracy(best)),
        "best_score_per_ffc": np.asarray(evaluator.best_score_per_ffc, dtype=np.float64),
        "best_idx_per_ffc": np.asarray(evaluator.best_idx_per_ffc, dtype=np.int64),
        "queried_idx_per_ffc": np.asarray(evaluator.queried_idx_per_ffc, dtype=np.int64),
        "wall_time_seconds": float(elapsed),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return str(out_path), elapsed


def main() -> None:
    args = parse_args()
    methods = tuple(s.strip() for s in args.methods.split(",") if s.strip())
    datasets = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    proxies = tuple(s.strip() for s in args.proxies.split(",") if s.strip())
    seeds = tuple(range(int(args.seeds)))

    cells = [
        (m, d, p, s, args.ffc, args.data_dir, args.out_dir, args.overwrite)
        for m in methods for d in datasets for p in proxies for s in seeds
    ]
    print(f"running {len(cells)} cells with {args.workers} workers...")

    t0 = time.time()
    if args.workers <= 1:
        for cell in cells:
            _run_cell(cell)
    else:
        with mp.Pool(args.workers) as pool:
            for path, dt in pool.imap_unordered(_run_cell, cells):
                if dt > 0.0:
                    print(f"  wrote {path} ({dt:.2f}s)")

    print(f"done in {time.time() - t0:.1f}s; results in {args.out_dir}")


if __name__ == "__main__":
    main()
