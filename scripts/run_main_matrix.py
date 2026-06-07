"""Run the 84-cell main comparison matrix on NAS-Bench-201.

The matrix is

    methods (5)  x  proxies (7)  x  datasets (3)  x  seeds (N)

where the methods are EEM-NAS plus four baselines (random,
aging_evolution, simple_mutation, generic_ga). Every candidate is
evaluated by a real PyTorch zero-cost proxy computation; the
``NAS-Bench-201-v1_1-096897.pth`` is auto-downloaded and used only to
record the trained test accuracy of the returned architecture.

Examples::

    python -m scripts.run_main_matrix --workers 8 --seeds 100
    python -m scripts.run_main_matrix --workers 8 --seeds 100 \\
        --methods eem_nas,generic_ga --proxies zico,nwot
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eem_nas.baselines import BASELINES
from eem_nas.evaluator import FitnessEvaluator
from eem_nas.proxy import NB201Api, NB201_DATASETS, OnlineProxyBackend
from eem_nas.proxy.proxies import PROXY_NAMES
from eem_nas.eem_nas import run as run_eem_nas
from scripts.download_nb201 import ensure_nb201_api

DEFAULT_METHODS = ("eem_nas", "random", "aging_evolution",
                   "simple_mutation", "generic_ga")
DEFAULT_PROXIES = PROXY_NAMES  # 7 proxies
DEFAULT_FFC = 100
DEFAULT_SEEDS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str,
                        default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", type=str,
                        default=",".join(NB201_DATASETS))
    parser.add_argument("--proxies", type=str,
                        default=",".join(DEFAULT_PROXIES))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ffc", type=int, default=DEFAULT_FFC)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--data_source", choices=("torchvision", "imagenet16", "random"),
                        default="torchvision")
    parser.add_argument("--imagenet16_root", type=str, default=None)
    parser.add_argument("--cells_per_stage", type=int, default=5)
    parser.add_argument("--no_test_accuracy", action="store_true")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] /
                                    "results" / "main_matrix"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run_cell(args: tuple) -> tuple[str, float]:
    method, dataset, proxy, seed, ffc, opts, out_dir, overwrite = args

    out_path = Path(out_dir) / (
        f"{method}_{dataset}_{proxy}_seed{seed}_ffc{ffc}.pkl"
    )
    if out_path.exists() and not overwrite:
        return str(out_path), 0.0

    nb201_api = None
    if not opts.get("no_test_accuracy", False):
        api_path = ensure_nb201_api()
        nb201_api = NB201Api(api_path, datasets=(dataset,))

    backend = OnlineProxyBackend(
        proxy_name=proxy,
        dataset=dataset,
        batch_size=int(opts.get("batch_size", 16)),
        n_batches=int(opts.get("n_batches", 2)),
        device=opts.get("device", "cpu"),
        data_source=opts.get("data_source", "torchvision"),
        init_seed=int(seed),
        cell_seed=int(seed),
        cells_per_stage=int(opts.get("cells_per_stage", 5)),
        nb201_api=nb201_api,
        imagenet16_root=opts.get("imagenet16_root"),
    )

    np.random.seed(int(seed))
    evaluator = FitnessEvaluator(backend, max_evals=int(ffc))
    t0 = time.time()
    if method == "eem_nas":
        best, _, _ = run_eem_nas(evaluator)
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
        "device": opts.get("device", "cpu"),
        "data_source": opts.get("data_source", "torchvision"),
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
    opts = {
        "device": args.device,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "data_source": args.data_source,
        "cells_per_stage": args.cells_per_stage,
        "imagenet16_root": args.imagenet16_root,
        "no_test_accuracy": args.no_test_accuracy,
    }

    cells = [
        (m, d, p, s, args.ffc, opts, args.out_dir, args.overwrite)
        for m in methods for d in datasets for p in proxies for s in seeds
    ]
    print(f"running {len(cells)} cells with {args.workers} workers (online proxy compute)")

    t0 = time.time()
    if args.workers <= 1:
        for cell in cells:
            path, dt = _run_cell(cell)
            if dt > 0.0:
                print(f"  wrote {path} ({dt:.2f}s)")
    else:
        with mp.Pool(args.workers) as pool:
            for path, dt in pool.imap_unordered(_run_cell, cells):
                if dt > 0.0:
                    print(f"  wrote {path} ({dt:.2f}s)")

    print(f"done in {time.time() - t0:.1f}s; results in {args.out_dir}")


if __name__ == "__main__":
    main()
