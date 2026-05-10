"""Run a single (method, dataset, proxy, seed) cell at one FFC budget.

Example usage::

    python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0
    python -m scripts.run_one --method generic_ga --dataset cifar100 --proxy nwot --seed 7
    python -m scripts.run_one --method aging_evolution --dataset cifar10 --proxy snip --ffc 200

The result pickle is written to ``--out_dir`` and contains the same fields
expected by the analysis scripts in the paper repository.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the parent package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem_nas.baselines import BASELINES
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.sem_nas import run as run_sem_nas
from scripts.load_proxy_pickle import DATASETS, load_proxy

DEFAULT_FFC = 100


METHOD_CHOICES = ("sem_nas", *BASELINES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--proxy", required=True,
                        help="proxy key in the pickle, e.g. zico/nwot/synflow/...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffc", type=int, default=DEFAULT_FFC,
                        help="fitness function call (FFC) budget")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="override SEM_NAS_DATA_DIR")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "results"))
    return parser.parse_args()


def run_method(method: str, evaluator: FitnessEvaluator) -> np.ndarray:
    if method == "sem_nas":
        best, _, _ = run_sem_nas(evaluator)
        return best
    return BASELINES[method](evaluator)


def main() -> None:
    args = parse_args()
    proxy_scores, test_accuracy = load_proxy(args.dataset, args.proxy,
                                             data_dir=args.data_dir)
    np.random.seed(args.seed)

    evaluator = FitnessEvaluator(proxy_scores, test_accuracy, max_evals=args.ffc)
    t0 = time.time()
    best = run_method(args.method, evaluator)
    elapsed = time.time() - t0

    best_score = float(np.max(evaluator.best_score_per_ffc)) \
        if evaluator.best_score_per_ffc else float("-inf")
    test_acc = evaluator.get_test_accuracy(best)

    out = {
        "method": args.method,
        "dataset": args.dataset,
        "proxy": args.proxy,
        "seed": int(args.seed),
        "ffc_budget": int(args.ffc),
        "ffc_used": int(evaluator.ffc),
        "best_arch_encoding": np.asarray(best, dtype=int),
        "best_proxy_score": best_score,
        "best_test_accuracy": float(test_acc),
        "best_score_per_ffc": np.asarray(evaluator.best_score_per_ffc, dtype=np.float64),
        "best_idx_per_ffc": np.asarray(evaluator.best_idx_per_ffc, dtype=np.int64),
        "queried_idx_per_ffc": np.asarray(evaluator.queried_idx_per_ffc, dtype=np.int64),
        "wall_time_seconds": float(elapsed),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}_{args.dataset}_{args.proxy}_seed{args.seed}_ffc{args.ffc}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[{args.method}] {args.dataset}/{args.proxy} seed={args.seed} "
          f"ffc={evaluator.ffc} proxy={best_score:.4f} test_acc={test_acc:.2f}% "
          f"({elapsed:.2f}s) -> {out_path}")


if __name__ == "__main__":
    main()
