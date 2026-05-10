"""Run a single (method, dataset, proxy, seed) cell at one FFC budget.

Two proxy backends are supported:

* ``--backend online`` (default): every candidate triggers a real NB-201
  build plus PyTorch zero-cost proxy computation on a cached minibatch.
* ``--backend precomputed``: looks the proxy score up in
  ``data/nb201_<dataset>.pkl``. Fast for reproducing the paper headline.

Examples::

    python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0
    python -m scripts.run_one --method generic_ga --dataset cifar100 --proxy nwot \
        --backend precomputed --data_dir ../data
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sem_nas.baselines import BASELINES
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.proxy import OnlineProxyBackend, PrecomputedProxyBackend
from sem_nas.proxy.proxies import PROXY_NAMES
from sem_nas.sem_nas import run as run_sem_nas
from scripts.load_proxy_pickle import DATASETS, load_proxy

DEFAULT_FFC = 100
METHOD_CHOICES = ("sem_nas", *BASELINES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--proxy", required=True, choices=PROXY_NAMES,
                        help="zero-cost proxy to optimize")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffc", type=int, default=DEFAULT_FFC)
    parser.add_argument("--backend", choices=("online", "precomputed"),
                        default="online")

    online = parser.add_argument_group("online-backend options")
    online.add_argument("--device", default="cpu", help="cpu or cuda")
    online.add_argument("--batch_size", type=int, default=16)
    online.add_argument("--n_batches", type=int, default=2,
                        help="ZiCo needs >= 2; other proxies use the first batch only")
    online.add_argument("--data_source", choices=("random", "torchvision"),
                        default="random")
    online.add_argument("--cells_per_stage", type=int, default=2)

    precomp = parser.add_argument_group("precomputed-backend options")
    precomp.add_argument("--data_dir", type=str, default=None,
                         help="directory containing nb201_<dataset>.pkl")

    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "results"))
    return parser.parse_args()


def make_backend(args: argparse.Namespace):
    if args.backend == "precomputed":
        proxy_scores, test_accuracy = load_proxy(args.dataset, args.proxy,
                                                 data_dir=args.data_dir)
        return PrecomputedProxyBackend(proxy_scores, test_accuracy)
    return OnlineProxyBackend(
        proxy_name=args.proxy,
        dataset=args.dataset,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=args.device,
        data_source=args.data_source,
        init_seed=args.seed,
        cell_seed=args.seed,
        cells_per_stage=args.cells_per_stage,
    )


def run_method(method: str, evaluator: FitnessEvaluator) -> np.ndarray:
    if method == "sem_nas":
        best, _, _ = run_sem_nas(evaluator)
        return best
    return BASELINES[method](evaluator)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    backend = make_backend(args)
    evaluator = FitnessEvaluator(backend, max_evals=args.ffc)

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
        "backend": args.backend,
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
    out_path = out_dir / (
        f"{args.method}_{args.dataset}_{args.proxy}_"
        f"seed{args.seed}_ffc{args.ffc}_{args.backend}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    test_acc_str = "n/a" if not np.isfinite(test_acc) else f"{test_acc:.2f}%"
    print(f"[{args.method}/{args.backend}] {args.dataset}/{args.proxy} "
          f"seed={args.seed} ffc={evaluator.ffc} "
          f"proxy={best_score:.4f} test_acc={test_acc_str} "
          f"({elapsed:.2f}s) -> {out_path}")


if __name__ == "__main__":
    main()
