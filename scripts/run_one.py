"""Run one (method, dataset, proxy, seed) cell at a given FFC budget.

Every candidate produced by the search loop triggers a real PyTorch
zero-cost proxy computation on a NAS-Bench-201 architecture (no
precomputed lookup tables). Test accuracy of the returned architecture
is read from the auto-downloaded ``NAS-Bench-201-v1_1-096897.pth``.

Examples::

    # CIFAR-10 / ZiCo, online compute (default), torchvision data
    python -m scripts.run_one --method eem_nas --dataset cifar10 --proxy zico --seed 0

    # CIFAR-100 / NWOT, generic GA, single CPU
    python -m scripts.run_one --method generic_ga --dataset cifar100 --proxy nwot --seed 7

    # Run on a GPU and use a smaller backbone for faster CPU prototyping
    python -m scripts.run_one --method eem_nas --dataset cifar10 --proxy synflow \
        --device cuda --cells_per_stage 5
"""
from __future__ import annotations

import argparse
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

DEFAULT_FFC = 100
METHOD_CHOICES = ("eem_nas", *BASELINES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--dataset", required=True, choices=NB201_DATASETS)
    parser.add_argument("--proxy", required=True, choices=PROXY_NAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffc", type=int, default=DEFAULT_FFC)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=2,
                        help="ZiCo needs >= 2; other proxies use the first batch only")
    parser.add_argument("--data_source", choices=("torchvision", "imagenet16", "random"),
                        default="torchvision",
                        help="default 'torchvision' covers cifar10/cifar100; "
                             "imagenet16_120 falls back to 'imagenet16' (NB-201 archive) "
                             "or 'random'")
    parser.add_argument("--imagenet16_root", type=str, default=None,
                        help="local NB-201 ImageNet-16 directory "
                             "(only needed when dataset=imagenet16_120 and "
                             "data_source=imagenet16)")
    parser.add_argument("--cells_per_stage", type=int, default=5)
    parser.add_argument("--no_test_accuracy", action="store_true",
                        help="skip auto-download and trained-accuracy lookup")
    parser.add_argument("--out_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-download the NB-201 .pth and load test accuracy for the dataset.
    nb201_api = None
    if not args.no_test_accuracy:
        api_path = ensure_nb201_api()
        nb201_api = NB201Api(api_path, datasets=(args.dataset,))

    backend = OnlineProxyBackend(
        proxy_name=args.proxy,
        dataset=args.dataset,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=args.device,
        data_source=args.data_source,
        init_seed=args.seed,
        cell_seed=args.seed,
        cells_per_stage=args.cells_per_stage,
        nb201_api=nb201_api,
        imagenet16_root=args.imagenet16_root,
    )

    np.random.seed(args.seed)
    evaluator = FitnessEvaluator(backend, max_evals=args.ffc)

    t0 = time.time()
    if args.method == "eem_nas":
        best, _, _ = run_eem_nas(evaluator)
    else:
        best = BASELINES[args.method](evaluator)
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
        "device": args.device,
        "data_source": args.data_source,
        "cells_per_stage": int(args.cells_per_stage),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{args.method}_{args.dataset}_{args.proxy}_seed{args.seed}_ffc{args.ffc}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    test_acc_str = "n/a" if not np.isfinite(test_acc) else f"{test_acc:.2f}%"
    print(f"[{args.method}] {args.dataset}/{args.proxy} seed={args.seed} "
          f"ffc={evaluator.ffc} proxy={best_score:.4f} test_acc={test_acc_str} "
          f"({elapsed:.2f}s) -> {out_path}")


if __name__ == "__main__":
    main()
