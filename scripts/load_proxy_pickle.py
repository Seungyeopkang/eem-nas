"""Helpers for loading precomputed NAS-Bench-201 proxy pickles.

Each pickle is a dict with the following keys:

* ``meta`` — benchmark metadata (dataset name, proxy list, etc.).
* ``test_accuracy`` — length-15625 array of test accuracy.
* ``proxies`` — dict mapping proxy name to a length-15625 score array.

The code release does not bundle the pickles; see ``README.md`` for the
generation script. The default location is::

    <project_root>/data/nb201_<dataset>.pkl
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

DATASETS = ("cifar10", "cifar100", "imagenet16_120")
DEFAULT_DATA_DIR = Path(
    os.environ.get("SEM_NAS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
)


def proxy_pickle_path(dataset: str, data_dir: str | os.PathLike | None = None) -> Path:
    """Return the path to ``nb201_<dataset>.pkl``."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; must be one of {DATASETS}")
    base = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return base / f"nb201_{dataset}.pkl"


def load_proxy(dataset: str, proxy: str,
               data_dir: str | os.PathLike | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(proxy_scores, test_accuracy)`` arrays for ``(dataset, proxy)``."""
    path = proxy_pickle_path(dataset, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"NAS-Bench-201 pickle not found at {path}. "
            "See README.md > 'Preparing the data' for instructions."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if proxy not in bundle["proxies"]:
        available = sorted(bundle["proxies"].keys())
        raise KeyError(f"proxy {proxy!r} not in pickle; available: {available}")
    return (
        np.asarray(bundle["proxies"][proxy], dtype=np.float64),
        np.asarray(bundle["test_accuracy"], dtype=np.float64),
    )
