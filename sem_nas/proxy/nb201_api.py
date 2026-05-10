"""Trained-test-accuracy lookup via the NAS-Bench-201 .pth file.

The official NAS-Bench-201 release (``NAS-Bench-201-v1_1-096897.pth``,
~2.2 GB, Google Drive) is read through the ``nas_201_api`` Python package.
We scan it once and cache a length-15,625 ``test_accuracy`` array per
dataset, indexed by the same length-6 encoding used by SEM-NAS.

Test accuracy is a downstream diagnostic; the search loop never reads
it. Zero-cost proxy scores are still computed online by
:class:`sem_nas.proxy.OnlineProxyBackend` on each candidate.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..encoding import N_ARCHS, OPS, encoding_to_index

NB201_DATASETS = ("cifar10", "cifar100", "imagenet16_120")

# Map our dataset names to the tags used by the upstream NB-201 API.
_NB201_API_TAG = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "imagenet16_120": "ImageNet16-120",
}


def _arch_string_to_encoding(arch_str: str) -> np.ndarray:
    """Inverse of the canonical NB-201 arch-string format.

    NB-201 strings look like::

        |op_01~0|+|op_02~0|op_12~1|+|op_03~0|op_13~1|op_23~2|

    The integer after ``~`` is the source-node index and is ignored here.
    """
    parts = arch_str.split("+")
    if len(parts) != 3:
        raise ValueError(f"bad NB-201 arch string: {arch_str!r}")
    edges: list[str] = []
    for block in parts:
        edges.extend(c for c in block.split("|") if c)
    encoding = np.zeros(6, dtype=int)
    for i, edge in enumerate(edges):
        op_name = edge.split("~")[0]
        encoding[i] = OPS.index(op_name)
    return encoding


class NB201Api:
    """Cache trained test accuracy for all 15,625 architectures × 3 datasets.

    On first construction the .pth file is loaded once via the
    ``nas_201_api`` package, and ``get_more_info`` is queried for every
    architecture and every dataset. The result is a dict of length-15625
    NumPy arrays.

    Args:
        path: location of ``NAS-Bench-201-v1_1-096897.pth``. If ``None``,
            the constructor will look for the file in
            ``code/data/nb201/`` (the default download target of
            ``scripts/download_nb201.py``).
        datasets: which of the three NB-201 datasets to preload. Skipping
            a dataset here saves ~30 seconds of API scan time.
    """

    def __init__(self, path: Optional[str | os.PathLike] = None,
                 datasets: tuple[str, ...] = NB201_DATASETS):
        if path is None:
            default = Path(__file__).resolve().parents[2] / "data" / "nb201" / \
                "NAS-Bench-201-v1_1-096897.pth"
            path = default
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"NAS-Bench-201 .pth not found at {path}.\n"
                f"Run `python -m scripts.download_nb201` to fetch it."
            )
        for ds in datasets:
            if ds not in NB201_DATASETS:
                raise ValueError(
                    f"unknown dataset {ds!r}; must be one of {NB201_DATASETS}"
                )
        self.path = path
        self._test_accuracy: dict[str, np.ndarray] = {
            ds: np.full(N_ARCHS, np.nan, dtype=np.float64) for ds in datasets
        }
        self._scan_api(datasets)

    def _scan_api(self, datasets: tuple[str, ...]) -> None:
        api = _load_nb201_api(self.path)
        for api_idx in range(len(api)):
            arch_str = api.query_meta_info_by_index(api_idx).arch_str
            try:
                encoding = _arch_string_to_encoding(arch_str)
            except ValueError:
                continue
            flat_idx = encoding_to_index(encoding)
            for ds in datasets:
                tag = _NB201_API_TAG[ds]
                more_info = api.get_more_info(api_idx, tag, hp="200",
                                              is_random=False)
                self._test_accuracy[ds][flat_idx] = float(more_info["test-accuracy"])

    def test_accuracy(self, encoding, dataset: str) -> float:
        """Return the trained test accuracy for ``encoding`` on ``dataset``."""
        if dataset not in self._test_accuracy:
            raise KeyError(
                f"dataset {dataset!r} was not preloaded; "
                f"available: {tuple(self._test_accuracy.keys())}"
            )
        return float(self._test_accuracy[dataset][encoding_to_index(encoding)])

    def test_accuracy_array(self, dataset: str) -> np.ndarray:
        """Return the full length-15625 trained-accuracy array for ``dataset``."""
        if dataset not in self._test_accuracy:
            raise KeyError(
                f"dataset {dataset!r} was not preloaded; "
                f"available: {tuple(self._test_accuracy.keys())}"
            )
        return self._test_accuracy[dataset]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_nb201_api(path: Path):
    """Open the .pth via ``nas_201_api``, working around PyTorch 2.6 strictness.

    PyTorch 2.6+ rejects unknown classes in ``torch.load`` by default.
    The NB-201 archive is trusted (it is the official release shared via
    the upstream README), so we monkey-patch ``weights_only=False`` for
    the duration of this load.
    """
    try:
        import torch  # noqa: F401
        from nas_201_api import NASBench201API
    except ImportError as exc:
        raise ImportError(
            "Reading the NB-201 .pth file requires the nas-bench-201 package. "
            "Install it with `pip install nas-bench-201`."
        ) from exc
    import torch

    _orig_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _trusted_load
    try:
        return NASBench201API(str(path), verbose=False)
    finally:
        torch.load = _orig_load
