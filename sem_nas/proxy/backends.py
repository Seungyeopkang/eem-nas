"""Online proxy evaluation backend for NAS-Bench-201.

Every call to :meth:`OnlineProxyBackend.evaluate` builds the actual
NAS-Bench-201 architecture from its length-6 encoding, initializes the
network with a deterministic seed, runs the chosen zero-cost proxy on a
fixed cached minibatch, and returns one scalar. The search loop sees a
real PyTorch proxy computation per FFC.

The optional :class:`NB201Api` provides the trained test accuracy of the
returned architecture (used only as a downstream diagnostic); the search
loop never reads it.

A bare-bones :class:`PrecomputedProxyBackend` is also included for unit
testing with synthetic 15,625-entry arrays. It is not part of the
recommended user-facing flow.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..encoding import encoding_to_index
from .nb201_api import NB201Api


class ProxyBackend:
    """Common interface used by :class:`FitnessEvaluator`."""

    def evaluate(self, encoding) -> float:
        raise NotImplementedError

    def test_accuracy(self, encoding) -> float:
        return float("nan")


# ---------------------------------------------------------------------------
# Online (per-candidate PyTorch compute)
# ---------------------------------------------------------------------------


class OnlineProxyBackend(ProxyBackend):
    """Build a NAS-Bench-201 architecture and compute the proxy from scratch.

    Args:
        proxy_name: one of ``zico, nwot, synflow, jacov, snip, grad_norm,
            fisher`` (see :mod:`sem_nas.proxy.proxies`).
        dataset: one of ``cifar10, cifar100, imagenet16_120``.
        batch_size: minibatch size for proxy evaluation.
        n_batches: number of cached minibatches. ZiCo needs at least 2.
        device: ``cpu`` or ``cuda``.
        data_source: ``torchvision`` (default for cifar10/cifar100) /
            ``imagenet16`` (NB-201 ImageNet-16-120 directory) / ``random``
            (synthetic Gaussian fallback for tests and no-data settings).
        init_seed: deterministic re-initialization seed for every
            architecture build.
        cell_seed: random seed for the cached minibatches.
        cells_per_stage: number of NB-201 cells per stage (paper-final
            backbone uses 5).
        cache_results: if True, repeated queries for the same architecture
            return the cached score (still charged 1 FFC). Highly
            recommended online: RTS replacement and the paired-seeds
            protocol both make duplicates frequent.
        nb201_api: optional :class:`NB201Api` for trained-accuracy lookup.
            If provided, :meth:`test_accuracy` returns the NB-201 published
            test accuracy of the queried architecture.
        torchvision_root, imagenet16_root: where to read / cache the real
            datasets when ``data_source`` is set accordingly.
    """

    def __init__(self, proxy_name: str, dataset: str, *,
                 batch_size: int = 16, n_batches: int = 2,
                 device: str = "cpu",
                 data_source: str = "torchvision",
                 init_seed: int = 0,
                 cell_seed: int = 0,
                 cells_per_stage: int = 5,
                 cache_results: bool = True,
                 nb201_api: Optional[NB201Api] = None,
                 torchvision_root: Optional[str] = None,
                 imagenet16_root: Optional[str] = None):
        # Lazy imports keep the rest of the package usable without PyTorch.
        from .data import BatchSource, NB201_DATASET_SHAPES
        from .proxies import PROXY_NAMES, needs_multi_batch

        if proxy_name not in PROXY_NAMES:
            raise ValueError(
                f"unknown proxy {proxy_name!r}; must be one of {PROXY_NAMES}"
            )
        if dataset not in NB201_DATASET_SHAPES:
            raise ValueError(
                f"unknown dataset {dataset!r}; must be one of "
                f"{tuple(NB201_DATASET_SHAPES)}"
            )
        if needs_multi_batch(proxy_name) and n_batches < 2:
            raise ValueError(
                f"proxy {proxy_name!r} requires n_batches >= 2 (was {n_batches})"
            )

        # imagenet16_120 with torchvision is unsupported; fall through to
        # imagenet16 / random.
        if dataset == "imagenet16_120" and data_source == "torchvision":
            data_source = "imagenet16"

        self.proxy_name = proxy_name
        self.dataset = dataset
        self.device = device
        self.init_seed = int(init_seed)
        self.cells_per_stage = int(cells_per_stage)
        self.num_classes = int(NB201_DATASET_SHAPES[dataset][3])
        self._batches = BatchSource(
            dataset=dataset,
            batch_size=int(batch_size),
            n_batches=int(n_batches),
            device=device,
            source=data_source,
            seed=int(cell_seed),
            **({"torchvision_root": torchvision_root}
               if torchvision_root is not None else {}),
            **({"imagenet16_root": imagenet16_root}
               if imagenet16_root is not None else {}),
        )
        self._cache: Optional[dict[int, float]] = {} if cache_results else None
        self._nb201_api = nb201_api

    # ------------------------------------------------------------------

    def _build_and_score(self, encoding) -> float:
        from .nb201 import build_network
        from . import proxies

        net = build_network(
            encoding,
            num_classes=self.num_classes,
            cells_per_stage=self.cells_per_stage,
            init_seed=self.init_seed,
            device=self.device,
        )
        net.train()  # BN must be in train mode so per-batch statistics flow.
        name = self.proxy_name
        if name == "synflow":
            return proxies.synflow(net, x=self._batches.first_batch()[0])
        if name == "snip":
            x, y = self._batches.first_batch()
            return proxies.snip(net, x, y)
        if name == "grad_norm":
            x, y = self._batches.first_batch()
            return proxies.grad_norm(net, x, y)
        if name == "nwot":
            x, _ = self._batches.first_batch()
            return proxies.nwot(net, x)
        if name == "jacov":
            x, _ = self._batches.first_batch()
            return proxies.jacov(net, x)
        if name == "fisher":
            x, y = self._batches.first_batch()
            return proxies.fisher(net, x, y)
        if name == "zico":
            return proxies.zico(net, self._batches.batches())
        raise ValueError(f"unknown proxy {name!r}")  # pragma: no cover

    def evaluate(self, encoding) -> float:
        if self._cache is not None:
            idx = encoding_to_index(encoding)
            cached = self._cache.get(idx)
            if cached is not None:
                return cached
        score = float(self._build_and_score(encoding))
        if self._cache is not None:
            self._cache[idx] = score
        return score

    def test_accuracy(self, encoding) -> float:
        if self._nb201_api is None:
            return float("nan")
        return self._nb201_api.test_accuracy(encoding, self.dataset)


# ---------------------------------------------------------------------------
# Precomputed (kept for unit tests)
# ---------------------------------------------------------------------------


class PrecomputedProxyBackend(ProxyBackend):
    """Lookup from precomputed length-15625 arrays. Used by unit tests.

    This is not the main user-facing path; the package is built around
    online proxy compute via :class:`OnlineProxyBackend`.
    """

    def __init__(self, proxy_scores, test_accuracy):
        self.proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
        self.test_accuracy_arr = np.asarray(test_accuracy, dtype=np.float64)

    def evaluate(self, encoding) -> float:
        idx = encoding_to_index(encoding)
        score = float(self.proxy_scores[idx])
        if not np.isfinite(score):
            return float("-inf")
        return score

    def test_accuracy(self, encoding) -> float:
        return float(self.test_accuracy_arr[encoding_to_index(encoding)])
