"""Proxy evaluation backends.

The search side calls ``backend.evaluate(encoding) -> float`` on every
fitness function call (FFC). Two reference backends are provided.

* :class:`PrecomputedProxyBackend` — table lookup over the precomputed
  NAS-Bench-201 proxy/test arrays. Used to reproduce the paper headline
  with millisecond-scale per-FFC compute time.
* :class:`OnlineProxyBackend` — builds the actual NB-201 architecture from
  the length-6 encoding, initializes its weights, runs one or more
  forward/backward passes through the chosen zero-cost proxy on a fixed
  cached minibatch, and returns the resulting scalar. Each candidate
  produced by the search loop triggers a real PyTorch proxy computation.

Both backends share the same minimal interface, so the SEM-NAS and baseline
search loops are unchanged when swapping between offline and online modes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..encoding import encoding_to_index


class ProxyBackend:
    """Common backend interface used by :class:`FitnessEvaluator`."""

    def evaluate(self, encoding) -> float:
        raise NotImplementedError

    def test_accuracy(self, encoding) -> float:
        return float("nan")


# ---------------------------------------------------------------------------
# Precomputed (table lookup)
# ---------------------------------------------------------------------------


class PrecomputedProxyBackend(ProxyBackend):
    """Look up proxy/test scores from precomputed NAS-Bench-201 arrays.

    Args:
        proxy_scores: length-15625 array of proxy scores in higher-is-better
            orientation.
        test_accuracy: length-15625 array of trained test accuracy.
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


# ---------------------------------------------------------------------------
# Online (PyTorch)
# ---------------------------------------------------------------------------


class OnlineProxyBackend(ProxyBackend):
    """Build the NB-201 architecture and compute the proxy from scratch.

    Args:
        proxy_name: one of ``zico, nwot, synflow, jacov, snip, grad_norm,
            fisher`` (see :mod:`sem_nas.proxy.proxies`).
        dataset: one of ``cifar10, cifar100, imagenet16_120``. Determines
            the input shape and number of classes used by the network and
            data source.
        batch_size: minibatch size for proxy evaluation. Smaller is faster.
        n_batches: number of cached minibatches. Most proxies use one;
            ZiCo needs at least two.
        device: ``'cpu'`` or ``'cuda'``.
        data_source: ``'random'`` (default, no download) or
            ``'torchvision'`` (downloads CIFAR-10/100 once).
        init_seed: deterministic re-initialization seed for every
            architecture build.
        cell_seed: random seed for the input minibatch.
        cells_per_stage: number of NB-201 cells per stage in the network
            (default 2). Lower values reduce per-FFC wall time.
        cache_results: if ``True``, repeated queries for the same
            architecture skip the recomputation (still charged 1 FFC by
            the evaluator). Recommended in the online setting because
            duplicate queries do occur during the search loop.
    """

    def __init__(self, proxy_name: str, dataset: str, *,
                 batch_size: int = 16, n_batches: int = 2,
                 device: str = "cpu",
                 data_source: str = "random",
                 init_seed: int = 0,
                 cell_seed: int = 0,
                 cells_per_stage: int = 2,
                 cache_results: bool = True):
        # Lazy imports so the rest of the package stays usable without PyTorch.
        from .proxies import PROXY_NAMES, needs_multi_batch
        from .data import BatchSource, DATASET_SHAPES

        if proxy_name not in PROXY_NAMES:
            raise ValueError(
                f"unknown proxy {proxy_name!r}; must be one of {PROXY_NAMES}"
            )
        if needs_multi_batch(proxy_name) and n_batches < 2:
            raise ValueError(
                f"proxy {proxy_name!r} requires n_batches >= 2 (was {n_batches})"
            )

        self.proxy_name = proxy_name
        self.dataset = dataset
        self.device = device
        self.init_seed = int(init_seed)
        self.cells_per_stage = int(cells_per_stage)
        self.num_classes = int(DATASET_SHAPES[dataset][3])
        self._batches = BatchSource(
            dataset=dataset, batch_size=int(batch_size),
            n_batches=int(n_batches), device=device,
            source=data_source, seed=int(cell_seed),
        )
        self._cache: Optional[dict[int, float]] = {} if cache_results else None

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
