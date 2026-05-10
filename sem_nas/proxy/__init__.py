"""Online proxy evaluation for NAS-Bench-201.

* :class:`OnlineProxyBackend` — on-the-fly NB-201 build + zero-cost proxy
  computation in PyTorch. Each call to ``evaluate(encoding)`` constructs
  the architecture, initializes its weights, and runs the chosen proxy on
  a fixed cached minibatch.
* :class:`NB201Api` — wrapper around ``NAS-Bench-201-v1_1-096897.pth`` for
  trained-test-accuracy lookup (used only as a downstream diagnostic).
* :class:`PrecomputedProxyBackend` — synthetic-array lookup, retained for
  unit tests.

Auto-download the .pth file with::

    python -m scripts.download_nb201
"""
from .backends import OnlineProxyBackend, PrecomputedProxyBackend, ProxyBackend
from .nb201_api import NB201_DATASETS, NB201Api

__all__ = [
    "ProxyBackend",
    "OnlineProxyBackend",
    "PrecomputedProxyBackend",
    "NB201Api",
    "NB201_DATASETS",
]
