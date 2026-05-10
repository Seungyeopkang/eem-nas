"""Proxy evaluation backends for SEM-NAS.

* :class:`PrecomputedProxyBackend` — table lookup over precomputed pickles.
* :class:`OnlineProxyBackend` — on-the-fly NB-201 build + zero-cost proxy
  computation in PyTorch. Each call to ``evaluate(encoding)`` constructs
  the architecture, initializes its weights, and runs the chosen proxy on
  a fixed cached minibatch.

Both backends are interchangeable as the ``backend`` argument to
:class:`sem_nas.evaluator.FitnessEvaluator`.
"""
from .backends import OnlineProxyBackend, PrecomputedProxyBackend, ProxyBackend

__all__ = ["ProxyBackend", "PrecomputedProxyBackend", "OnlineProxyBackend"]
