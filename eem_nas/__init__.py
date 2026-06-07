"""Evaluation Efficient Memetic NAS (EEM-NAS) reference implementation.

Reference for the paper *Budgeted Fixed-Proxy Search for Zero-Shot NAS on
NAS-Bench-201 via Evaluation Efficient Memetic NAS* (Electronics, 2026).

Public entry points:

* :func:`eem_nas.run` — the proposed EEM-NAS algorithm (Algorithm 1).
* :data:`baselines.BASELINES` — registry of the four baselines plus the two
  forced-edit controls used in the §4 fairness diagnostic.
* :class:`evaluator.FitnessEvaluator` — strict-FFC evaluator. Construct it
  with either a :class:`proxy.PrecomputedProxyBackend` (table lookup) or a
  :class:`proxy.OnlineProxyBackend` (on-the-fly NB-201 build + zero-cost
  proxy computation).

NAS-Bench-201 TSS encoding constants and bijection are in
:mod:`eem_nas.encoding`.
"""
from . import baselines, encoding, evaluator, local_search, operators, primitives, proxy, eem_nas

__all__ = [
    "baselines",
    "encoding",
    "evaluator",
    "local_search",
    "operators",
    "primitives",
    "proxy",
    "eem_nas",
]
