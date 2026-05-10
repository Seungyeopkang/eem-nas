"""FFC-bounded fitness evaluator with a pluggable proxy backend.

The evaluator delegates the actual proxy score to a
:class:`sem_nas.proxy.ProxyBackend`. Two reference backends are bundled:

* :class:`PrecomputedProxyBackend` — looks up the score from precomputed
  NAS-Bench-201 arrays. Use this to reproduce the paper headline at
  millisecond-scale per-FFC compute time.
* :class:`OnlineProxyBackend` — builds the actual NB-201 architecture from
  the length-6 encoding and computes the chosen zero-cost proxy in
  PyTorch on every call. Each candidate produced by the search loop
  triggers one real proxy computation.

Every call to :meth:`FitnessEvaluator.evaluate` charges exactly one fitness
function call (FFC), including duplicate architecture queries. This is the
budget convention used in the paper. ``best_score_per_ffc`` records the
running best after each FFC unit so the same run can be sliced to any FFC
budget after the fact.
"""
from __future__ import annotations

import numpy as np

from .encoding import encoding_to_index
from .proxy import PrecomputedProxyBackend, ProxyBackend


class FitnessEvaluator:
    """Strict-FFC evaluator.

    Args:
        backend: a :class:`ProxyBackend` whose ``evaluate(encoding) -> float``
            is invoked exactly once per FFC.
        max_evals: FFC budget. Once ``ffc >= max_evals``, ``budget_exhausted``
            is true and the search loop should stop.
    """

    def __init__(self, backend: ProxyBackend, max_evals: int):
        self.backend = backend
        self.max_evals = int(max_evals)

        self.ffc = 0
        # Per-evaluation running best. After a run completes,
        # best_score_per_ffc[t-1] is the best proxy score over the first t
        # evaluations. Combined with queried_idx_per_ffc, this lets the same
        # run answer "what was the running best at FFC = b?" for any
        # b <= max_evals without rerunning the search.
        self.best_score_per_ffc: list[float] = []
        self.best_idx_per_ffc: list[int] = []
        self.queried_idx_per_ffc: list[int] = []
        self._best_score: float = -np.inf
        self._best_idx: int = -1

    # -----------------------------------------------------------------
    # Convenience constructor: precomputed-array shorthand
    # -----------------------------------------------------------------

    @classmethod
    def from_arrays(cls, proxy_scores, test_accuracy,
                    max_evals: int) -> "FitnessEvaluator":
        """Construct an evaluator backed by precomputed proxy/test arrays."""
        backend = PrecomputedProxyBackend(proxy_scores, test_accuracy)
        return cls(backend, max_evals=max_evals)

    # -----------------------------------------------------------------
    # FFC-charged evaluation
    # -----------------------------------------------------------------

    def evaluate(self, individual) -> float:
        """Charge one FFC and return the proxy score for ``individual``."""
        idx = encoding_to_index(individual)
        self.ffc += 1
        score = float(self.backend.evaluate(individual))
        if not np.isfinite(score):
            score = -np.inf
        if score > self._best_score:
            self._best_score = score
            self._best_idx = int(idx)
        self.best_score_per_ffc.append(self._best_score)
        self.best_idx_per_ffc.append(self._best_idx)
        self.queried_idx_per_ffc.append(int(idx))
        return score

    def evaluate_batch(self, population) -> np.ndarray:
        return np.array([self.evaluate(ind) for ind in population], dtype=np.float64)

    def get_test_accuracy(self, individual) -> float:
        return float(self.backend.test_accuracy(individual))

    def budget_exhausted(self) -> bool:
        return self.ffc >= self.max_evals
