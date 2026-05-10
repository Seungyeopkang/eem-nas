"""FFC-bounded fitness evaluator over precomputed NAS-Bench-201 proxy arrays.

Every call to :meth:`FitnessEvaluator.evaluate` charges exactly one fitness
function call (FFC), including duplicate architecture queries by the search
loop. This is the budget convention used in the paper. ``best_score_per_ffc``
records the running best proxy score after each FFC unit, so the same run can
be sliced to any FFC budget after the fact (used by the budget-curve sweep).
"""
from __future__ import annotations

import numpy as np

from .encoding import encoding_to_index


class FitnessEvaluator:
    """One-proxy lookup evaluator with strict FFC accounting.

    Args:
        proxy_scores: length-15625 array of precomputed proxy scores in the
            higher-is-better orientation expected by the search loop.
        test_accuracy: length-15625 array of trained test accuracy. Used only
            for the optional archive-level diagnostic; never read inside the
            search algorithms.
        max_evals: FFC budget. Once ``ffc >= max_evals``, ``budget_exhausted``
            is true and the search loop should stop.
    """

    def __init__(self, proxy_scores: np.ndarray, test_accuracy: np.ndarray,
                 max_evals: int):
        self.proxy_scores = np.asarray(proxy_scores, dtype=np.float64)
        self.test_accuracy = np.asarray(test_accuracy, dtype=np.float64)
        self.max_evals = int(max_evals)

        self.ffc = 0
        # Running best across all evaluate() calls. After a run completes,
        # best_score_per_ffc[t-1] is the best proxy score over the first t
        # evaluations and best_idx_per_ffc[t-1] is the matching architecture
        # index. Together with queried_idx_per_ffc, this lets the same run
        # answer "what was the running best at FFC = b?" for any b <= max_evals.
        self.best_score_per_ffc: list[float] = []
        self.best_idx_per_ffc: list[int] = []
        self.queried_idx_per_ffc: list[int] = []
        self._best_score: float = -np.inf
        self._best_idx: int = -1

    def evaluate(self, individual) -> float:
        """Return ``proxy_scores[idx]`` and charge one FFC."""
        idx = encoding_to_index(individual)
        self.ffc += 1
        score = float(self.proxy_scores[idx])
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
        return float(self.test_accuracy[encoding_to_index(individual)])

    def budget_exhausted(self) -> bool:
        return self.ffc >= self.max_evals
