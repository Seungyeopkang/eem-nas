"""NAS-Bench-201 TSS encoding.

The topology search space (TSS) of NAS-Bench-201 is a four-node DAG with six
ordered edges. Each edge selects one operation from a five-operation set, so
the search space contains exactly ``5**6 = 15,625`` distinct architectures.

We encode an architecture as a length-six integer vector
``a = (a_01, a_02, a_12, a_03, a_13, a_23)``, where each entry is the operation
index on the corresponding edge in lexicographic edge order.
The bijection ``encoding_to_index`` / ``index_to_encoding`` matches the index
convention used by the precomputed proxy pickles.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Search-space constants
# ---------------------------------------------------------------------------
N_NODES = 4
N_EDGES = 6
N_OPS = 5
N_ARCHS = N_OPS ** N_EDGES  # 15,625

OPS = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)

# Edge order in the length-6 encoding: (src_node, dst_node).
EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))


# ---------------------------------------------------------------------------
# Bijection between length-6 vector and the integer index used by the
# precomputed NAS-Bench-201 proxy/test-accuracy arrays.
# ---------------------------------------------------------------------------

def encoding_to_index(encoding) -> int:
    """Convert a length-6 op vector to its base-N_OPS integer index."""
    idx = 0
    for op in encoding:
        idx = idx * N_OPS + int(op)
    return int(idx)


def index_to_encoding(idx: int) -> np.ndarray:
    """Inverse of :func:`encoding_to_index`."""
    encoding = np.zeros(N_EDGES, dtype=int)
    idx = int(idx)
    for i in range(N_EDGES - 1, -1, -1):
        encoding[i] = idx % N_OPS
        idx //= N_OPS
    return encoding


def random_individual(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a uniformly random length-6 op vector."""
    if rng is None:
        return np.random.randint(0, N_OPS, size=N_EDGES)
    return rng.integers(0, N_OPS, size=N_EDGES, dtype=np.int64).astype(int)
