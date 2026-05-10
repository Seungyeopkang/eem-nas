"""NAS-Bench-201 TSS architecture, built from a length-6 encoding.

A NAS-Bench-201 cell is a complete four-node DAG with six ordered edges; each
edge selects one of five operations. The encoding ``a = (a_01, a_02, a_12,
a_03, a_13, a_23)`` is the same length-6 vector used by the search side.

For the online proxy backend we build a small NAS-Bench-201-style network:
``stem -> N cells -> reduction -> N cells -> reduction -> N cells -> head``.
This follows the original NB-201 backbone shape (stem, three stages, two
reductions, classifier) but with a small ``N`` per stage to keep the per-FFC
proxy compute time low. Operation primitives (``ReLUConvBN``, residual
reduction, etc.) match the conventions in the NAS-Bench-201 repository.

Requires PyTorch.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The online proxy backend requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc

from ..encoding import EDGES, N_EDGES, N_OPS, OPS

# ---------------------------------------------------------------------------
# Cell-level operation primitives
# ---------------------------------------------------------------------------


class ReLUConvBN(nn.Module):
    """Standard NB-201 ``ReLU-Conv-BN`` op with kernel size in {1, 3}."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    """``none`` op: outputs a zero tensor with the same shape as the input."""

    def forward(self, x):
        return x.mul(0.0)


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


def make_op(op_id: int, channels: int) -> nn.Module:
    """Instantiate a NB-201 operation by integer id (0..4)."""
    if op_id == 0:
        return Zero()
    if op_id == 1:
        return Identity()
    if op_id == 2:
        return ReLUConvBN(channels, kernel_size=1)
    if op_id == 3:
        return ReLUConvBN(channels, kernel_size=3)
    if op_id == 4:
        return AvgPool()
    raise ValueError(f"unknown op id {op_id!r}; must be in [0, 5)")


# ---------------------------------------------------------------------------
# Cell: 4-node DAG, 6 ordered edges
# ---------------------------------------------------------------------------


class NB201Cell(nn.Module):
    """One NAS-Bench-201 cell with operations selected by the encoding.

    The encoding entries map to edges in the following order::

        encoding[0]  ->  edge (0 -> 1)
        encoding[1]  ->  edge (0 -> 2)
        encoding[2]  ->  edge (1 -> 2)
        encoding[3]  ->  edge (0 -> 3)
        encoding[4]  ->  edge (1 -> 3)
        encoding[5]  ->  edge (2 -> 3)

    Node 3 is the cell output, computed as the sum of the three ops on its
    incoming edges.
    """

    def __init__(self, encoding, channels: int):
        super().__init__()
        encoding = list(int(op) for op in encoding)
        assert len(encoding) == N_EDGES, f"expected length-{N_EDGES} encoding"
        self.ops = nn.ModuleList([make_op(op, channels) for op in encoding])

    def forward(self, x):
        # Node 0 is the cell input. We compute n1, n2, n3 by summing
        # incoming edges in the canonical NB-201 edge order.
        n1 = self.ops[0](x)
        n2 = self.ops[1](x) + self.ops[2](n1)
        n3 = self.ops[3](x) + self.ops[4](n1) + self.ops[5](n2)
        return n3


class ResNetReductionStem(nn.Module):
    """Reduction (channel double, spatial halve) used between NB-201 stages."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Network: stem -> N x cell -> reduction -> N x cell -> reduction -> N x cell -> head
# ---------------------------------------------------------------------------


class NB201Network(nn.Module):
    """A NAS-Bench-201-style network built from a length-6 encoding.

    The cell template is repeated ``cells_per_stage`` times in each of the
    three stages (initial / mid / late), with channel-doubling reductions in
    between. Default values give a small, fast network that is sufficient
    for zero-cost proxy computation.
    """

    def __init__(self, encoding, num_classes: int = 10,
                 stem_channels: int = 16, cells_per_stage: int = 2):
        super().__init__()
        c1 = int(stem_channels)
        c2 = c1 * 2
        c3 = c2 * 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
        )

        cells = []
        for _ in range(cells_per_stage):
            cells.append(NB201Cell(encoding, c1))
        cells.append(ResNetReductionStem(c1, c2))
        for _ in range(cells_per_stage):
            cells.append(NB201Cell(encoding, c2))
        cells.append(ResNetReductionStem(c2, c3))
        for _ in range(cells_per_stage):
            cells.append(NB201Cell(encoding, c3))
        self.cells = nn.Sequential(*cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(c3), nn.ReLU(inplace=False))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, int(num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.lastact(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


def build_network(encoding, num_classes: int = 10, *,
                  stem_channels: int = 16, cells_per_stage: int = 2,
                  init_seed: int | None = None,
                  device: str | torch.device = "cpu") -> NB201Network:
    """Construct an :class:`NB201Network` and move it to ``device``.

    If ``init_seed`` is given, the network parameters are reinitialized from
    that seed so subsequent zero-cost proxy calls on the same architecture
    are deterministic.
    """
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))
    net = NB201Network(
        encoding,
        num_classes=int(num_classes),
        stem_channels=int(stem_channels),
        cells_per_stage=int(cells_per_stage),
    )
    return net.to(device)
