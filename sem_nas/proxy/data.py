"""Minibatch sources for online proxy computation.

Two source modes are exposed:

* ``"random"`` — synthetic Gaussian images plus uniformly-sampled labels.
  Useful for environments without internet access, GPU, or torchvision; the
  proxy values will not match the published NB-201 numbers but the search
  loop drives real PyTorch computation end-to-end.
* ``"torchvision"`` — real CIFAR-10 / CIFAR-100 batches via torchvision
  (download required on first use). Use this to reproduce the actual
  zero-cost proxy values of the corresponding architectures.

Both modes return cached batches so repeated calls during the search loop
do not redownload or resample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The online proxy backend requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc


DATASET_SHAPES = {
    "cifar10": (3, 32, 32, 10),
    "cifar100": (3, 32, 32, 100),
    "imagenet16_120": (3, 16, 16, 120),
}


@dataclass
class BatchSource:
    """A reusable supplier of fixed minibatches.

    The first ``n_batches`` batches are generated up front and cached, so
    every proxy computation during a single run sees the same data
    distribution and so wall-clock cost stays bounded.
    """

    dataset: str
    batch_size: int = 16
    n_batches: int = 2
    device: str | torch.device = "cpu"
    source: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_SHAPES:
            raise ValueError(
                f"unknown dataset {self.dataset!r}; must be one of "
                f"{tuple(DATASET_SHAPES)}"
            )
        if self.source not in ("random", "torchvision"):
            raise ValueError(
                f"unknown source {self.source!r}; must be 'random' or 'torchvision'"
            )
        self._cached: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def _build_cached(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.source == "random":
            return _build_random_batches(
                self.dataset, self.batch_size, self.n_batches,
                seed=int(self.seed), device=self.device,
            )
        return _build_torchvision_batches(
            self.dataset, self.batch_size, self.n_batches,
            seed=int(self.seed), device=self.device,
        )

    def batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._cached is None:
            self._cached = self._build_cached()
        return self._cached

    def first_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batches()[0]


def _build_random_batches(dataset: str, batch_size: int, n_batches: int,
                          *, seed: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    c, h, w, n_classes = DATASET_SHAPES[dataset]
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for b in range(n_batches):
        x = torch.randn((batch_size, c, h, w), generator=g)
        y = torch.randint(0, n_classes, (batch_size,), generator=g, dtype=torch.long)
        out.append((x.to(device), y.to(device)))
    return out


def _build_torchvision_batches(dataset: str, batch_size: int, n_batches: int,
                               *, seed: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Real-data batches require torchvision. "
            "Install it with `pip install torchvision`."
        ) from exc

    if dataset == "imagenet16_120":
        raise NotImplementedError(
            "ImageNet-16-120 is not bundled with torchvision; please use the "
            "'random' source for it, or build a custom DataLoader."
        )

    mean_std = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    }[dataset]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])
    dataset_cls = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }[dataset]
    ds = dataset_cls(root="./data_torchvision", train=True, download=True,
                     transform=transform)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g,
    )
    out = []
    it = iter(loader)
    for _ in range(n_batches):
        x, y = next(it)
        out.append((x.to(device), y.to(device)))
    return out
