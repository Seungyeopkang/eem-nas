"""Minibatch sources for the NAS-Bench-201 datasets.

The online proxy backend uses one or more cached minibatches per
``(dataset, seed)`` pair. Three sources are supported:

* ``torchvision`` — real CIFAR-10 / CIFAR-100 batches downloaded by
  torchvision on first use. This is the default for CIFAR-10 / CIFAR-100.
* ``imagenet16`` — real ImageNet-16-120 batches read from a local
  directory in the NAS-Bench-201 release format. The dataset is not on
  torchvision; you must download it separately and point ``data_root``
  at the unpacked archive (e.g., ``data/ImageNet16/``). See the
  NAS-Bench-201 README for the link.
* ``random`` — synthetic Gaussian batches. Used by unit tests and as a
  fallback when the real ImageNet-16-120 directory is unavailable.

All sources cache the requested batches in memory so repeated calls
during a single search run do not redownload, redecode, or resample.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The online proxy backend requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc

# (channels, height, width, num_classes) for each NB-201 dataset.
NB201_DATASET_SHAPES = {
    "cifar10": (3, 32, 32, 10),
    "cifar100": (3, 32, 32, 100),
    "imagenet16_120": (3, 16, 16, 120),
}

DEFAULT_TORCHVISION_ROOT = Path(__file__).resolve().parents[2] / "data" / "torchvision"
DEFAULT_IMAGENET16_ROOT = Path(__file__).resolve().parents[2] / "data" / "ImageNet16"


@dataclass
class BatchSource:
    """A reusable supplier of fixed minibatches for one NB-201 dataset.

    Args:
        dataset: ``cifar10`` / ``cifar100`` / ``imagenet16_120``.
        batch_size: minibatch size.
        n_batches: how many minibatches to cache. ZiCo needs at least 2;
            other proxies use only the first batch.
        device: ``cpu`` or ``cuda``.
        source: ``torchvision`` (CIFAR-10/100), ``imagenet16``
            (ImageNet-16-120 from a local NB-201 archive), or ``random``
            (Gaussian fallback, always works).
        seed: deterministic batch-sampling seed.
        torchvision_root: where torchvision should cache its downloads.
        imagenet16_root: directory containing the ImageNet-16-120 pickles.
    """

    dataset: str
    batch_size: int = 16
    n_batches: int = 2
    device: str | torch.device = "cpu"
    source: str = "torchvision"
    seed: int = 0
    torchvision_root: str | os.PathLike = field(
        default_factory=lambda: str(DEFAULT_TORCHVISION_ROOT)
    )
    imagenet16_root: str | os.PathLike = field(
        default_factory=lambda: str(DEFAULT_IMAGENET16_ROOT)
    )

    def __post_init__(self) -> None:
        if self.dataset not in NB201_DATASET_SHAPES:
            raise ValueError(
                f"unknown dataset {self.dataset!r}; must be one of "
                f"{tuple(NB201_DATASET_SHAPES)}"
            )
        if self.source not in ("torchvision", "imagenet16", "random"):
            raise ValueError(
                f"unknown source {self.source!r}; "
                "must be 'torchvision', 'imagenet16', or 'random'"
            )
        self._cached: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

    # ------------------------------------------------------------------

    def _build(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.source == "random":
            return _build_random(self.dataset, self.batch_size, self.n_batches,
                                 seed=int(self.seed), device=self.device)
        if self.source == "torchvision":
            return _build_torchvision(self.dataset, self.batch_size, self.n_batches,
                                      seed=int(self.seed), device=self.device,
                                      root=str(self.torchvision_root))
        return _build_imagenet16(self.batch_size, self.n_batches,
                                 seed=int(self.seed), device=self.device,
                                 root=str(self.imagenet16_root))

    def batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._cached is None:
            self._cached = self._build()
        return self._cached

    def first_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batches()[0]


# ---------------------------------------------------------------------------
# Source implementations
# ---------------------------------------------------------------------------


def _build_random(dataset: str, batch_size: int, n_batches: int,
                  *, seed: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    c, h, w, n_classes = NB201_DATASET_SHAPES[dataset]
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n_batches):
        x = torch.randn((batch_size, c, h, w), generator=g)
        y = torch.randint(0, n_classes, (batch_size,), generator=g, dtype=torch.long)
        out.append((x.to(device), y.to(device)))
    return out


def _build_torchvision(dataset: str, batch_size: int, n_batches: int,
                       *, seed: int, device,
                       root: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if dataset == "imagenet16_120":
        raise NotImplementedError(
            "ImageNet-16-120 is not bundled with torchvision; use "
            "data_source='imagenet16' with a local NAS-Bench-201 ImageNet16 folder, "
            "or fall back to data_source='random'."
        )
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Real CIFAR-10 / CIFAR-100 batches require torchvision. "
            "Install it with `pip install torchvision`."
        ) from exc

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
    Path(root).mkdir(parents=True, exist_ok=True)
    ds = dataset_cls(root=root, train=True, download=True, transform=transform)
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


# ---------------------------------------------------------------------------
# ImageNet-16-120 loader (NAS-Bench-201 release format)
# ---------------------------------------------------------------------------


_IMAGENET16_MEAN = (0.4811, 0.4575, 0.4078)
_IMAGENET16_STD = (0.2603, 0.2532, 0.2682)


def _build_imagenet16(batch_size: int, n_batches: int, *,
                      seed: int, device,
                      root: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Read NB-201 ImageNet-16-120 train pickles and return cached batches.

    The expected layout matches the official NAS-Bench-201 ImageNet-16
    release::

        <root>/train_data_batch_1
        <root>/train_data_batch_2
        ...
        <root>/val_data

    Each train file is a pickled dict with keys ``data`` (uint8 N×3072 array
    flattened from 3×16×16) and ``labels`` (1-indexed class labels in
    [1, 1000]). For ImageNet-16-120 we keep only labels in [1, 120].
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"ImageNet-16-120 root not found: {root_path}. "
            "Download the NAS-Bench-201 ImageNet16 archive and point "
            "data_source='imagenet16' (or imagenet16_root) at the unpacked folder. "
            "Alternatively, use data_source='random'."
        )

    images, labels = _load_imagenet16_120(root_path)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(images), generator=g)
    images = images[perm]
    labels = labels[perm]

    out = []
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        if end > len(images):
            raise RuntimeError(
                "not enough ImageNet-16-120 samples to build the requested batches"
            )
        x = images[start:end].to(device)
        y = labels[start:end].to(device)
        out.append((x, y))
    return out


def _load_imagenet16_120(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the train pickles and filter to the 120-class subset."""
    images_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    for batch_id in range(1, 11):
        path = root / f"train_data_batch_{batch_id}"
        if not path.exists():
            break
        with open(path, "rb") as f:
            payload = pickle.load(f, encoding="latin1")
        data = payload["data"]
        # data is uint8 of shape (N, 3072). Reshape to (N, 3, 16, 16).
        n = data.shape[0]
        x = torch.from_numpy(data).view(n, 3, 16, 16).float().div_(255.0)
        for c in range(3):
            x[:, c].sub_(_IMAGENET16_MEAN[c]).div_(_IMAGENET16_STD[c])
        y = torch.tensor(payload["labels"], dtype=torch.long) - 1  # 1-indexed
        # Keep only the 120-class subset.
        mask = y < 120
        images_chunks.append(x[mask])
        labels_chunks.append(y[mask])
    if not images_chunks:
        raise RuntimeError(
            f"no train_data_batch_* files found under {root}; "
            "verify that the NB-201 ImageNet16 archive is unpacked correctly."
        )
    return torch.cat(images_chunks, dim=0), torch.cat(labels_chunks, dim=0)
