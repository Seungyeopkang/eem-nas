"""Auto-download the NAS-Bench-201 release files needed by this code package.

The mandatory file is

    NAS-Bench-201-v1_1-096897.pth   (~2.2 GB, Google Drive)

It contains the trained-test-accuracy table for all 15,625 architectures
across CIFAR-10, CIFAR-100, and ImageNet-16-120. We use it only as the
source of test accuracy (a downstream diagnostic). All zero-cost proxy
scores are still computed online by the search loop.

CIFAR-10 and CIFAR-100 image batches are downloaded automatically by
torchvision the first time you instantiate :class:`OnlineProxyBackend`
with ``data_source='torchvision'``, so they need not be handled here.
ImageNet-16-120 images (the actual 16x16 dataset) are not bundled with
torchvision; if you want to run experiments on that dataset, follow the
manual instructions in the NAS-Bench-201 README.

Run::

    python -m scripts.download_nb201

After this command,
``data/nb201/NAS-Bench-201-v1_1-096897.pth`` is ready for use by
:class:`sem_nas.proxy.nb201_api.NB201Api`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# File ID copied from `main.ipynb` cell 6, which is the link published in
# the upstream NAS-Bench-201 README.
NB201_API_GDRIVE_ID = "16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_"
NB201_API_FILENAME = "NAS-Bench-201-v1_1-096897.pth"


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "nb201"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="directory in which to place the .pth file")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    return parser.parse_args()


def download_nb201_api(out_dir: str | os.PathLike,
                      *, force: bool = False) -> Path:
    """Download ``NAS-Bench-201-v1_1-096897.pth`` to ``out_dir`` via gdown.

    Returns the local path of the downloaded file. If the file already
    exists and ``force`` is false, the existing file is reused.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / NB201_API_FILENAME

    if out_path.exists() and not force:
        size_gb = out_path.stat().st_size / (1024 ** 3)
        print(f"[download_nb201] already exists: {out_path} ({size_gb:.2f} GB)")
        return out_path

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Auto-download requires gdown. Install with `pip install gdown`."
        ) from exc

    print(f"[download_nb201] downloading {NB201_API_FILENAME} from Google Drive...")
    print("                this is ~2.2 GB and may take several minutes")
    gdown.download(id=NB201_API_GDRIVE_ID, output=str(out_path), quiet=False)
    if not out_path.exists():
        raise RuntimeError(
            f"download did not produce {out_path}; "
            "the Drive quota may have been hit, please retry later."
        )
    return out_path


def ensure_nb201_api(out_dir: str | os.PathLike = DEFAULT_OUT_DIR) -> Path:
    """Idempotent helper for use from notebooks and scripts.

    Returns the local .pth path; downloads the file if it is missing.
    """
    return download_nb201_api(out_dir, force=False)


def main() -> None:
    args = parse_args()
    api_path = download_nb201_api(args.out_dir, force=args.force)
    print(f"[download_nb201] OK: {api_path}")


if __name__ == "__main__":
    main()
