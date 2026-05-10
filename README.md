# Sample-Efficient Memetic NAS (SEM-NAS)

Reference implementation accompanying the paper

> **Budgeted Fixed-Proxy Search for Zero-Shot NAS on NAS-Bench-201 via Sample-Efficient Memetic NAS** (Electronics, 2026).

SEM-NAS is a single-population memetic procedure for **budgeted fixed-proxy zero-shot NAS** on NAS-Bench-201 TSS. Each run optimizes one fixed zero-cost proxy under a strict fitness function call budget (FFC = 100), and queries only 100 of the 15,625 architectures.

This package contains a clean reproduction of the four primitives described in Section 3 of the paper, the four budgeted fixed-proxy search baselines, an **online** zero-cost proxy backend (every candidate triggers a real PyTorch forward/backward pass on the actual NB-201 architecture), and a notebook + scripts that auto-download the required NB-201 release file.

---

## Repository layout

```
code/
├── README.md                       this file
├── requirements.txt                numpy + torch + torchvision + gdown + nas-bench-201
├── LICENSE                         MIT
├── notebooks/
│   └── run_experiments.ipynb       end-to-end walkthrough following main.ipynb
├── sem_nas/
│   ├── encoding.py                 NB-201 TSS encoding (length-6 op vector)
│   ├── evaluator.py                FFC-bounded evaluator (takes a ProxyBackend)
│   ├── operators.py                tournament selection, uniform crossover, block mutation
│   ├── local_search.py             1-flip first-improvement LS + per-call FFC view
│   ├── primitives.py               RTS, rank+distance LS targets, entropy-guided mutation
│   ├── sem_nas.py                  Algorithm 1 (proposed method)
│   ├── baselines.py                4 baselines + 2 forced-edit fairness controls
│   └── proxy/
│       ├── backends.py             OnlineProxyBackend (NB-201 only) + PrecomputedProxyBackend (tests)
│       ├── nb201.py                NB-201 cell + full backbone (PyTorch)
│       ├── nb201_api.py            wrapper around the .pth file for trained-accuracy lookup
│       ├── proxies.py              ZiCo, NWOT, SynFlow, Jacov, SNIP, GradNorm, Fisher
│       └── data.py                 torchvision (CIFAR-10/100) / imagenet16 / random
├── scripts/
│   ├── download_nb201.py           gdown-based auto-download of the NB-201 .pth
│   ├── run_one.py                  single (method, dataset, proxy, seed) cell
│   └── run_main_matrix.py          full 84-cell matrix driver
└── tests/
    └── test_sem_nas.py             smoke tests including online-backend coverage
```

---

## Online proxy compute (the only mode)

Every candidate produced by the search loop builds the actual NAS-Bench-201 architecture from its length-6 encoding, initializes the network with the run seed, and runs the chosen zero-cost proxy on a fixed cached minibatch via PyTorch.

```python
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.proxy import NB201Api, OnlineProxyBackend
from sem_nas.sem_nas import run as run_sem_nas
from scripts.download_nb201 import ensure_nb201_api

# Auto-download NAS-Bench-201-v1_1-096897.pth (~2.2 GB) on first call.
api = NB201Api(ensure_nb201_api(), datasets=('cifar10',))

backend = OnlineProxyBackend(
    proxy_name='zico',
    dataset='cifar10',
    batch_size=16, n_batches=2,
    device='cuda',                # or 'cpu'
    data_source='torchvision',    # auto-downloads CIFAR-10/100
    init_seed=0, cell_seed=0,
    nb201_api=api,                # so test_accuracy of the returned arch is filled in
)
evaluator = FitnessEvaluator(backend, max_evals=100)
best, pop, fit = run_sem_nas(evaluator)
```

The seven supported proxies are `zico, nwot, synflow, jacov, snip, grad_norm, fisher`. Each is invoked according to its standard formulation; see `sem_nas/proxy/proxies.py`.

The `.pth` file is consulted only for the trained test accuracy of the returned architecture (a downstream diagnostic). It is never read by the search loop.

---

## Required data

| What | How to get it | Used for |
|---|---|---|
| `NAS-Bench-201-v1_1-096897.pth` (~2.2 GB) | `python -m scripts.download_nb201` (gdown from Google Drive) | trained test accuracy lookup |
| CIFAR-10 / CIFAR-100 | torchvision auto-downloads on first use | proxy compute batches |
| ImageNet-16-120 (~3.7 GB) | manual: download from the [NAS-Bench-201 README](https://github.com/D-X-Y/NAS-Bench-201), unpack, then point `--imagenet16_root <path>` | proxy compute batches for the imagenet16_120 dataset |

If you only want a quick sanity run, set `data_source='random'` to use synthetic Gaussian batches; this needs no external data and is what the unit tests use.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10 is recommended. PyTorch ≥ 2.0 is required.

---

## Quick start

### Notebook

The fastest way to see everything end-to-end is the notebook (Phase 0/1/2/3 layout following the upstream `main.ipynb`):

```bash
jupyter notebook notebooks/run_experiments.ipynb
```

### Single cell (CLI)

The first call auto-downloads the `.pth` if it is missing:

```bash
python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0
python -m scripts.run_one --method generic_ga --dataset cifar100 --proxy nwot --seed 7
```

Skip the download / accuracy lookup with `--no_test_accuracy` (useful in environments without internet):

```bash
python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy synflow --seed 0 \
    --no_test_accuracy --data_source random --cells_per_stage 2
```

### Full main matrix

5 methods × 7 proxies × 3 datasets × 100 seeds at FFC = 100, parallelized across CPU workers:

```bash
python -m scripts.download_nb201
python -m scripts.run_main_matrix --workers 8 --seeds 100
```

For ImageNet-16-120 cells, additionally pass `--data_source imagenet16 --imagenet16_root <path>`.

Per-cell pickles land under `code/results/main_matrix/`. They contain the running best per FFC, the queried-architecture history, and the returned architecture's encoding / proxy score / test accuracy.

---

## SEM-NAS hyperparameters (paper-final)

| Symbol | Value | Meaning |
|---|---|---|
| `pop_size` | 10 | population size |
| `K_gen` | 4 | LS trigger period (generations) |
| `K_LS` | 3 | LS targets per trigger |
| `W` | 3 | RTS replacement-window size |
| `b_LS` | 25 | per-LS-call FFC cap |
| `mutation_prob` | 1/L = 1/6 | per-edge mutation rate |
| `tournament_size` | 5 | tournament selection (reproduction) |
| `crossover_prob` | 0.9 | uniform crossover probability |
| `cells_per_stage` | 5 | NB-201 backbone (stem → 5 cells → reduction → 5 cells → reduction → 5 cells → head) |

These values produce the 82 W / 2 T / 0 L Holm-corrected main result on the 84-cell family.

---

## Wall-clock notes

* **CPU**: depends on the proxy and `cells_per_stage`. With the full backbone (`cells_per_stage=5`) and batch size 16, expect roughly 100 ms / FFC for SynFlow, 500 ms / FFC for ZiCo. One full FFC = 100 SEM-NAS run takes 10 s – 1 minute on CPU.
* **GPU**: typically 5–15× faster for the gradient-based proxies.
* The online backend caches per-encoding scores by default so the duplicate queries inherent to RTS replacement do not re-trigger PyTorch. Each unique architecture is still charged exactly one FFC the first time it is evaluated.

---

## Available baselines

```python
from sem_nas.baselines import BASELINES
best = BASELINES["aging_evolution"](evaluator)
```

* `random` — uniform random sampling
* `aging_evolution` — Real et al. (2019)
* `simple_mutation` — steady-state mutation-only
* `generic_ga` — uniform crossover + tournament + block mutation + elitism
* `simple_mutation_forced` — Simple Mutation with at-least-one-edit fallback (paper §4 fairness control)
* `generic_ga_forced` — Generic GA with the same forced-edit fallback

---

## Tests

```bash
python -m pytest tests/ -q
```

Includes 8 search-side smoke tests plus 3 online-backend tests (build NB-201 → run every proxy → finite-output check; SEM-NAS with the online backend at a tiny FFC budget; ImageNet-16-120 routing). The tests do not require the `.pth` or downloaded image data.

```

---

## License

MIT. See `LICENSE`.
