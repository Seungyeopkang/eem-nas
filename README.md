# Sample-Efficient Memetic NAS (SEM-NAS)

Reference implementation accompanying the paper

> **Budgeted Fixed-Proxy Search for Zero-Shot NAS on NAS-Bench-201 via Sample-Efficient Memetic NAS** (Electronics, 2026).

SEM-NAS is a single-population memetic procedure for **budgeted fixed-proxy zero-shot NAS** on NAS-Bench-201 TSS. Each run optimizes one fixed zero-cost proxy under a strict fitness function call budget (FFC = 100), and queries only 100 of the 15,625 architectures.

This package contains a clean reproduction of the four primitives described in Section 3 of the paper, the four budgeted fixed-proxy search baselines, two reference proxy backends (offline lookup and online PyTorch compute), and minimal scripts and a notebook to reproduce the 84-cell main comparison matrix.

---

## Repository layout

```
code/
├── README.md                       this file
├── requirements.txt                numpy + torch (torchvision optional)
├── LICENSE                         MIT
├── notebooks/
│   └── run_experiments.ipynb       end-to-end walkthrough (online proxy mode)
├── sem_nas/
│   ├── encoding.py                 NB-201 TSS encoding (length-6 op vector)
│   ├── evaluator.py                FFC-bounded evaluator (takes a ProxyBackend)
│   ├── operators.py                tournament selection, uniform crossover, block mutation
│   ├── local_search.py             1-flip first-improvement LS + per-call FFC view
│   ├── primitives.py               RTS, rank+distance LS targets, entropy-guided mutation
│   ├── sem_nas.py                  Algorithm 1 (proposed method)
│   ├── baselines.py                4 baselines + 2 forced-edit fairness controls
│   └── proxy/
│       ├── backends.py             ProxyBackend ABC + Precomputed + Online
│       ├── nb201.py                NB-201 cell + small backbone (PyTorch)
│       ├── proxies.py              ZiCo, NWOT, SynFlow, Jacov, SNIP, GradNorm, Fisher
│       └── data.py                 random + optional torchvision batch sources
├── scripts/
│   ├── run_one.py                  single (method, dataset, proxy, seed) cell
│   ├── run_main_matrix.py          full 84-cell matrix driver
│   └── load_proxy_pickle.py        helper for nb201_<dataset>.pkl loading (precomputed mode)
└── tests/
    └── test_sem_nas.py             smoke tests including online-backend coverage
```

---

## Two proxy backends

The search side is identical in both cases; only the **evaluator backend** changes.

### Online (default)

Every candidate produced by the search loop builds the actual NAS-Bench-201 architecture from its length-6 encoding, initializes the network with the run seed, and runs the chosen zero-cost proxy on a fixed cached minibatch via PyTorch.

```python
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.proxy import OnlineProxyBackend
from sem_nas.sem_nas import run as run_sem_nas

backend = OnlineProxyBackend(
    proxy_name='zico', dataset='cifar10',
    batch_size=16, n_batches=2,
    device='cuda',                # or 'cpu'
    data_source='random',         # or 'torchvision' (downloads CIFAR-10/100)
    init_seed=0, cell_seed=0,
)
evaluator = FitnessEvaluator(backend, max_evals=100)
best, pop, fit = run_sem_nas(evaluator)
```

The seven supported proxies (matching the paper) are
`zico, nwot, synflow, jacov, snip, grad_norm, fisher`. Each is invoked according to its standard formulation; see `sem_nas/proxy/proxies.py`.

The `'random'` data source generates synthetic Gaussian batches, so the package works with no internet access. Switch to `'torchvision'` (CIFAR-10/CIFAR-100 only) for proxy values that match those reported in the public benchmarks.

### Precomputed lookup

For reproducing the paper headline at millisecond-scale FFC compute time, every candidate is looked up in the precomputed pickles:

```python
import pickle, numpy as np
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.proxy import PrecomputedProxyBackend

with open('data/nb201_cifar10.pkl', 'rb') as f:
    pkl = pickle.load(f)
proxy_scores = np.asarray(pkl['proxies']['zico'])
test_accuracy = np.asarray(pkl['test_accuracy'])

backend = PrecomputedProxyBackend(proxy_scores, test_accuracy)
evaluator = FitnessEvaluator(backend, max_evals=100)
```

The pickle layout is

```
{'meta': {...},
 'test_accuracy': np.ndarray,           # shape (15625,)
 'proxies': {'zico': ..., 'nwot': ..., 'synflow': ...,
             'jacov': ..., 'snip': ..., 'grad_norm': ..., 'fisher': ...}}
```

The release does not bundle the pickles; regenerate them from the public NAS-Bench-201 release and the standard zero-cost proxy implementations, or use the online backend.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional, for the torchvision data source:
# pip install torchvision
# Optional, for the notebook:
# pip install matplotlib pandas
```

Python ≥ 3.10 is recommended. PyTorch ≥ 2.0 is required for the online backend.

---

## Quick start

### Notebook

The fastest way to see everything end-to-end (online proxy compute on a single architecture, all seven proxies, SEM-NAS run, baseline comparison) is the notebook:

```bash
jupyter notebook notebooks/run_experiments.ipynb
```

### Single cell (CLI)

Run one `(method, dataset, proxy, seed)` cell at FFC = 100 with the **online** backend (default):

```bash
python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0
python -m scripts.run_one --method aging_evolution --dataset cifar100 --proxy nwot --seed 7
```

Or with the **precomputed** backend (requires `data/nb201_<dataset>.pkl`):

```bash
python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0 \
    --backend precomputed --data_dir data
```

### Full main matrix

5 methods × 7 proxies × 3 datasets × 100 seeds at FFC = 100, parallelized across CPU workers:

```bash
# Online (every FFC triggers real PyTorch proxy compute)
python -m scripts.run_main_matrix --workers 8 --seeds 100 --backend online

# Precomputed lookup (millisecond per FFC)
python -m scripts.run_main_matrix --workers 8 --seeds 100 --backend precomputed --data_dir data
```

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

These values produce the 82 W / 2 T / 0 L Holm-corrected main result on the 84-cell family.

---

## Wall-clock notes

* **Precomputed lookup**: ≈ 5–6 ms per run at FFC = 100 (the search loop itself).
* **Online compute, CPU**: depends on the proxy. SynFlow ≈ 50 ms/FFC, ZiCo ≈ 300 ms/FFC, NWOT ≈ 100 ms/FFC, others in between (with `cells_per_stage=2`, batch size 16). One full FFC = 100 SEM-NAS run takes seconds to a minute on CPU.
* **Online compute, GPU**: typically 5–15× faster than CPU for the gradient-based proxies.

The online backend caches per-encoding scores by default so the duplicate queries that are inherent to RTS replacement do not re-trigger PyTorch. Each unique architecture is still charged exactly one FFC the first time it is evaluated.

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

Includes 8 search-side smoke tests plus 2 online-backend tests (build network → compute every proxy → finite-output check; run SEM-NAS with the online backend at a tiny FFC budget). The tests do not require the precomputed pickles or downloaded data.


---

## License

MIT. See `LICENSE`.
