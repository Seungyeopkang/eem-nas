# Sample-Efficient Memetic NAS (SEM-NAS)

Reference implementation accompanying the paper

> **Budgeted Fixed-Proxy Search for Zero-Shot NAS on NAS-Bench-201 via Sample-Efficient Memetic NAS** (Electronics, 2026).

SEM-NAS is a single-population memetic procedure for **budgeted fixed-proxy zero-shot NAS** on NAS-Bench-201 TSS. Each run optimizes one fixed zero-cost proxy under a strict fitness function call budget (FFC = 100), and queries only 100 of the 15,625 architectures.

This package contains a clean reproduction of the four primitives described in Section 3 of the paper, the four budgeted fixed-proxy search baselines, and minimal scripts to reproduce the 84-cell main comparison matrix.

---

## Repository layout

```
code/
├── README.md                  this file
├── requirements.txt           numpy only (search uses table lookups, no PyTorch)
├── LICENSE                    MIT
├── sem_nas/
│   ├── encoding.py            NB-201 TSS encoding (length-6 op vector)
│   ├── evaluator.py           FFC-bounded proxy evaluator
│   ├── operators.py           tournament selection, uniform crossover, block mutation
│   ├── local_search.py        1-flip first-improvement LS + per-call FFC view
│   ├── primitives.py          RTS, rank+distance LS targets, entropy-guided mutation
│   ├── sem_nas.py             Algorithm 1 (proposed method)
│   └── baselines.py           4 baselines + 2 forced-edit fairness controls
├── scripts/
│   ├── run_one.py             single (method, dataset, proxy, seed) cell
│   ├── run_main_matrix.py     full 84-cell matrix driver
│   └── load_proxy_pickle.py   helper for nb201_<dataset>.pkl loading
└── tests/
    └── test_sem_nas.py        smoke tests (no pickles needed)
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10 is recommended. The search itself only depends on NumPy because every proxy and test-accuracy value is a precomputed table lookup. No GPU, no PyTorch.

---

## Preparing the data

The release does **not** bundle the precomputed proxy pickles. To reproduce the experiments you need three files:

```
data/nb201_cifar10.pkl
data/nb201_cifar100.pkl
data/nb201_imagenet16_120.pkl
```

Each pickle is a dict::

    {
      "meta": {"benchmark": "NB201", "dataset": "...", "proxies": [...]},
      "test_accuracy": np.ndarray of shape (15625,),
      "proxies": {"zico": ..., "nwot": ..., "synflow": ..., "jacov": ...,
                  "snip": ..., "grad_norm": ..., "fisher": ...},
    }

The pickles can be regenerated from the public NAS-Bench-201 release and the standard zero-cost proxy implementations. See the parent paper repository for the exact build script.

The default lookup directory is `<repo>/code/data/`. Override it with the `SEM_NAS_DATA_DIR` environment variable or with the `--data_dir` CLI flag.

---

## Quick start

Run one cell of the main matrix (SEM-NAS at FFC = 100):

```bash
python -m scripts.run_one --method sem_nas --dataset cifar10 --proxy zico --seed 0
```

Run a baseline:

```bash
python -m scripts.run_one --method aging_evolution --dataset cifar100 --proxy nwot --seed 7
```

Run the full 84-cell matrix (5 methods × 7 proxies × 3 datasets, 100 seeds each, FFC = 100, parallelized across CPU workers):

```bash
python -m scripts.run_main_matrix --workers 8 --seeds 100
```

Subset of methods or proxies:

```bash
python -m scripts.run_main_matrix --workers 8 --seeds 100 \
    --methods sem_nas,generic_ga --proxies zico,nwot
```

Result pickles land under `code/results/main_matrix/` and contain the running best per FFC, the queried-architecture history, and the returned architecture's encoding / proxy score / test accuracy.

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

These values are the operating point used to produce the 82 W / 2 T / 0 L Holm-corrected main result on the 84-cell family.

---

## API

```python
from sem_nas.evaluator import FitnessEvaluator
from sem_nas.sem_nas import run as run_sem_nas

evaluator = FitnessEvaluator(proxy_scores, test_accuracy, max_evals=100)
np.random.seed(0)  # determinism
best_arch, final_population, final_fitness = run_sem_nas(evaluator)
```

The four baselines and two forced-edit controls share the same evaluator interface:

```python
from sem_nas.baselines import BASELINES
best = BASELINES["aging_evolution"](evaluator)
```

Available baseline keys:

* `random` — uniform random sampling
* `aging_evolution` — Real et al. (2019)
* `simple_mutation` — steady-state mutation-only
* `generic_ga` — uniform crossover + tournament + block mutation + elitism
* `simple_mutation_forced` — Simple Mutation with at-least-one-edit fallback (paper §4 fairness control)
* `generic_ga_forced` — Generic GA with the same forced-edit fallback

---

## Reproducing paper figures and tables

The main matrix pickles produced by `scripts/run_main_matrix.py` are the ground truth for:

* **Table 2** (Holm-corrected Win/Tie/Lose against the four baselines)
* **Table 3** (per-cell mean ± std proxy score)
* **Table 5** (forced-edit substitution headline 77/7/0)
* **Figures 4–5** (FFC budget curves on ZiCo and NWOT)

The aggregate paired Wilcoxon + Holm correction + Cliff's δ analysis is intentionally not bundled here, because that part is closer to one-off statistical post-processing than to the search method itself. The bundled `best_score_per_ffc` arrays are sufficient to reproduce both the headline table and the budget-sweep figures.

---

## Tests

```bash
python -m pytest tests/ -q
```

The tests do not require the NB-201 pickles; they use a synthetic 15,625-entry score array.

---

## Citation

```bibtex
@article{seo2026semnas,
  title={Budgeted Fixed-Proxy Search for Zero-Shot NAS on NAS-Bench-201 via Sample-Efficient Memetic NAS},
  author={Seo, Wangduk},
  journal={Electronics},
  year={2026},
  publisher={MDPI}
}
```

---

## License

MIT. See `LICENSE`.
