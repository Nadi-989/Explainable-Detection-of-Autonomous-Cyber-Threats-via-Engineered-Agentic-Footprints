# When Attacks Think for Themselves — Reproducibility Code

Code that reproduces every result and figure in the paper
*"When Attacks Think for Themselves: Explainable Detection of Autonomous
Cyber Threats via Engineered Agentic Footprints."*

A single script (`src/main.py`) runs the full pipeline: it loads three real
benchmark datasets, builds the standard **and** agentic-footprint features,
trains five classifiers, evaluates them under **two protocols** (five random
seeds and stratified 5-fold cross-validation) with **MCC** as the primary
metric, and generates all interpretability figures (SHAP, three-method
importance comparison, ROC, MCC bar chart, agentic-rank plot).

## Datasets (real, in `data/`)

| File | Dataset | Attack stage | Source |
|---|---|---|---|
| `nsl_train.csv` | NSL-KDD | Penetration (intrusion) | UCI / CIC (KDDTrain+) |
| `phishing.csv` | Phishing URLs | Initial access | UCI Phishing Websites |
| `spambase.csv` | Spambase | Propagation (spam) | UCI Spambase (HP Labs) |

These are the exact files used in the paper. To re-download:

```bash
curl -L -o data/nsl_train.csv "https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv"
curl -L -o data/phishing.csv  "https://raw.githubusercontent.com/Danish-lakhwani/Phishing-Detection-Using-Machine-Learning/main/Phishing%20dataset.csv"
curl -L -o data/spambase.csv  "https://raw.githubusercontent.com/samujjwaal/Spam-Email-Classifier/master/spambase.csv"
```

## How to run

```bash
pip install -r requirements.txt
python src/main.py
```

Runs end-to-end in ~4–6 minutes. Outputs:

- `results/kfold_results.csv` — full 5-fold metrics (Table 3, 5).
- `results/seed_summary.csv` — 5-seed mean ± std (Table 4, 5).
- `results/per_seed_results.csv` — raw per-seed records.
- `figures/` — 11 PNG figures (Fig. 1–9 in the paper).

### Colab / Jupyter
The script uses a `__file__` fallback, so it also runs in a notebook. Put the
`data/` folder next to `src/` (or edit `BASE`), then:

```python
%run src/main.py
```

## What maps to what in the paper

| Paper element | Code location |
|---|---|
| Eqs. 1–4 (standard features) | `load_nsl / load_phishing / load_spambase` |
| Eqs. 5–10 (agentic footprint) | same loaders, `ag_*` features |
| Section 5.1 (five classifiers) | `build_models` |
| Section 5.2 (5 seeds) | `evaluate_five_seeds` |
| Section 5.2 (5-fold, leakage-safe) | `evaluate_kfold` (scaling inside a `Pipeline`) |
| Eq. 11 (MCC) | `matthews_corrcoef` via `all_metrics` |
| Section 5.3 / 6.3 (SHAP, Gini, permutation) | `figures_for_dataset` |
| Fig. 1 (MCC bar) | `figure_mcc_bar` |
| Fig. 9 (agentic ranks) | `figure_agentic_ranks` |

## Notes

- `MAX_ROWS = 10000` sub-samples NSL-KDD for runtime; raise it for the full set.
- The agentic-footprint features are **proxy indicators**, not ground-truth
  labels of the actor (see the paper's Limitations). They are designed to fire
  more strongly for machine-driven attacks than for manual ones.
- Random seeds are fixed for reproducibility.

## Project layout

```
paper_code/
├── data/           # the three real datasets (CSV)
├── src/
│   └── main.py     # the full reproducibility pipeline
├── results/        # CSV tables (generated)
├── figures/        # PNG figures (generated)
├── requirements.txt
└── README.md
```
