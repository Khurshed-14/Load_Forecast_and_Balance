# Load Forecast and Balance

Replication and experimentation for **GLFN-TC** (Graph Learning + Feature Network with Temporal Convolution): short-term load forecasting on multiple public datasets, with sensitivity analysis and baseline comparisons.

## Repository layout

```
Load_Forecast_and_Balance/
├── data/                    # Canonical datasets
│   ├── at/                  # Austrian load (AT)
│   ├── iso_ne/              # ISO New England
│   ├── sh/                  # Shanghai hub / regional CSVs
│   ├── ncent/               # NCENT regional splits
│   └── other/               # Additional datasets (BD, PL, TN, …)
├── notebooks/
│   ├── load_forecast_*.ipynb    # Primary training notebooks (AT, ISO-NE, SH)
│   ├── load_forecast.ipynb      # Legacy combined notebook
│   ├── eda/                     # Exploratory data analysis
│   ├── archive/                 # Earlier model / graph variants
│   └── sensitivity analysis/    # Hyperparameter sweeps (AT, ISO-NE, SH)
├── Baseline Metrics/        # Baseline model metrics (CSV)
├── external/                # Third-party baselines (e.g. ISO-NE ResNet+)
├── replications/
│   └── bilstm_cnn_bilstm/   # Separate paper replication (BiLSTM / CNN-BiLSTM)
├── dataset_classes.py       # PyTorch `Dataset` implementations
├── models.py                # Model definitions (earlier variants)
├── models_with_temporal_graph.py
├── models_temporal_feature.py
├── helper_functions.py
├── helper_functions_trial.py
└── pyproject.toml
```

## Requirements

- Python **≥ 3.12**
- Dependencies are listed in [`pyproject.toml`](pyproject.toml) (PyTorch, pandas, scikit-learn, TensorBoard, etc.)

### Install

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync --group dev
```

For Jupyter, register the environment kernel after install:

```bash
python -m ipykernel install --user --name load-forecast --display-name "Load Forecast"
```

## Running notebooks

1. **Start Jupyter from the repository root** so imports resolve (`dataset_classes`, `models_with_temporal_graph`, `helper_functions_trial`, etc.).

   ```bash
   cd /path/to/Load_Forecast_and_Balance
   jupyter lab
   ```

2. Open notebooks under `notebooks/` (see table below).

3. Point dataset paths at files under `data/` (for example `data/at/AT Dataset.csv`, `data/iso_ne/selected_data_ISONE.csv`, `data/sh/shanghai.csv`). Some notebooks may still reference older paths from before the folder reorganization; update `csv_path` in those cells if loading fails.

| Notebook | Purpose |
|----------|---------|
| `notebooks/load_forecast_at.ipynb` | Train / evaluate on AT |
| `notebooks/load_forecast_iso_ne.ipynb` | Train / evaluate on ISO-NE |
| `notebooks/load_forecast_sh.ipynb` | Train / evaluate on Shanghai (`shanghai.csv`) |
| `notebooks/sensitivity analysis/` | Hyperparameter sensitivity (per dataset) |
| `notebooks/eda/` | Dataset exploration |
| `notebooks/archive/` | Superseded experiments (reference only) |

## Datasets (`data/`)

| Directory | Description |
|-----------|-------------|
| `data/at/` | `AT Dataset.csv` |
| `data/iso_ne/` | `selected_data_ISONE.csv`, sample file |
| `data/sh/` | Regional load CSVs (e.g. `shanghai.csv`, `beijing.csv`) |
| `data/ncent/` | NCENT and regional splits (`NCENT.csv`, `COAST.csv`, …) |
| `data/other/` | Bangladesh, PL, TN, and other auxiliary sets |

## Python modules

| Module | Role |
|--------|------|
| `dataset_classes.py` | `AT`, `ISO_NE`, `SH_Dataset`, `NCENT_Dataset`, and related loaders |
| `models_with_temporal_graph.py` | Main GLFN-TC variants used by current training notebooks |
| `models_temporal_feature.py` | Hierarchical / spectral feature-clustering variants |
| `models.py` | Earlier monolithic model definitions |
| `helper_functions_trial.py` | Training loop with regularization hooks (`train_model`, `test_model`) |
| `helper_functions.py` | Simpler training / validation helpers |

## Baselines and other work

- **`Baseline Metrics/`** — CSV metrics for baseline models (train/validation and test splits) on AT, ISO-NE, and SH.
- **`replications/bilstm_cnn_bilstm/`** — Independent replication of *Short-Term Aggregated Residential Load Forecasting using BiLSTM and CNN-BiLSTM*; not part of the GLFN-TC pipeline.
- **`external/ResNetPlus_ISONE.py`** — External Keras baseline for ISO-NE (reference only).

## Git ignore policy

Generated artifacts (checkpoints at repo root, TensorBoard `runs/`, sensitivity run folders, virtualenvs, and notebook checkpoints) are listed in [`.gitignore`](.gitignore). Large `.pth` files under `notebooks/sensitivity analysis/` may already be tracked; new root-level `Models/` output is ignored by default.
