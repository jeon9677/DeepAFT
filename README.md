# Deep Gaussian Process AFT — Simulation Code

Official simulation code for the paper:

> **A Deep Gaussian Process Approach for Survival Prediction under the Accelerated Failure Time Model**

---

## Overview

This repository contains two components:

| Script | Role |
|---|---|
| `generate_data.py` | Generates stratified train / test CSV datasets from a log-normal AFT DGP |
| `deepgp_aft.py` | Trains the DeepGP-AFT model and evaluates it on the generated datasets |

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
pandas>=2.0
tensorflow>=2.13
```

---

## Quickstart

### Step 1 — Generate simulation datasets

```bash
python generate_data.py \
    --p 100 --sigma 0.025 --n 1000 --n_sims 100 \
    --tau 3.5 --rho 0.3 --train_frac 0.7 \
    --base_seed 1000 --outdir simul/SimulData
```

This writes stratified train / test CSV pairs to:

```
simul/SimulData/n1000_p100_sigma0.025_tau3.5/
  seed_1000_train.csv
  seed_1000_test.csv
  seed_1001_train.csv
  ...
```

### Step 2 — Train and evaluate

```bash
python deepgp_aft.py \
    --setting n1000_p100_sigma0.025_tau3.5 \
    --base_dir simul/SimulData \
    --first_seed 1000 --n_seeds 100 \
    --width 128 --depth 3 --dropout 0.2 \
    --lr 1e-3 --batch_size 128 --epochs 500 \
    --mc_passes 30 --patience 20
```

---

## Data-Generating Process

$$X \sim \mathcal{N}(0,\, \Sigma_{\mathrm{AR1}}(\rho))\,/\,2, \qquad \Sigma_{\mathrm{AR1}}(\rho)_{jk} = \rho^{|j-k|}$$

$$\log T_i = g(X_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\,\sigma)$$

$$C_i \sim \mathrm{Uniform}(0,\,\tau), \quad Y_i = \min(T_i, C_i, \tau), \quad \delta_i = \mathbf{1}[T_i \le C_i \text{ and } T_i \le \tau]$$

### Non-linear prognostic index

$$g(X) = X_1 X_2 \;-\; 0.2\,X_3^3 \;+\; \sin(X_4 X_5) \;-\; 0.5\,X_5$$

Columns $X_6, \ldots, X_p$ are pure noise variables (zero weight).

Alternative error distributions (commented out in `generate_data.py`):

| Distribution | Expression |
|---|---|
| Gaussian (default) | $\varepsilon \sim \mathcal{N}(0, \sigma)$ |
| Student-t | $\varepsilon \sim t(\mathrm{df}) / 2$ |
| Gumbel | $\varepsilon \sim \mathrm{Gumbel}(\mathrm{loc}, \mathrm{scale})$ |

---

## Data Splits

```
Full data  (n)
 ├── 70 %  →  train_pool     (stratified on δ)
 │    ├── 5/7  →  train_core   (model optimisation)
 │    └── 2/7  →  val          (early stopping / LR schedule)
 └── 30 %  →  test             (final evaluation only)
```

Final proportions: **50 % / 20 % / 30 %**.

---

## Model

The DeepGP-AFT model uses a fully-connected neural network with tanh activations and MC-Dropout as the mean function of a deep Gaussian process surrogate, trained under the log-normal AFT likelihood.

```
Input (p,)
 └─ [Dense(width, tanh) → Dropout(p=dropout)]  ×  depth
     └─ Dense(2)  →  [μ,  raw_scale]
```

- **μ** : predicted log-median survival time $\hat{\mu}(X)$
- **σ** = softplus(raw\_scale) + 1e-6 : predicted log-normal scale
- At inference, `mc_passes` stochastic forward passes are averaged (MC-Dropout)

### Log-normal AFT negative log-likelihood

$$\ell(\mu, \sigma;\, y, \delta) = -\left[\delta \log f(y;\,\mu,\sigma) + (1-\delta)\log S(y;\,\mu,\sigma)\right]$$

where $f$ and $S$ are the log-normal PDF and survival function, respectively.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `rmse_logT` | RMSE of log(T) on event-only rows |
| `mae_logT` | MAE of log(T) on event-only rows |
| `ipcw_rmse_logT` | IPCW-weighted RMSE of log(T) |
| `rmse_time_median` | RMSE of predicted median T on event-only rows |
| `mae_time_median` | MAE of predicted median T on event-only rows |
| `cindex_median_ipcw` | IPCW C-index (Uno et al.) using predicted median T |

IPCW weights use the Kaplan–Meier estimator of the censoring distribution $\hat{G}(t) = P(C > t)$.

---

## Output Files

All results are written to `<base_dir>/<setting>/`:

| File | Contents |
|---|---|
| `seed_<seed>_train.csv` | Training split (columns: x1…xp, y, delta, logT, mu_true, sigma_true) |
| `seed_<seed>_test.csv` | Test split |
| `metrics_seed_<seed>.csv` | Per-seed evaluation metrics |
| `metrics_summary.csv` | All seeds concatenated |

---

## Script Arguments

### `generate_data.py`

| Argument | Default | Description |
|---|---|---|
| `--n` | 1000 | Sample size |
| `--p` | 100 | Number of covariates |
| `--sigma` | 0.025 | Error variance σ |
| `--tau` | 3.5 | Administrative censoring time τ |
| `--rho` | 0.3 | AR(1) correlation ρ |
| `--train_frac` | 0.7 | Training fraction |
| `--n_sims` | 100 | Number of replicates |
| `--base_seed` | 1000 | Base seed |
| `--outdir` | `simul/SimulData` | Output root directory |

### `deepgp_aft.py`

| Argument | Default | Description |
|---|---|---|
| `--setting` | — | Sub-directory name (required) |
| `--base_dir` | `simul` | Root directory |
| `--first_seed` | 1000 | First seed to evaluate |
| `--n_seeds` | 100 | Number of seeds |
| `--width` | 128 | Hidden layer width |
| `--depth` | 3 | Number of hidden layers |
| `--dropout` | 0.2 | MC-Dropout rate |
| `--lr` | 1e-3 | Initial Adam learning rate |
| `--batch_size` | 128 | Mini-batch size |
| `--epochs` | 500 | Maximum training epochs |
| `--mc_passes` | 30 | MC-Dropout inference passes |
| `--patience` | 20 | Early-stopping patience |

---

## Citation

If you use this code, please cite:

```bibtex
@article{deepgp_aft_2025,
  title   = {A Deep Gaussian Process Approach for Survival Prediction under the Accelerated Failure Time Model},
  year    = {2025}
}
```

---

## License

MIT
