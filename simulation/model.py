#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepGP-like Log-normal AFT with MC-Dropout
===========================================

Stratified 50/20/30 split (train / val / test) via two-step stratification:
  1. 70 % stratified  →  train_pool  vs  test
  2. train_pool  →  stratified  5/7  train_core  and  2/7  val
     (final proportions: 50 % / 20 % / 30 % of total)

The model is trained on train_core, validated on val, and evaluated on test.

Outputs (under ``simul/<setting>/``)
-------------------------------------
  seed_<seed>_train.csv       raw training split
  seed_<seed>_test.csv        raw test split
  metrics_seed_<seed>.csv     per-seed evaluation metrics
  metrics_summary.csv         aggregated metrics across all seeds

Metrics
-------
  rmse_logT            RMSE of log(T) on event-only rows
  mae_logT             MAE  of log(T) on event-only rows
  ipcw_rmse_logT       IPCW-weighted RMSE of log(T)
  rmse_time_median     RMSE of median survival time T on event-only rows
  mae_time_median      MAE  of median survival time T on event-only rows
  cindex_median_ipcw   IPCW C-index using predicted median T
  runtime_seconds      wall-clock training time per seed
"""

from __future__ import annotations

import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI = tf.constant(np.pi, dtype=tf.float32)
SQRT2 = tf.constant(np.sqrt(2.0), dtype=tf.float32)
LOG_HALF = tf.math.log(tf.constant(0.5, dtype=tf.float32))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1000) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_ar1_cov(p: int, rho: float = 0.3) -> np.ndarray:
    """AR(1) covariance matrix of size (p, p)."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def g_nonlinear(X: np.ndarray) -> np.ndarray:
    """Non-linear prognostic index used to generate log(T)."""
    p = X.shape[1]

    def _col(k):
        return X[:, k] if k < p else 0.0

    g = (
        _col(0) * _col(1)
        + 0.5 * (_col(2) ** 3)
        + _col(3) * _col(4)
        - 0.8 * _col(5)
    )
    if p > 6:
        w = np.linspace(0.5, 0.1, num=p - 6)
        g = g + (X[:, 6:] @ w)
    return g


def generate_correlated_data(
    n: int = 1000,
    p: int = 30,
    sigma: float = 0.5,
    tau: float = 7.0,
    rho: float = 0.3,
    seed: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate log-normal AFT survival data with AR(1) covariates.

    Parameters
    ----------
    n     : sample size
    p     : number of covariates
    sigma : variance of the log-normal error  (epsilon ~ N(0, sigma))
    tau   : administrative censoring time
    rho   : AR(1) correlation parameter
    seed  : random seed

    Returns
    -------
    X          : (n, p) float32 covariates
    y          : (n,)   float32 observed times  min(T, C, tau)
    delta      : (n,)   float32 event indicators
    mu_true    : (n,)   float32 true log(T) means
    sigma_true : (n,)   float32 true log(T) std (constant = sqrt(sigma))
    """
    set_seed(seed)
    Sigma = make_ar1_cov(p, rho=rho) + 1e-8 * np.eye(p)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n).astype(
        np.float32
    )

    gX = g_nonlinear(X)
    eps = np.random.normal(0.0, np.sqrt(sigma), size=n)
    logT = gX + eps
    T = np.exp(logT)

    C = np.random.uniform(0, tau, size=n)
    y = np.minimum(np.minimum(T, C), tau)
    delta = ((T <= C) & (T <= tau)).astype(np.float32)

    mu_true = gX.astype(np.float32)
    sigma_true = np.full_like(mu_true, fill_value=np.sqrt(sigma), dtype=np.float32)

    return X.astype(np.float32), y.astype(np.float32), delta, mu_true, sigma_true


# ---------------------------------------------------------------------------
# Stratified splitting utilities
# ---------------------------------------------------------------------------
def _stratified_train_test_split(
    df: pd.DataFrame,
    label_col: str = "delta",
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified binary-label train/test split."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for val in sorted(df[label_col].unique()):
        block = df[df[label_col] == val]
        idx = block.index.to_numpy()
        rng.shuffle(idx)
        n_train = int(np.floor(train_frac * len(idx)))
        train_parts.append(block.loc[idx[:n_train]])
        test_parts.append(block.loc[idx[n_train:]])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, test_df


def _stratified_split(
    df: pd.DataFrame,
    label_col: str = "delta",
    frac: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (A, B) where A receives ``frac`` proportion per class."""
    rng = np.random.default_rng(seed)
    A_parts, B_parts = [], []
    for val in sorted(df[label_col].unique()):
        block = df[df[label_col] == val]
        idx = block.index.to_numpy()
        rng.shuffle(idx)
        n_A = int(np.floor(frac * len(idx)))
        A_parts.append(block.loc[idx[:n_A]])
        B_parts.append(block.loc[idx[n_A:]])
    A = pd.concat(A_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    B = pd.concat(B_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return A, B


# ---------------------------------------------------------------------------
# IPCW helpers (Kaplan–Meier estimate of the censoring distribution)
# ---------------------------------------------------------------------------
def _km_survival_of_censoring(y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Kaplan–Meier estimate of G(t) = P(C > t) evaluated at each observed time."""
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    c = 1 - delta  # censoring indicator
    uniq = np.unique(y)
    r = np.array([(y >= t).sum() for t in uniq], dtype=float)
    d = np.array([c[y == t].sum() for t in uniq], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = 1.0 - np.divide(d, r, out=np.zeros_like(d), where=(r > 0))
    G_at_times = np.cumprod(step, dtype=float)
    idx = np.searchsorted(uniq, y, side="right") - 1
    G_hat = np.where(idx >= 0, G_at_times[idx], 1.0)
    return np.clip(G_hat, 1e-12, 1.0)


def c_index_ipcw_rstyle(
    y: np.ndarray,
    delta: np.ndarray,
    t_pred: np.ndarray,
    censor_prob: np.ndarray | None = None,
) -> float:
    """IPCW C-index (Uno et al. style)."""
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    t_pred = np.asarray(t_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(delta) & np.isfinite(t_pred) & (y > 0)
    y, delta, t_pred = y[ok], delta[ok], t_pred[ok]
    n = len(y)
    if n < 2:
        return np.nan
    G_hat = (
        _km_survival_of_censoring(y, delta)
        if censor_prob is None
        else np.clip(np.asarray(censor_prob, dtype=float), 1e-12, 1.0)
    )
    num, den = 0.0, 0.0
    idx_all = np.arange(n)
    for i in range(n):
        if delta[i] != 1:
            continue
        w = delta[i] / (G_hat[i] ** 2)
        mask = (idx_all != i) & (y[i] < y)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        den += w * cnt
        num += w * int((mask & (t_pred[i] < t_pred)).sum())
    return float(num / den) if den > 0 else np.nan


def ipcw_rmse_logT(
    y: np.ndarray,
    delta: np.ndarray,
    mu_hat: np.ndarray,
    censor_prob: np.ndarray | None = None,
) -> float:
    """IPCW-weighted RMSE of log(T).  w_i = delta_i / G_hat(y_i)."""
    y = np.asarray(y, dtype=float)
    delta = np.asarray(delta, dtype=int)
    mu_hat = np.asarray(mu_hat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(delta) & np.isfinite(mu_hat) & (y > 0)
    y, delta, mu_hat = y[ok], delta[ok], mu_hat[ok]
    G_hat = (
        _km_survival_of_censoring(y, delta)
        if censor_prob is None
        else np.clip(np.asarray(censor_prob, dtype=float), 1e-12, 1.0)
    )
    w = delta / G_hat
    num = np.sum(w * (np.log(y) - mu_hat) ** 2)
    den = np.sum(w)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


# ---------------------------------------------------------------------------
# Model architecture & loss
# ---------------------------------------------------------------------------
def build_deepgp_aft(
    input_dim: int = 30,
    width: int = 128,
    depth: int = 3,
    dropout: float = 0.2,
) -> Model:
    """
    Deep-GP-like AFT network with MC-Dropout.

    Output head has two units: [mu, raw_scale].
    sigma = softplus(raw_scale) + 1e-6 ensures positivity.
    """
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for d in range(depth):
        x = L.Dense(width, activation="tanh", name=f"dense_{d}")(x)
        x = L.Dropout(dropout, name=f"drop_{d}")(x)
    out = L.Dense(2, activation=None, name="head_mu_raw")(x)
    return Model(inp, out, name="DeepGP_AFT")


def aft_lognormal_nll(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Log-normal AFT negative log-likelihood.

    y_true columns: [observed_time, event_indicator]
    y_pred columns: [mu, raw_scale]
    """
    y = y_true[:, 0]
    delta = y_true[:, 1]
    mu = y_pred[:, 0]
    sigma = tf.nn.softplus(y_pred[:, 1]) + 1e-6

    y = tf.clip_by_value(y, 1e-30, 1e30)
    logy = tf.math.log(y)
    z = (logy - mu) / sigma

    logf = (
        -tf.math.log(y)
        - tf.math.log(sigma)
        - 0.5 * tf.math.log(2.0 * PI)
        - 0.5 * tf.square(z)
    )
    erfc_val = tf.clip_by_value(tf.math.erfc(z / SQRT2), 1e-45, 1.0)
    logS = LOG_HALF + tf.math.log(erfc_val)

    nll = -(delta * logf + (1.0 - delta) * logS)
    return tf.reduce_mean(nll)


def predict_mu_sigma(
    model: Model,
    X: np.ndarray,
    mc_passes: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo Dropout inference.

    Returns
    -------
    mu_mean    : (N,) mean predicted log(T) across MC samples
    sigma_mean : (N,) mean predicted scale across MC samples
    """
    outs = []
    for _ in range(mc_passes):
        raw = model(X, training=True).numpy()
        mu = raw[:, 0]
        sigma = np.log1p(np.exp(raw[:, 1])) + 1e-6  # softplus
        outs.append(np.stack([mu, sigma], axis=1))
    outs = np.stack(outs, axis=0)  # (M, N, 2)
    return outs[:, :, 0].mean(axis=0), outs[:, :, 1].mean(axis=0)


# ---------------------------------------------------------------------------
# Data generation & persistence
# ---------------------------------------------------------------------------
def save_simulations(
    p: int = 30,
    sigma: float = 0.5,
    n: int = 1000,
    n_sims: int = 5,
    rho: float = 0.3,
    tau: float = 10.0,
    base_seed: int = 1000,
    outdir: str = "simul",
    train_frac: float = 0.7,
) -> Path:
    """
    Generate ``n_sims`` datasets and save train / test CSVs.

    Returns the output directory ``Path``.
    """
    setting_name = f"p{p}_sigma{sigma}_tau{tau}"
    path = Path(outdir) / setting_name
    path.mkdir(parents=True, exist_ok=True)

    for k in range(n_sims):
        seed = base_seed + k
        print(f"[{setting_name}] Simulation {k + 1}/{n_sims}, seed={seed}")
        X, y, delta, mu_true, sigma_true = generate_correlated_data(
            n=n, p=p, sigma=sigma, tau=tau, rho=rho, seed=seed
        )

        df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)])
        df["y"] = y
        df["delta"] = delta
        df["mu_true"] = mu_true
        df["sigma_true"] = sigma_true

        train_df, test_df = _stratified_train_test_split(
            df, label_col="delta", train_frac=train_frac, seed=seed
        )

        train_df.to_csv(path / f"seed_{seed}_train.csv", index=False)
        test_df.to_csv(path / f"seed_{seed}_test.csv", index=False)

        cr_overall = 1.0 - df["delta"].mean()
        cr_train = 1.0 - train_df["delta"].mean()
        cr_test = 1.0 - test_df["delta"].mean()
        print(
            f"  total n={len(df)} (censoring={cr_overall:.3f}) | "
            f"train n={len(train_df)} (censoring={cr_train:.3f}) | "
            f"test n={len(test_df)} (censoring={cr_test:.3f})"
        )

    print(f"Saved {n_sims} train/test pairs under {path}")
    return path


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def _df_to_xy(df: pd.DataFrame, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y_target) arrays from a dataframe."""
    X = df[[f"x{j + 1}" for j in range(p)]].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    d = df["delta"].to_numpy(dtype=np.float32)
    y_target = np.stack([y, d], axis=1).astype(np.float32)
    return X, y_target


def train_and_eval_for_setting(
    setting_path: Path,
    p: int,
    seeds: list[int] | None = None,
    model_width: int = 128,
    model_depth: int = 3,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 500,
    mc_passes: int = 30,
    patience: int = 20,
) -> pd.DataFrame:
    """
    Train and evaluate the DeepGP-AFT model for every available seed.

    If ``seeds`` is None, all seeds found in ``setting_path`` are used.

    Returns a DataFrame of per-seed metrics.
    """
    # discover seeds
    all_train_files = sorted(setting_path.glob("seed_*_train.csv"))
    if seeds is not None:
        seed_set = set(seeds)
        all_train_files = [
            f for f in all_train_files
            if int(f.stem.split("_")[1]) in seed_set
        ]

    metrics_rows = []

    for train_csv in all_train_files:
        seed = int(train_csv.stem.split("_")[1])
        test_csv = setting_path / f"seed_{seed}_test.csv"
        if not test_csv.exists():
            print(f"  [seed={seed}] test file not found – skipping.")
            continue

        # ------------------------------------------------------------------
        # Split: train_pool  →  val (2/7)  +  train_core (5/7)
        # ------------------------------------------------------------------
        train_pool = pd.read_csv(train_csv)
        val_df, train_core_df = _stratified_split(
            train_pool, label_col="delta", frac=2.0 / 7.0, seed=seed
        )
        test_df = pd.read_csv(test_csv)

        X_tr, y_tr = _df_to_xy(train_core_df, p)
        X_va, y_va = _df_to_xy(val_df, p)
        X_te, y_te = _df_to_xy(test_df, p)

        # ------------------------------------------------------------------
        # Build & train
        # ------------------------------------------------------------------
        tf.keras.backend.clear_session()
        model = build_deepgp_aft(
            input_dim=p, width=model_width, depth=model_depth, dropout=dropout
        )
        model.compile(optimizer=Adam(lr), loss=aft_lognormal_nll)

        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
        ]

        t0 = time.time()
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        runtime = time.time() - t0

        # ------------------------------------------------------------------
        # Predict & evaluate
        # ------------------------------------------------------------------
        mu_hat, _ = predict_mu_sigma(model, X_te, mc_passes=mc_passes)
        y_test = y_te[:, 0]
        d_test = y_te[:, 1]
        medT_hat = np.exp(mu_hat)

        def _rmse(a, b):
            return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

        def _mae(a, b):
            return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

        evt = d_test == 1
        if evt.sum() > 0:
            rmse_logT = _rmse(np.log(y_test[evt]), mu_hat[evt])
            mae_logT = _mae(np.log(y_test[evt]), mu_hat[evt])
            rmse_time_med = _rmse(y_test[evt], medT_hat[evt])
            mae_time_med = _mae(y_test[evt], medT_hat[evt])
        else:
            rmse_logT = mae_logT = rmse_time_med = mae_time_med = np.nan

        ipcw_rmse = ipcw_rmse_logT(y_test, d_test, mu_hat)
        cindex_ipcw = c_index_ipcw_rstyle(y_test, d_test, medT_hat)

        row = dict(
            seed=seed,
            n_test=len(y_test),
            n_events=int(evt.sum()),
            rmse_logT=rmse_logT,
            mae_logT=mae_logT,
            ipcw_rmse_logT=ipcw_rmse,
            rmse_time_median=rmse_time_med,
            mae_time_median=mae_time_med,
            cindex_median_ipcw=cindex_ipcw,
            runtime_seconds=runtime,
        )
        pd.DataFrame([row]).to_csv(setting_path / f"metrics_seed_{seed}.csv", index=False)
        metrics_rows.append(row)

        print(
            f"[{setting_path.name}] seed={seed} | "
            f"rmse_logT={rmse_logT:.3f}  ipcw_rmse_logT={ipcw_rmse:.3f}  "
            f"cindex_ipcw={cindex_ipcw:.3f}  ({runtime:.1f}s)"
        )

    if not metrics_rows:
        print("No metrics computed – no valid seeds found.")
        return pd.DataFrame()

    df = pd.DataFrame(metrics_rows)
    df.to_csv(setting_path / "metrics_summary.csv", index=False)

    agg_cols = [
        "rmse_logT", "ipcw_rmse_logT", "rmse_time_median",
        "cindex_median_ipcw", "runtime_seconds",
    ]
    print("\nAggregate (mean ± std):")
    print(df[agg_cols].agg(["mean", "std"]).to_string())
    return df


def infer_p_from_setting(setting_path: Path) -> int:
    """Infer number of covariates from an arbitrary train CSV in ``setting_path``."""
    any_train = next(setting_path.glob("seed_*_train.csv"))
    cols = pd.read_csv(any_train, nrows=1).columns.tolist()
    return sum(1 for c in cols if c.startswith("x"))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train & evaluate DeepGP-AFT on pre-generated survival data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--setting",
        type=str,
        required=True,
        help="Sub-directory name under --base_dir, e.g. p30_sigma0.5_tau10.0",
    )
    ap.add_argument("--base_dir", type=str, default="simul", help="Root directory for simulations.")
    ap.add_argument("--first_seed", type=int, default=1000, help="First seed index.")
    ap.add_argument("--n_seeds", type=int, default=100, help="Number of consecutive seeds to evaluate.")
    ap.add_argument("--width", type=int, default=128, help="Hidden layer width.")
    ap.add_argument("--depth", type=int, default=3, help="Number of hidden layers.")
    ap.add_argument("--dropout", type=float, default=0.2, help="MC-Dropout rate.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial Adam learning rate.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.")
    ap.add_argument("--mc_passes", type=int, default=30, help="MC-Dropout inference passes.")
    ap.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    setting_path = Path(args.base_dir) / args.setting
    if not setting_path.exists():
        raise FileNotFoundError(f"Setting directory not found: {setting_path}")

    p = infer_p_from_setting(setting_path)
    print(f"Inferred p={p} covariates from {setting_path}")

    candidate_seeds = range(args.first_seed, args.first_seed + args.n_seeds)
    seeds = [
        s for s in candidate_seeds
        if (setting_path / f"seed_{s}_train.csv").exists()
        and (setting_path / f"seed_{s}_test.csv").exists()
    ]
    if not seeds:
        raise FileNotFoundError(
            f"No seed_*_train/test.csv files found in {setting_path} "
            f"for seeds {args.first_seed}..{args.first_seed + args.n_seeds - 1}."
        )
    print(f"Found {len(seeds)} seeds: {seeds[0]} … {seeds[-1]}")

    train_and_eval_for_setting(
        setting_path=setting_path,
        p=p,
        seeds=seeds,
        model_width=args.width,
        model_depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        mc_passes=args.mc_passes,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
