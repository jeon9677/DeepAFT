#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Survival Data Generator — Correlated Log-normal AFT
====================================================

Generates stratified train / test CSV pairs for simulation studies.

Data-generating process
-----------------------
  X ~ MVN(0, Σ_AR1(ρ)) / 2
  log T = g(X) + ε,   ε ~ N(0, σ)          [default]
                    or ε ~ t(df) / 2         [uncomment in generate_correlated_data]
                    or ε ~ Gumbel(loc, scale) [uncomment in generate_correlated_data]
  C ~ Uniform(0, τ)
  Y = min(T, C, τ),  δ = 1{T ≤ C and T ≤ τ}

Non-linear prognostic index
---------------------------
  g(X) = X₁X₂ − 0.2·X₃³ + sin(X₄·X₅) − 0.5·X₅

Outputs (under ``<outdir>/<setting>/``)
---------------------------------------
  seed_<seed>_train.csv
  seed_<seed>_test.csv

Usage
-----
  python generate_data.py \\
      --p 100 --sigma 0.025 --n 1000 --n_sims 100 \\
      --tau 3.5 --rho 0.3 --train_frac 0.7 \\
      --base_seed 1000 --outdir simul/SimulData
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1000) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Covariance & prognostic index
# ---------------------------------------------------------------------------
def make_ar1_cov(p: int, rho: float = 0.3) -> np.ndarray:
    """AR(1) covariance matrix of size (p, p)."""
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def g_nonlinear(X: np.ndarray) -> np.ndarray:
    """
    Non-linear prognostic index.

    g(X) = X₁X₂ − 0.2·X₃³ + sin(X₄·X₅) − 0.5·X₅

    Columns beyond index 4 receive zero weight (noise variables).
    """
    p = X.shape[1]

    def _col(k: int) -> np.ndarray | float:
        return X[:, k] if k < p else 0.0

    g = (
        _col(0) * _col(1)
        - 0.2 * (_col(2) ** 3)
        + np.sin(_col(3) * _col(4))
        - 0.5 * _col(4)
    )
    if p > 5:
        w = np.zeros(p - 5)          # noise columns contribute nothing
        g = g + ((X[:, 5:] ** 3) @ w)
    return g


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_correlated_data(
    n: int = 1000,
    p: int = 30,
    sigma: float = 0.5,
    tau: float = 7.0,
    rho: float = 0.3,
    t_df: float = 5.0,
    gumbel_loc: float = 0.0,
    gumbel_scale: float = 1.0,
    seed: int = 1000,
) -> tuple[np.ndarray, ...]:
    """
    Generate one dataset from the log-normal AFT model.

    Parameters
    ----------
    n            : sample size
    p            : number of covariates
    sigma        : variance of the Gaussian error term
    tau          : administrative censoring time
    rho          : AR(1) correlation parameter
    t_df         : degrees of freedom for t-distributed error (unused by default)
    gumbel_loc   : location for Gumbel error (unused by default)
    gumbel_scale : scale for Gumbel error (unused by default)
    seed         : random seed

    Returns
    -------
    X          : (n, p) float32 covariates
    y          : (n,)   float32 observed times
    delta      : (n,)   float32 event indicators
    logT       : (n,)   float32 true log(T)
    mu_true    : (n,)   float32 true E[log T | X]
    sigma_true : (n,)   float32 true std of log T (constant = sqrt(sigma))

    Error distribution (select one by uncommenting in the source):
      - Gaussian  : ε ~ N(0, σ)           [default]
      - Student-t : ε ~ t(df) / 2
      - Gumbel    : ε ~ Gumbel(loc, scale)
    """
    set_seed(seed)
    Sigma = make_ar1_cov(p, rho=rho) + 1e-8 * np.eye(p)
    X = (
        np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n).astype(np.float32)
    ) / 2

    gX = g_nonlinear(X)

    # --- error term (choose one) ---
    eps = np.random.normal(0.0, np.sqrt(sigma), size=n)           # Gaussian
    # eps = np.random.standard_t(df=t_df, size=n) / 2             # Student-t
    # eps = np.random.gumbel(loc=gumbel_loc, scale=gumbel_scale, size=n)  # Gumbel

    logT = gX + eps
    T = np.exp(logT)

    C = np.random.uniform(0, tau, size=n)
    y = np.minimum(np.minimum(T, C), tau)
    delta = ((T <= C) & (T <= tau)).astype(np.float32)

    mu_true = gX.astype(np.float32)
    sigma_true = np.full_like(mu_true, fill_value=np.sqrt(sigma), dtype=np.float32)

    return (
        X.astype(np.float32),
        y.astype(np.float32),
        delta,
        logT.astype(np.float32),
        mu_true,
        sigma_true,
    )


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def _stratified_train_test_split(
    df: pd.DataFrame,
    label_col: str = "delta",
    train_frac: float = 0.7,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified binary-label train / test split."""
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
    test_df  = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def save_simulations(
    p: int = 5,
    sigma: float = 0.1,
    n: int = 1000,
    n_sims: int = 5,
    rho: float = 0.3,
    tau: float = 8.5,
    t_df: float = 3.0,
    gumbel_loc: float = 0.0,
    gumbel_scale: float = 0.8,
    base_seed: int = 1000,
    outdir: str = "simul",
    train_frac: float = 0.7,
) -> Path:
    """
    Generate ``n_sims`` datasets and save stratified train / test CSVs.

    The output directory is ``<outdir>/n<n>_p<p>_sigma<sigma>_tau<tau>/``.

    Returns the output directory ``Path``.
    """
    setting_name = f"n{n}_p{p}_sigma{sigma}_tau{tau}"
    out_path = Path(outdir) / setting_name
    out_path.mkdir(parents=True, exist_ok=True)

    for k in range(n_sims):
        seed = base_seed + k
        print(f"[{setting_name}] Simulation {k + 1}/{n_sims}, seed={seed}")

        X, y, delta, logT, mu_true, sigma_true = generate_correlated_data(
            n=n, p=p, sigma=sigma, tau=tau, rho=rho,
            t_df=t_df, gumbel_loc=gumbel_loc, gumbel_scale=gumbel_scale,
            seed=seed,
        )

        df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)])
        df["y"]          = y
        df["delta"]      = delta
        df["logT"]       = logT
        df["mu_true"]    = mu_true
        df["sigma_true"] = sigma_true

        train_df, test_df = _stratified_train_test_split(
            df, label_col="delta", train_frac=train_frac, seed=seed
        )

        cr_overall = 1.0 - df["delta"].mean()
        cr_train   = 1.0 - train_df["delta"].mean()
        cr_test    = 1.0 - test_df["delta"].mean()
        print(
            f"  total n={len(df)} (censoring={cr_overall:.3f}) | "
            f"train n={len(train_df)} (censoring={cr_train:.3f}) | "
            f"test n={len(test_df)} (censoring={cr_test:.3f})"
        )

        train_df.to_csv(out_path / f"seed_{seed}_train.csv", index=False)
        test_df.to_csv(out_path / f"seed_{seed}_test.csv",  index=False)

    print(f"\nSaved {n_sims} train/test pairs under {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate stratified survival simulation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n",            type=int,   default=1000,           help="Sample size per dataset.")
    ap.add_argument("--p",            type=int,   default=100,            help="Number of covariates.")
    ap.add_argument("--sigma",        type=float, default=0.025,          help="Gaussian error variance σ.")
    ap.add_argument("--tau",          type=float, default=3.5,            help="Administrative censoring time τ.")
    ap.add_argument("--rho",          type=float, default=0.3,            help="AR(1) correlation ρ.")
    ap.add_argument("--train_frac",   type=float, default=0.7,            help="Fraction of data used for training.")
    ap.add_argument("--n_sims",       type=int,   default=100,            help="Number of simulation replicates.")
    ap.add_argument("--base_seed",    type=int,   default=1000,           help="Base random seed (seed_k = base_seed + k).")
    ap.add_argument("--outdir",       type=str,   default="simul/SimulData", help="Root output directory.")
    ap.add_argument("--t_df",         type=float, default=3.0,            help="Degrees of freedom for t error (unused by default).")
    ap.add_argument("--gumbel_loc",   type=float, default=0.0,            help="Gumbel error location (unused by default).")
    ap.add_argument("--gumbel_scale", type=float, default=0.8,            help="Gumbel error scale (unused by default).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    save_simulations(
        p=args.p,
        sigma=args.sigma,
        n=args.n,
        n_sims=args.n_sims,
        rho=args.rho,
        tau=args.tau,
        t_df=args.t_df,
        gumbel_loc=args.gumbel_loc,
        gumbel_scale=args.gumbel_scale,
        base_seed=args.base_seed,
        outdir=args.outdir,
        train_frac=args.train_frac,
    )


if __name__ == "__main__":
    main()
