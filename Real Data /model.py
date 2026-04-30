#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Output CNN with MC-Dropout for Malaria Data
==================================================

Two model architectures are trained and compared on a real malaria dataset:
  - **Ours**  : deeper per-head architecture with head-level dropout
  - **Old-3** : classic flat-merge architecture (baseline)

Each model predicts three outcomes simultaneously:
  y1 (binary)      → Binary cross-entropy  + sigmoid output
  y2 (count)       → Poisson deviance      + exponential output
  y3 (continuous)  → MSE                   + linear output (z-standardised during training)

Loss weights are calibrated automatically from a mini-batch before training.
Uncertainty is estimated via Monte-Carlo Dropout (N_MC forward passes at inference).

Outputs (under ``Data/malaria/``)
----------------------------------
  y1_binary_prob_mean_ours_update3.ny
  y2_count_lambda_mean_ours_update3.ny
  y3_continuous_pred_mean_ours_update3.ny
  y1_binary_prob_mean_old3_update3.ny
  y2_count_lambda_mean_old3_update3.ny
  y3_continuous_pred_mean_old3_update3.ny
  y{1,2,3}_*_true_update3.ny
  metrics_comparison_update3.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    mean_poisson_deviance,
    r2_score,
    roc_curve,
)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
    concatenate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("Data/malaria")
SEQ_LEN = 239        # length of the 1-D spectral / sequence input
N_MC = 100           # MC-Dropout inference passes
BATCH_SIZE = 64
EPOCHS_OURS = 300
EPOCHS_OLD3 = 1000
PATIENCE = 30
LR = 1e-3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_np(arr: np.ndarray, path: os.PathLike) -> None:
    """Save a numpy array to disk."""
    with open(path, "wb") as f:
        np.save(f, arr)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.reshape(-1) - y_pred.reshape(-1)) ** 2)))


def brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((y_true.reshape(-1) - p_hat.reshape(-1)) ** 2))


def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure input array has shape (N, SEQ_LEN, 1)."""
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == SEQ_LEN:
        return x[..., None]
    if x.ndim == 3 and x.shape[1] == SEQ_LEN and x.shape[2] == 1:
        return x
    raise ValueError(
        f"Unexpected X shape {x.shape}; expected (N, {SEQ_LEN}) or (N, {SEQ_LEN}, 1)."
    )


def find_best_threshold_youden(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Youden-J optimal classification threshold from the ROC curve."""
    fpr, tpr, thr = roc_curve(y_true.reshape(-1), p_hat.reshape(-1))
    best = thr[np.argmax(tpr - fpr)]
    return float(np.clip(best, 1e-6, 1 - 1e-6))


# ---------------------------------------------------------------------------
# MC-Dropout layer helper
# ---------------------------------------------------------------------------
def mc_dropout(x: Any, rate: float = 0.25, training: bool = True) -> Any:
    """Dropout applied with ``training=True`` even at inference (MC-Dropout)."""
    return Dropout(rate)(x, training=training)


# ---------------------------------------------------------------------------
# Automatic loss-weight calibration
# ---------------------------------------------------------------------------
def compute_initial_loss_weights(
    model: Model,
    X_inputs: list[np.ndarray],
    y_dict: dict[str, np.ndarray],
    frac: float = 0.25,
) -> dict[str, float]:
    """
    Estimate inverse-loss weights from a random mini-batch so that all
    task losses start at a comparable scale.

    Parameters
    ----------
    model    : compiled Keras model
    X_inputs : list of input arrays  [X_seq, X_cov]
    y_dict   : dict mapping output name → target array
    frac     : fraction of training data to sample for calibration

    Returns
    -------
    weights  : dict of loss weights normalised so their mean equals 1
    """
    n = X_inputs[0].shape[0]
    m = max(32, int(n * frac))
    idx = np.random.choice(n, m, replace=False)

    batch = [X_inputs[0][idx], X_inputs[1][idx]]
    yb = {k: v[idx] for k, v in y_dict.items()}

    # deterministic forward pass for calibration
    p_bin, lam_cnt, mu_reg = model(batch, training=False)

    L = np.array([
        tf.keras.losses.binary_crossentropy(yb["y1_binary"], p_bin).numpy().mean(),
        tf.keras.losses.poisson(yb["y2_count"], lam_cnt).numpy().mean(),
        tf.keras.losses.mse(yb["y3_continuous"], mu_reg).numpy().mean(),
    ])
    inv = 1.0 / (L + 1e-8)
    inv = inv * (inv.size / inv.mean())  # normalise so mean == 1

    weights = {
        "y1_binary":     float(inv[0]),
        "y2_count":      float(inv[1]),
        "y3_continuous": float(inv[2]),
    }
    print(
        f"[loss_weights] y1={weights['y1_binary']:.3f}  "
        f"y2={weights['y2_count']:.3f}  "
        f"y3={weights['y3_continuous']:.3f}"
    )
    return weights


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
def build_model_ours(mc: bool = True) -> Model:
    """
    Our proposed architecture.

    Shared CNN backbone → three independent heads, each with a dedicated
    dense layer, head-level dropout, and covariate concatenation.
    """
    X_seq = Input(shape=(SEQ_LEN, 1), name="X_phi")
    cov   = Input(shape=(1,),         name="covariate")

    # --- shared backbone ---
    x = Conv1D(32, 3, activation="tanh")(X_seq)
    x = mc_dropout(x, rate=0.25, training=mc)
    x = MaxPooling1D()(x)

    x = Conv1D(64, 3, activation="tanh")(x)
    x = mc_dropout(x, rate=0.25, training=mc)
    x = MaxPooling1D()(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = mc_dropout(x, rate=0.25, training=mc)

    # --- per-task heads ---
    def _head(trunk, activation: str, name: str) -> Any:
        h = Dense(8, activation="relu")(trunk)
        h = mc_dropout(h, rate=0.10, training=mc)
        h = concatenate([h, cov])
        return Dense(1, activation=activation, name=name)(h)

    y1 = _head(x, "sigmoid",     "y1_binary")
    y2 = _head(x, "exponential", "y2_count")       # ensures λ > 0
    y3 = _head(x, "linear",      "y3_continuous")

    return Model(inputs=[X_seq, cov], outputs=[y1, y2, y3], name="model_ours")


def build_model_old3(mc: bool = True) -> Model:
    """
    Baseline architecture.

    Shared CNN backbone → single merged representation with covariate →
    three output heads (no per-head sub-networks).
    """
    X_seq = Input(shape=(SEQ_LEN, 1), name="X_phi")
    cov   = Input(shape=(1,),         name="cov")

    x = Conv1D(32, 3, activation="tanh")(X_seq)
    x = mc_dropout(x, rate=0.25, training=mc)
    x = MaxPooling1D()(x)

    x = Conv1D(64, 3, activation="tanh")(x)
    x = mc_dropout(x, rate=0.25, training=mc)
    x = MaxPooling1D()(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = mc_dropout(x, rate=0.25, training=mc)
    x = Dense(16, activation="linear")(x)

    merged = concatenate([x, cov], name="merged")

    y1 = Dense(1, activation="sigmoid",     name="y1_binary")(merged)
    y2 = Dense(1, activation="exponential", name="y2_count")(merged)
    y3 = Dense(1, activation="linear",      name="y3_continuous")(merged)

    return Model(inputs=[X_seq, cov], outputs=[y1, y2, y3], name="model_old3")


def compile_model(model: Model, lr: float, loss_weights: dict | None = None) -> None:
    """Compile a multi-output model with task-specific losses and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            "y1_binary":     "binary_crossentropy",
            "y2_count":      tf.keras.losses.Poisson(),
            "y3_continuous": "mse",
        },
        loss_weights=loss_weights,
        metrics={
            "y1_binary": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="auprc", curve="PR"),
            ],
            "y2_count":      ["mse"],
            "y3_continuous": ["mse"],
        },
    )


def _make_callbacks() -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_y1_binary_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def fit_model(
    model: Model,
    X_train: list[np.ndarray],
    y_train: dict[str, np.ndarray],
    X_val: list[np.ndarray],
    y_val: dict[str, np.ndarray],
    epochs: int,
) -> tf.keras.callbacks.History:
    """
    1. Temporarily compile to measure initial loss scales.
    2. Re-compile with calibrated loss weights.
    3. Train with early stopping + LR schedule.
    """
    # Step 1: calibrate weights
    compile_model(model, lr=LR, loss_weights=None)
    init_w = compute_initial_loss_weights(model, X_train, y_train)

    # Step 2: re-compile with weights
    compile_model(model, lr=LR, loss_weights=init_w)

    # Step 3: train
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=_make_callbacks(),
        verbose=1,
    )


# ---------------------------------------------------------------------------
# MC-Dropout inference
# ---------------------------------------------------------------------------
def mc_predict(
    model: Model,
    X: list[np.ndarray],
    n_passes: int = N_MC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ``n_passes`` stochastic forward passes with training=True.

    Returns
    -------
    y1_mean : (N,) mean predicted probability for the binary outcome
    y2_mean : (N,) mean predicted Poisson rate for the count outcome
    y3_mean : (N,) mean predicted value for the continuous outcome (z-scale)
    """
    y1_mc, y2_mc, y3_mc = [], [], []
    for _ in range(n_passes):
        p1, p2, p3 = model(X, training=True)
        y1_mc.append(p1.numpy().reshape(-1, 1))
        y2_mc.append(p2.numpy().reshape(-1, 1))
        y3_mc.append(p3.numpy().reshape(-1, 1))

    return (
        np.hstack(y1_mc).mean(axis=1),
        np.hstack(y2_mc).mean(axis=1),
        np.hstack(y3_mc).mean(axis=1),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    label: str,
    y1_true: np.ndarray,
    y2_true: np.ndarray,
    y3_true: np.ndarray,
    y1_prob: np.ndarray,
    y2_pred: np.ndarray,
    y3_pred: np.ndarray,
) -> dict[str, float]:
    """Compute and print all metrics for one model."""
    thr = find_best_threshold_youden(y1_true, y1_prob)

    y1_acc_05 = accuracy_score(y1_true.reshape(-1), (y1_prob >= 0.5).astype(int))
    y1_acc_bt = accuracy_score(y1_true.reshape(-1), (y1_prob >= thr).astype(int))
    y1_brier  = brier_score(y1_true, y1_prob)

    y2_rmse = rmse(y2_true, y2_pred)
    y2_mpd  = float(mean_poisson_deviance(y2_true.reshape(-1), y2_pred.reshape(-1)))

    y3_rmse = rmse(y3_true, y3_pred)
    y3_r2   = float(r2_score(y3_true.reshape(-1), y3_pred.reshape(-1)))

    print(f"\n=== {label} ===")
    print(
        f"y1  ACC@0.5={y1_acc_05:.4f}  ACC@Youden({thr:.3f})={y1_acc_bt:.4f}  "
        f"Brier={y1_brier:.4f}"
    )
    print(f"y2  RMSE={y2_rmse:.4f}  MPD={y2_mpd:.4f}")
    print(f"y3  RMSE={y3_rmse:.4f}  R²={y3_r2:.4f}")

    return dict(
        y1_acc_05=y1_acc_05,
        y1_acc_best=y1_acc_bt,
        y1_brier=y1_brier,
        y2_rmse=y2_rmse,
        y2_mpd=y2_mpd,
        y3_rmse=y3_rmse,
        y3_r2=y3_r2,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir: Path) -> dict[str, np.ndarray]:
    """
    Load all pre-processed numpy arrays for training and validation.

    Expected files (``update3`` suffix):
      X_train_basis_update3.ny   X_cv_basis_update3.ny
      cov_data_update3.ny        cov_cv_update3.ny
      Zmat_data_{bin,count,cont}_y1_update3.ny
      Zmat_cv_{bin,count,cont}_y1_update3.ny
    """
    def _load(fname: str) -> np.ndarray:
        return np.load(data_dir / fname, allow_pickle=False)

    return {
        "X_train": ensure_3d(_load("X_train_basis_update3.ny")),
        "X_val":   ensure_3d(_load("X_cv_basis_update3.ny")),
        "cov_tr":  _load("cov_data_update3.ny").reshape(-1, 1),
        "cov_va":  _load("cov_cv_update3.ny").reshape(-1, 1),
        "y1_tr":   (_load("Zmat_data_bin_y1_update3.ny")   >= 0.5).astype(np.float32).reshape(-1, 1),
        "y1_va":   (_load("Zmat_cv_bin_y1_update3.ny")     >= 0.5).astype(np.float32).reshape(-1, 1),
        "y2_tr":   _load("Zmat_data_count_y1_update3.ny").astype(np.float32).reshape(-1, 1),
        "y2_va":   _load("Zmat_cv_count_y1_update3.ny").astype(np.float32).reshape(-1, 1),
        "y3_tr":   _load("Zmat_data_cont_y1_update3.ny").astype(np.float32).reshape(-1, 1),
        "y3_va":   _load("Zmat_cv_cont_y1_update3.ny").astype(np.float32).reshape(-1, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    tf.get_logger().setLevel("ERROR")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {DATA_DIR} …")
    d = load_data(DATA_DIR)

    # z-standardise y3 (continuous target); restore at evaluation
    y3_mean = float(d["y3_tr"].mean())
    y3_std  = float(d["y3_tr"].std() + 1e-8)
    y3_tr_z = (d["y3_tr"] - y3_mean) / y3_std
    y3_va_z = (d["y3_va"] - y3_mean) / y3_std

    X_tr = [d["X_train"], d["cov_tr"]]
    X_va = [d["X_val"],   d["cov_va"]]

    y_tr = {"y1_binary": d["y1_tr"], "y2_count": d["y2_tr"], "y3_continuous": y3_tr_z}
    y_va = {"y1_binary": d["y1_va"], "y2_count": d["y2_va"], "y3_continuous": y3_va_z}

    # ------------------------------------------------------------------
    # Train: Our model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training: OUR model")
    print("=" * 60)
    tf.keras.backend.clear_session()
    model_ours = build_model_ours(mc=True)
    fit_model(model_ours, X_tr, y_tr, X_va, y_va, epochs=EPOCHS_OURS)

    y1_ours, y2_ours, y3_ours_z = mc_predict(model_ours, X_va)
    y3_ours = y3_ours_z * y3_std + y3_mean

    metrics_ours = evaluate(
        "OUR model (MC mean)",
        d["y1_va"], d["y2_va"], d["y3_va"],
        y1_ours, y2_ours, y3_ours,
    )

    save_np(y1_ours,   DATA_DIR / "y1_binary_prob_mean_ours_update3.ny")
    save_np(y2_ours,   DATA_DIR / "y2_count_lambda_mean_ours_update3.ny")
    save_np(y3_ours,   DATA_DIR / "y3_continuous_pred_mean_ours_update3.ny")
    save_np(d["y1_va"], DATA_DIR / "y1_binary_true_update3.ny")
    save_np(d["y2_va"], DATA_DIR / "y2_count_true_update3.ny")
    save_np(d["y3_va"], DATA_DIR / "y3_continuous_true_update3.ny")

    # ------------------------------------------------------------------
    # Train: Old-3 baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training: OLD-3 baseline")
    print("=" * 60)
    tf.keras.backend.clear_session()
    model_old3 = build_model_old3(mc=True)
    fit_model(model_old3, X_tr, y_tr, X_va, y_va, epochs=EPOCHS_OLD3)

    y1_old3, y2_old3, y3_old3_z = mc_predict(model_old3, X_va)
    y3_old3 = y3_old3_z * y3_std + y3_mean

    metrics_old3 = evaluate(
        "OLD-3OUT baseline (MC mean)",
        d["y1_va"], d["y2_va"], d["y3_va"],
        y1_old3, y2_old3, y3_old3,
    )

    save_np(y1_old3, DATA_DIR / "y1_binary_prob_mean_old3_update3.ny")
    save_np(y2_old3, DATA_DIR / "y2_count_lambda_mean_old3_update3.ny")
    save_np(y3_old3, DATA_DIR / "y3_continuous_pred_mean_old3_update3.ny")

    # ------------------------------------------------------------------
    # Save metrics comparison CSV
    # ------------------------------------------------------------------
    csv_path = DATA_DIR / "metrics_comparison_update3.csv"
    header = ["model", "y1_acc@0.5", "y1_acc@best", "y1_brier",
              "y2_rmse", "y2_mpd", "y3_rmse", "y3_r2"]

    def _row(name: str, m: dict) -> list:
        return [
            name,
            f"{m['y1_acc_05']:.6f}",
            f"{m['y1_acc_best']:.6f}",
            f"{m['y1_brier']:.6f}",
            f"{m['y2_rmse']:.6f}",
            f"{m['y2_mpd']:.6f}",
            f"{m['y3_rmse']:.6f}",
            f"{m['y3_r2']:.6f}",
        ]

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([header, _row("ours", metrics_ours), _row("old3", metrics_old3)])

    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
