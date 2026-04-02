"""
Trainable Patch2Self baseline for 4D diffusion MRI volumes.

This module mirrors the core self-supervised idea used in Patch2Self:
for each held-out volume j, fit a linear model from all other volumes and
predict j (J-invariant denoising).

Implementation choices are aligned with the current DIPY implementation
(1.12.x, Patch2Self version=3):
- b0 and DWI volumes are fit separately (using ``b0_threshold``).
- optional skipping of b0 denoising.
- ``model`` in {"ols", "ridge", "lasso"}.
- optional CountSketch-style row sketching for faster fitting.

References
----------
- Fadnavis et al., NeurIPS 2020 (Patch2Self)
- Fadnavis et al., CVPR 2024 (Patch2Self2 / sketching)
- DIPY patch2self source (v1.12.x)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Patch2SelfConfig:
    """Configuration for fitting Patch2Self.

    Parameters
    ----------
    model
        Regression backend: "ols", "ridge", or "lasso".
        ``ols`` uses Ridge(alpha=1e-10), matching DIPY behavior.
    alpha
        Regularization coefficient for ridge/lasso.
    b0_threshold
        Volumes with b-value <= threshold are treated as b0 volumes.
    b0_denoising
        If False, b0 volumes are copied unchanged.
    sketch_fraction
        Fraction of voxel rows used for CountSketch-style coreset fitting.
        Set to 1.0 to disable sketching and fit on all voxels.
    random_state
        Seed for sketching randomness.
    sketch_chunk_size
        Number of rows processed per chunk while sketching.
    predict_chunk_size
        Number of rows processed per chunk during full-volume prediction.
    dtype
        Compute dtype used during fit/predict.
    """

    model: str = "ols"
    alpha: float = 1.0
    b0_threshold: float = 50.0
    b0_denoising: bool = True
    sketch_fraction: float = 0.30
    random_state: int | None = 42
    sketch_chunk_size: int = 200_000
    predict_chunk_size: int = 200_000
    dtype: Any = np.float32


@dataclass
class Patch2SelfModel:
    """Fitted Patch2Self linear model.

    Attributes
    ----------
    coefficients
        Array of shape (N, N). Column ``j`` contains source-volume weights used
        to predict held-out volume ``j``.
    intercepts
        Array of shape (N,), one intercept per held-out volume.
    b0_idx
        Indices of b0 volumes used at fit time.
    dwi_idx
        Indices of diffusion-weighted volumes used at fit time.
    config
        Fit configuration.
    """

    coefficients: np.ndarray
    intercepts: np.ndarray
    b0_idx: np.ndarray
    dwi_idx: np.ndarray
    config: Patch2SelfConfig

    def save(self, path: str | Path) -> None:
        """Persist fitted model to ``.npz``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            coefficients=self.coefficients,
            intercepts=self.intercepts,
            b0_idx=self.b0_idx,
            dwi_idx=self.dwi_idx,
            config=np.array(
                [
                    self.config.model,
                    self.config.alpha,
                    self.config.b0_threshold,
                    int(self.config.b0_denoising),
                    self.config.sketch_fraction,
                    -1 if self.config.random_state is None else int(self.config.random_state),
                    self.config.sketch_chunk_size,
                    self.config.predict_chunk_size,
                    str(np.dtype(self.config.dtype)),
                ],
                dtype=object,
            ),
        )

    @staticmethod
    def load(path: str | Path) -> "Patch2SelfModel":
        """Load a fitted model from ``.npz``."""
        z = np.load(path, allow_pickle=True)
        cfg_raw = z["config"]
        random_state = int(cfg_raw[5])
        if random_state < 0:
            random_state = None

        cfg = Patch2SelfConfig(
            model=str(cfg_raw[0]),
            alpha=float(cfg_raw[1]),
            b0_threshold=float(cfg_raw[2]),
            b0_denoising=bool(int(cfg_raw[3])),
            sketch_fraction=float(cfg_raw[4]),
            random_state=random_state,
            sketch_chunk_size=int(cfg_raw[6]),
            predict_chunk_size=int(cfg_raw[7]),
            dtype=np.dtype(str(cfg_raw[8])),
        )
        return Patch2SelfModel(
            coefficients=z["coefficients"],
            intercepts=z["intercepts"],
            b0_idx=z["b0_idx"],
            dwi_idx=z["dwi_idx"],
            config=cfg,
        )


def _validate_inputs(data: np.ndarray, bvals: np.ndarray, cfg: Patch2SelfConfig) -> None:
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data (X, Y, Z, N), got shape {data.shape}.")
    if bvals.ndim != 1:
        raise ValueError(f"Expected 1D bvals of shape (N,), got {bvals.shape}.")
    if data.shape[-1] != bvals.shape[0]:
        raise ValueError(
            "Number of volumes in data and bvals must match: "
            f"{data.shape[-1]} vs {bvals.shape[0]}."
        )
    if cfg.model.lower() not in {"ols", "ridge", "lasso"}:
        raise ValueError("model must be one of: 'ols', 'ridge', 'lasso'.")
    if cfg.sketch_fraction <= 0:
        raise ValueError("sketch_fraction must be > 0.")


def _soft_threshold_scalar(x: float, t: float) -> float:
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


def _solve_ols_or_ridge(
    X: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, float]:
    """Solve linear regression with intercept via normal equations.

    We solve for `[w, b]` in:
        y ≈ X w + b
    with L2 penalty on w only (not on b).
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n_rows, n_feats = X64.shape

    xtx = X64.T @ X64
    if ridge_alpha > 0:
        xtx = xtx + ridge_alpha * np.eye(n_feats, dtype=np.float64)
    xty = X64.T @ y64
    sum_x = X64.sum(axis=0)
    sum_y = float(y64.sum())

    A = np.empty((n_feats + 1, n_feats + 1), dtype=np.float64)
    A[:n_feats, :n_feats] = xtx
    A[:n_feats, n_feats] = sum_x
    A[n_feats, :n_feats] = sum_x
    A[n_feats, n_feats] = float(n_rows)

    b = np.empty(n_feats + 1, dtype=np.float64)
    b[:n_feats] = xty
    b[n_feats] = sum_y

    try:
        params = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(A, b, rcond=None)[0]

    coef = params[:n_feats]
    intercept = float(params[n_feats])
    return coef, intercept


def _solve_lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Simple coordinate-descent Lasso with intercept.

    Objective:
        (1/(2m)) * ||y - Xw - b||^2 + alpha * ||w||_1
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    n_rows, n_feats = X64.shape

    coef = np.zeros(n_feats, dtype=np.float64)
    intercept = float(y64.mean())

    # Precompute per-feature squared norms.
    z = np.sum(X64 * X64, axis=0) / max(1, n_rows)
    z[z < 1e-12] = 1e-12

    y_pred = X64 @ coef + intercept
    for _ in range(max_iter):
        coef_old = coef.copy()
        old_intercept = intercept

        # Update intercept first (unregularized).
        intercept = float((y64 - X64 @ coef).mean())
        y_pred = X64 @ coef + intercept

        # Coordinate updates.
        for j in range(n_feats):
            y_pred_wo_j = y_pred - X64[:, j] * coef[j]
            r_j = y64 - y_pred_wo_j
            rho_j = float((X64[:, j] * r_j).sum() / max(1, n_rows))
            coef[j] = _soft_threshold_scalar(rho_j, alpha) / z[j]
            y_pred = y_pred_wo_j + X64[:, j] * coef[j]

        delta = np.max(np.abs(coef - coef_old))
        delta = max(delta, abs(intercept - old_intercept))
        if delta < tol:
            break

    return coef, intercept


def _fit_linear_model(X: np.ndarray, y: np.ndarray, model: str, alpha: float) -> tuple[np.ndarray, float]:
    kind = model.lower()
    if kind == "ols":
        # Matches DIPY's OLS behavior (ridge with tiny alpha for stability).
        return _solve_ols_or_ridge(X, y, ridge_alpha=1e-10)
    if kind == "ridge":
        return _solve_ols_or_ridge(X, y, ridge_alpha=float(alpha))
    return _solve_lasso_cd(X, y, alpha=float(alpha), max_iter=50)


def _count_sketch_rows(
    flat: np.ndarray,
    sketch_rows: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    """CountSketch-style row sketching used by Patch2Self v3 acceleration."""
    n_rows, n_cols = flat.shape
    if sketch_rows >= n_rows:
        return flat

    hashed = rng.integers(0, sketch_rows, size=n_rows, endpoint=False)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=flat.dtype), size=n_rows)
    sketched = np.zeros((sketch_rows, n_cols), dtype=flat.dtype)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        block = flat[start:end] * signs[start:end, None]
        np.add.at(sketched, hashed[start:end], block)

    return sketched


def _fit_group(
    train_group: np.ndarray,
    group_indices: np.ndarray,
    cfg: Patch2SelfConfig,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    verbose: bool,
    label: str,
) -> None:
    """Fit leave-one-volume-out regressors for a volume group."""
    n_group = group_indices.size
    if n_group == 0:
        return

    if n_group == 1:
        # Nothing to regress from; identity mapping keeps the volume unchanged.
        j = int(group_indices[0])
        coefficients[j, j] = 1.0
        intercepts[j] = 0.0
        return

    for local_j, global_j in enumerate(group_indices):
        X = np.delete(train_group, local_j, axis=1)
        y = train_group[:, local_j]
        coef, intercept = _fit_linear_model(X, y, model=cfg.model, alpha=cfg.alpha)

        others = np.delete(group_indices, local_j)
        coefficients[others, global_j] = coef.astype(coefficients.dtype, copy=False)
        intercepts[global_j] = np.asarray(intercept, dtype=intercepts.dtype)

        if verbose:
            print(f"  fitted {label} volume {local_j + 1}/{n_group}")


def fit_patch2self(
    data: np.ndarray,
    bvals: np.ndarray,
    cfg: Patch2SelfConfig | None = None,
    *,
    verbose: bool = False,
) -> Patch2SelfModel:
    """Fit trainable Patch2Self regressors for one 4D subject volume.

    Parameters
    ----------
    data
        Noisy DWI array of shape (X, Y, Z, N).
    bvals
        b-values of shape (N,).
    cfg
        Patch2Self hyperparameters.
    verbose
        Print progress.

    Returns
    -------
    Patch2SelfModel
        Fitted model with per-volume coefficients and intercepts.
    """
    cfg = cfg or Patch2SelfConfig()
    _validate_inputs(data, bvals, cfg)

    calc_dtype = np.dtype(cfg.dtype)
    data_4d = np.asarray(data, dtype=calc_dtype)
    flat = data_4d.reshape(-1, data_4d.shape[-1])

    b0_idx = np.flatnonzero(bvals <= cfg.b0_threshold)
    dwi_idx = np.flatnonzero(bvals > cfg.b0_threshold)
    if dwi_idx.size == 0:
        raise ValueError("No diffusion-weighted volumes found. Check b0_threshold and bvals.")

    b0_denoising = cfg.b0_denoising and b0_idx.size > 1
    if verbose and cfg.b0_denoising and not b0_denoising:
        print("b0 denoising disabled automatically (need at least 2 b0 volumes).")

    n_rows, n_vols = flat.shape
    sketch_rows = int(round(cfg.sketch_fraction * n_rows))
    sketch_rows = max(n_vols, min(n_rows, sketch_rows))

    if verbose:
        print(
            f"Patch2Self fit: rows={n_rows:,}, vols={n_vols}, "
            f"sketch_rows={sketch_rows:,}, model={cfg.model}"
        )

    rng = np.random.default_rng(cfg.random_state)
    train_matrix = _count_sketch_rows(flat, sketch_rows, rng, cfg.sketch_chunk_size)

    coefficients = np.zeros((n_vols, n_vols), dtype=calc_dtype)
    intercepts = np.zeros(n_vols, dtype=calc_dtype)

    if b0_denoising:
        train_b0 = train_matrix[:, b0_idx]
        _fit_group(
            train_group=train_b0,
            group_indices=b0_idx,
            cfg=cfg,
            coefficients=coefficients,
            intercepts=intercepts,
            verbose=verbose,
            label="b0",
        )
    else:
        # Keep b0s unchanged.
        for j in b0_idx:
            coefficients[j, j] = 1.0
            intercepts[j] = 0.0

    train_dwi = train_matrix[:, dwi_idx]
    _fit_group(
        train_group=train_dwi,
        group_indices=dwi_idx,
        cfg=cfg,
        coefficients=coefficients,
        intercepts=intercepts,
        verbose=verbose,
        label="dwi",
    )

    return Patch2SelfModel(
        coefficients=coefficients,
        intercepts=intercepts,
        b0_idx=b0_idx,
        dwi_idx=dwi_idx,
        config=cfg,
    )


def denoise_with_model(
    data: np.ndarray,
    model: Patch2SelfModel,
    *,
    clip_negative_vals: bool = False,
    shift_intensity: bool = True,
) -> np.ndarray:
    """Apply a fitted Patch2Self model to a 4D DWI array.

    Parameters
    ----------
    data
        Input DWI array of shape (X, Y, Z, N).
    model
        A fitted model from :func:`fit_patch2self`.
    clip_negative_vals
        If True, clip negative values to 0.
    shift_intensity
        If True and clipping is disabled, shift each volume by ``-min`` when
        needed so values are non-negative.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data (X, Y, Z, N), got shape {data.shape}.")
    if data.shape[-1] != model.coefficients.shape[0]:
        raise ValueError(
            "Data volume count does not match model: "
            f"{data.shape[-1]} vs {model.coefficients.shape[0]}."
        )

    cfg = model.config
    calc_dtype = np.dtype(cfg.dtype)
    data_4d = np.asarray(data, dtype=calc_dtype)
    flat = data_4d.reshape(-1, data_4d.shape[-1])

    out = np.empty_like(flat)
    chunk = max(1, int(cfg.predict_chunk_size))
    for start in range(0, flat.shape[0], chunk):
        end = min(start + chunk, flat.shape[0])
        out[start:end] = flat[start:end] @ model.coefficients + model.intercepts

    denoised = out.reshape(data_4d.shape)

    if clip_negative_vals:
        np.clip(denoised, 0, None, out=denoised)
    elif shift_intensity:
        # Shift each volume only if needed.
        for j in range(denoised.shape[-1]):
            min_val = float(denoised[..., j].min())
            if min_val < 0:
                denoised[..., j] -= min_val

    return denoised


def fit_predict_patch2self(
    data: np.ndarray,
    bvals: np.ndarray,
    cfg: Patch2SelfConfig | None = None,
    *,
    clip_negative_vals: bool = False,
    shift_intensity: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, Patch2SelfModel]:
    """Convenience wrapper: fit model and denoise in one call."""
    model = fit_patch2self(data, bvals, cfg=cfg, verbose=verbose)
    denoised = denoise_with_model(
        data,
        model,
        clip_negative_vals=clip_negative_vals,
        shift_intensity=shift_intensity,
    )
    return denoised, model
