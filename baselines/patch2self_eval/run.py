"""Tune and evaluate Patch2Self on the clean Zarr dataset.

The runner loads subjects through :class:`research.dataset.DWISliceDataset` so
the baseline follows the same dataset contract as model training. For every
final trial it:

1. degrades clean DWI with the same k-space/noise helpers used by the dataset,
2. runs Patch2Self,
3. fits a 6D DTI tensor from the denoised DWI,
4. evaluates tensor/FA/ADC metrics against ``target_dti_6d``.

Example
-------
    python -m baselines.patch2self_eval.run
    python -m baselines.patch2self_eval.run --subjects sub-03 sub-04 --repeats 5
    python -m baselines.patch2self_eval.run --skip_tuning --p2s_model ridge --p2s_alpha 0.1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from research.augment import degrade_dwi_volume
from research.dataset import DWISliceDataset
from research.utils import (
    dti6d_to_scalar_maps,
    fit_dti_to_6d,
    sanitize_dti6d,
    scalar_map_metrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

METRIC_COLS = [
    "tensor_rmse",
    "fa_rmse",
    "fa_mae",
    "fa_nrmse",
    "fa_r2",
    "adc_rmse",
    "adc_mae",
    "adc_nrmse",
    "adc_r2",
]
HIGHER_IS_BETTER = {"fa_r2", "adc_r2"}


@dataclass(frozen=True)
class Patch2SelfConfig:
    """Patch2Self parameters plus the downstream DTI fit method."""

    name: str
    model: str
    alpha: float
    b0_denoising: bool
    clip_negative: bool
    shift_intensity: bool
    dti_fit_method: str
    b0_threshold: float


@dataclass(frozen=True)
class DegradationTrial:
    repeat: int
    keep_fraction: float
    noise_level: float
    degrade_seed: int


@dataclass
class SubjectRecord:
    key: str
    target_dwi: np.ndarray
    target_dti6d: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    brain_mask: np.ndarray


def _expand_subjects(subject_ids: list[str] | None, all_keys: list[str]) -> list[str]:
    """Accept exact Zarr keys or biological subject IDs such as ``sub-03``."""
    if not subject_ids:
        return []

    subjects: list[str] = []
    seen: set[str] = set()
    for subject_id in subject_ids:
        if subject_id in all_keys:
            matches = [subject_id]
        else:
            matches = [key for key in all_keys if key.rsplit("_ses-", 1)[0] == subject_id]
        for match in matches:
            if match not in seen:
                seen.add(match)
                subjects.append(match)
    return subjects


def _load_subject_record(
    zarr_path: str,
    subject_key: str,
    *,
    b0_threshold: float,
) -> SubjectRecord:
    """Load one subject through ``DWISliceDataset`` from ``research.dataset``."""
    ds = DWISliceDataset(
        zarr_path,
        [subject_key],
        augment=False,
        b0_threshold=b0_threshold,
        use_brain_mask=True,
        on_the_fly_degradation=False,
        random_axis=False,
        eval_mode=True,
    )
    data = ds._data[subject_key]  # DWISliceDataset is the source of truth here.
    brain_mask = ds._brain_masks[subject_key]
    if brain_mask is None:
        brain_mask = np.ones(data["target_dwi"].shape[:3], dtype=bool)

    return SubjectRecord(
        key=subject_key,
        target_dwi=np.asarray(data["target_dwi"], dtype=np.float32),
        target_dti6d=np.asarray(data["target_dti_6d"], dtype=np.float32),
        bvals=np.asarray(data["bvals"], dtype=np.float32),
        bvecs=np.asarray(data["bvecs"], dtype=np.float32),
        brain_mask=np.asarray(brain_mask, dtype=bool),
    )


def _make_degradation_plan(
    repeats: int,
    *,
    seed: int,
    keep_fraction_range: tuple[float, float],
    noise_range: tuple[float, float],
) -> list[DegradationTrial]:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")

    keep_min, keep_max = keep_fraction_range
    noise_min, noise_max = noise_range
    if not (0.0 < keep_min <= keep_max <= 1.0):
        raise ValueError("keep fraction range must satisfy 0 < min <= max <= 1")
    if not (0.0 <= noise_min <= noise_max):
        raise ValueError("noise range must satisfy 0 <= min <= max")

    rng = np.random.default_rng(seed)
    trials: list[DegradationTrial] = []
    for repeat in range(repeats):
        keep_fraction = keep_min if keep_min == keep_max else float(rng.uniform(keep_min, keep_max))
        noise_level = noise_min if noise_min == noise_max else float(rng.uniform(noise_min, noise_max))
        trials.append(
            DegradationTrial(
                repeat=repeat,
                keep_fraction=keep_fraction,
                noise_level=noise_level,
                degrade_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
        )
    return trials


def _run_patch2self(
    noisy_dwi: np.ndarray,
    bvals: np.ndarray,
    p2s_cfg: Patch2SelfConfig,
    *,
    quiet: bool,
) -> np.ndarray:
    from dipy.denoise.patch2self import patch2self

    output_stream = open(os.devnull, "w") if quiet else contextlib.nullcontext()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if quiet:
                with contextlib.redirect_stdout(output_stream), contextlib.redirect_stderr(output_stream):
                    denoised = patch2self(
                        noisy_dwi,
                        bvals,
                        model=p2s_cfg.model,
                        alpha=p2s_cfg.alpha,
                        b0_threshold=p2s_cfg.b0_threshold,
                        shift_intensity=p2s_cfg.shift_intensity,
                        clip_negative_vals=p2s_cfg.clip_negative,
                        b0_denoising=p2s_cfg.b0_denoising,
                        verbose=False,
                    )
            else:
                denoised = patch2self(
                    noisy_dwi,
                    bvals,
                    model=p2s_cfg.model,
                    alpha=p2s_cfg.alpha,
                    b0_threshold=p2s_cfg.b0_threshold,
                    shift_intensity=p2s_cfg.shift_intensity,
                    clip_negative_vals=p2s_cfg.clip_negative,
                    b0_denoising=p2s_cfg.b0_denoising,
                    verbose=True,
                )
    finally:
        if quiet:
            output_stream.close()

    return np.asarray(denoised, dtype=np.float32)


def _compute_dti_metrics(
    pred_dti6d: np.ndarray,
    target_dti6d: np.ndarray,
    *,
    mask: np.ndarray | None,
    max_diffusivity: float,
) -> dict[str, float]:
    pred_clean = sanitize_dti6d(pred_dti6d, max_eigenvalue=max_diffusivity)
    target_clean = sanitize_dti6d(target_dti6d, max_eigenvalue=max_diffusivity)

    diff = pred_clean - target_clean
    if mask is not None:
        tensor_rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    else:
        tensor_rmse = float(np.sqrt(np.mean(diff ** 2)))

    pred_fa, pred_adc = dti6d_to_scalar_maps(pred_clean)
    target_fa, target_adc = dti6d_to_scalar_maps(target_clean)
    fa_metrics = scalar_map_metrics(target_fa, pred_fa, mask=mask)
    adc_metrics = scalar_map_metrics(target_adc, pred_adc, mask=mask)

    return {
        "tensor_rmse": round(tensor_rmse, 8),
        "fa_rmse": round(fa_metrics["rmse"], 8),
        "fa_mae": round(fa_metrics["mae"], 8),
        "fa_nrmse": round(fa_metrics["nrmse"], 8),
        "fa_r2": round(fa_metrics["r2"], 6),
        "adc_rmse": round(adc_metrics["rmse"], 10),
        "adc_mae": round(adc_metrics["mae"], 10),
        "adc_nrmse": round(adc_metrics["nrmse"], 8),
        "adc_r2": round(adc_metrics["r2"], 6),
    }


def _fit_dti_and_score(
    dwi_4d: np.ndarray,
    record: SubjectRecord,
    *,
    fit_method: str,
    b0_threshold: float,
    max_diffusivity: float,
) -> tuple[dict[str, float], np.ndarray]:
    bvecs_n3 = record.bvecs.T if record.bvecs.shape[0] == 3 else record.bvecs
    fitted = fit_dti_to_6d(
        np.maximum(dwi_4d, 0.0),
        record.bvals,
        bvecs_n3=bvecs_n3,
        fit_method=fit_method,
        b0_threshold=b0_threshold,
    )
    metrics = _compute_dti_metrics(
        fitted,
        record.target_dti6d,
        mask=record.brain_mask,
        max_diffusivity=max_diffusivity,
    )
    return metrics, fitted


def _row_metadata(subject: str, trial: DegradationTrial) -> dict:
    return {
        "subject": subject,
        "repeat": int(trial.repeat),
        "keep_fraction": round(float(trial.keep_fraction), 6),
        "noise_level": round(float(trial.noise_level), 6),
        "degrade_seed": int(trial.degrade_seed),
    }


def _config_metadata(p2s_cfg: Patch2SelfConfig) -> dict:
    return {
        "config_name": p2s_cfg.name,
        "p2s_model": p2s_cfg.model,
        "p2s_alpha": p2s_cfg.alpha,
        "p2s_b0_denoising": p2s_cfg.b0_denoising,
        "p2s_clip_negative": p2s_cfg.clip_negative,
        "p2s_shift_intensity": p2s_cfg.shift_intensity,
        "dti_fit_method": p2s_cfg.dti_fit_method,
    }


def _default_tuned_config(args) -> Patch2SelfConfig:
    b0_part = "b0" if args.p2s_b0_denoising else "no_b0"
    clip_part = "clip" if args.p2s_clip_negative else "no_clip"
    shift_part = "shift" if args.p2s_shift_intensity else "no_shift"
    return Patch2SelfConfig(
        name=f"{args.p2s_model}_{args.p2s_alpha:g}_{b0_part}_{clip_part}_{shift_part}",
        model=args.p2s_model,
        alpha=float(args.p2s_alpha),
        b0_denoising=bool(args.p2s_b0_denoising),
        clip_negative=bool(args.p2s_clip_negative),
        shift_intensity=bool(args.p2s_shift_intensity),
        dti_fit_method=args.dti_fit_method,
        b0_threshold=float(args.b0_threshold),
    )


def _build_tuning_grid(args) -> list[Patch2SelfConfig]:
    b0_threshold = float(args.b0_threshold)
    dti_fit_method = args.dti_fit_method
    configs: list[Patch2SelfConfig] = [
        Patch2SelfConfig(
            name="ols_project_default",
            model="ols",
            alpha=float(cfg.P2S_ALPHA),
            b0_denoising=bool(cfg.P2S_B0_DENOISING),
            clip_negative=bool(cfg.P2S_CLIP_NEGATIVE),
            shift_intensity=bool(cfg.P2S_SHIFT_INTENSITY),
            dti_fit_method=dti_fit_method,
            b0_threshold=b0_threshold,
        ),
        Patch2SelfConfig(
            name="ols_no_b0_clip",
            model="ols",
            alpha=float(cfg.P2S_ALPHA),
            b0_denoising=False,
            clip_negative=True,
            shift_intensity=True,
            dti_fit_method=dti_fit_method,
            b0_threshold=b0_threshold,
        ),
    ]

    for alpha in args.tune_alphas:
        configs.append(
            Patch2SelfConfig(
                name=f"ridge_{alpha:g}_no_b0_clip",
                model="ridge",
                alpha=float(alpha),
                b0_denoising=False,
                clip_negative=True,
                shift_intensity=True,
                dti_fit_method=dti_fit_method,
                b0_threshold=b0_threshold,
            )
        )

    if args.tune_grid == "broad":
        for alpha in args.tune_alphas:
            configs.extend(
                [
                    Patch2SelfConfig(
                        name=f"ridge_{alpha:g}_b0_clip",
                        model="ridge",
                        alpha=float(alpha),
                        b0_denoising=True,
                        clip_negative=True,
                        shift_intensity=True,
                        dti_fit_method=dti_fit_method,
                        b0_threshold=b0_threshold,
                    ),
                    Patch2SelfConfig(
                        name=f"lasso_{alpha:g}_no_b0_clip",
                        model="lasso",
                        alpha=float(alpha),
                        b0_denoising=False,
                        clip_negative=True,
                        shift_intensity=True,
                        dti_fit_method=dti_fit_method,
                        b0_threshold=b0_threshold,
                    ),
                ]
            )

    unique: list[Patch2SelfConfig] = []
    seen: set[tuple] = set()
    for p2s_cfg in configs:
        key = (
            p2s_cfg.model,
            p2s_cfg.alpha,
            p2s_cfg.b0_denoising,
            p2s_cfg.clip_negative,
            p2s_cfg.shift_intensity,
            p2s_cfg.dti_fit_method,
        )
        if key not in seen:
            seen.add(key)
            unique.append(p2s_cfg)
    return unique


def _summarize_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    available_metrics = [col for col in METRIC_COLS if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[available_metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _sort_summary(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric not in HIGHER_IS_BETTER
    return summary.sort_values(f"{metric}_mean", ascending=ascending).reset_index(drop=True)


def _write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def tune_patch2self(
    args,
    *,
    subjects: list[str],
    trials: list[DegradationTrial],
    out_dir: Path,
) -> Patch2SelfConfig:
    p2s_grid = _build_tuning_grid(args)
    tune_trials = trials[: args.tune_repeats]
    if not tune_trials:
        raise ValueError("--tune_repeats must be >= 1")

    log.info(
        "Tuning Patch2Self: %d configs x %d subjects x %d repeats",
        len(p2s_grid),
        len(subjects),
        len(tune_trials),
    )
    rows: list[dict] = []

    for subject_key in subjects:
        record = _load_subject_record(
            args.zarr_path,
            subject_key,
            b0_threshold=args.b0_threshold,
        )
        log.info("Tune subject %s  shape=%s", subject_key, record.target_dwi.shape)
        for trial in tune_trials:
            noisy = degrade_dwi_volume(
                record.target_dwi,
                keep_fraction=trial.keep_fraction,
                rel_noise_level=trial.noise_level,
                seed=trial.degrade_seed,
            )

            for config_idx, p2s_cfg in enumerate(p2s_grid):
                t0 = time.time()
                try:
                    denoised = _run_patch2self(
                        noisy,
                        record.bvals,
                        p2s_cfg,
                        quiet=not args.verbose_patch2self,
                    )
                    metrics, _ = _fit_dti_and_score(
                        denoised,
                        record,
                        fit_method=p2s_cfg.dti_fit_method,
                        b0_threshold=args.b0_threshold,
                        max_diffusivity=args.max_diffusivity,
                    )
                except Exception as exc:
                    log.warning(
                        "Tune failed subject=%s repeat=%d config=%s: %s",
                        subject_key,
                        trial.repeat,
                        p2s_cfg.name,
                        exc,
                    )
                    continue

                row = {
                    **_row_metadata(subject_key, trial),
                    "config_idx": int(config_idx),
                    **_config_metadata(p2s_cfg),
                    "elapsed_s": round(time.time() - t0, 2),
                    **metrics,
                }
                rows.append(row)
                log.info(
                    "  cfg=%02d %-24s %s=%.8f elapsed=%.1fs",
                    config_idx,
                    p2s_cfg.name,
                    args.tune_metric,
                    metrics[args.tune_metric],
                    row["elapsed_s"],
                )

    if not rows:
        log.warning("Patch2Self tuning produced no successful rows; using fallback config.")
        return _default_tuned_config(args)

    tune_df = pd.DataFrame(rows)
    tune_path = out_dir / "tuning_trials.csv"
    tune_df.to_csv(tune_path, index=False)
    log.info("Saved tuning trials -> %s", tune_path)

    group_cols = [
        "config_idx",
        "config_name",
        "p2s_model",
        "p2s_alpha",
        "p2s_b0_denoising",
        "p2s_clip_negative",
        "p2s_shift_intensity",
        "dti_fit_method",
    ]
    summary = _sort_summary(_summarize_rows(tune_df, group_cols), args.tune_metric)
    summary_path = out_dir / "tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved tuning summary -> %s", summary_path)

    best_row = summary.iloc[0]
    best_cfg = Patch2SelfConfig(
        name=str(best_row["config_name"]),
        model=str(best_row["p2s_model"]),
        alpha=float(best_row["p2s_alpha"]),
        b0_denoising=bool(best_row["p2s_b0_denoising"]),
        clip_negative=bool(best_row["p2s_clip_negative"]),
        shift_intensity=bool(best_row["p2s_shift_intensity"]),
        dti_fit_method=str(best_row["dti_fit_method"]),
        b0_threshold=float(args.b0_threshold),
    )
    log.info(
        "Best Patch2Self config: %s  %s_mean=%.8f",
        best_cfg.name,
        args.tune_metric,
        best_row[f"{args.tune_metric}_mean"],
    )
    return best_cfg


def _save_final_outputs(rows: list[dict], out_dir: Path, tune_metric: str) -> None:
    final_df = pd.DataFrame(rows)
    sort_cols = [col for col in ["method", "subject", "repeat"] if col in final_df.columns]
    final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    final_path = out_dir / "final_trials.csv"
    final_df.to_csv(final_path, index=False)
    log.info("Saved final trials -> %s", final_path)

    group_cols = ["method"]
    if "config_name" in final_df.columns:
        group_cols.extend(
            [
                "config_name",
                "p2s_model",
                "p2s_alpha",
                "p2s_b0_denoising",
                "p2s_clip_negative",
                "p2s_shift_intensity",
                "dti_fit_method",
            ]
        )
    summary = _sort_summary(_summarize_rows(final_df, group_cols), tune_metric)
    summary_path = out_dir / "final_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved final summary -> %s", summary_path)

    p2s_df = final_df[final_df["method"] == "patch2self"].copy()
    if not p2s_df.empty:
        p2s_path = out_dir / "metrics_patch2self.csv"
        p2s_df.to_csv(p2s_path, index=False)
        log.info("Saved Patch2Self metrics -> %s", p2s_path)


def _save_fitted_tensor(
    out_dir: Path,
    subject_key: str,
    trial: DegradationTrial,
    p2s_cfg: Patch2SelfConfig,
    fitted_dti6d: np.ndarray,
) -> None:
    tensor_dir = out_dir / "fitted_dti6d"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    path = tensor_dir / f"{subject_key}_repeat-{trial.repeat:02d}_{p2s_cfg.name}.npz"
    np.savez_compressed(
        path,
        fitted_dti6d=fitted_dti6d.astype(np.float32),
        repeat=np.array([trial.repeat], dtype=np.int32),
        keep_fraction=np.array([trial.keep_fraction], dtype=np.float32),
        noise_level=np.array([trial.noise_level], dtype=np.float32),
        degrade_seed=np.array([trial.degrade_seed], dtype=np.int64),
    )


def run_final_eval(
    args,
    *,
    subjects: list[str],
    trials: list[DegradationTrial],
    p2s_cfg: Patch2SelfConfig,
    out_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    total = len(subjects) * len(trials)
    done = 0

    for subject_key in subjects:
        record = _load_subject_record(
            args.zarr_path,
            subject_key,
            b0_threshold=args.b0_threshold,
        )
        log.info("Final subject %s  shape=%s", subject_key, record.target_dwi.shape)

        for trial in trials:
            done += 1
            noisy = degrade_dwi_volume(
                record.target_dwi,
                keep_fraction=trial.keep_fraction,
                rel_noise_level=trial.noise_level,
                seed=trial.degrade_seed,
            )

            if args.include_noisy_baseline:
                t_input = time.time()
                input_metrics, _ = _fit_dti_and_score(
                    noisy,
                    record,
                    fit_method=p2s_cfg.dti_fit_method,
                    b0_threshold=args.b0_threshold,
                    max_diffusivity=args.max_diffusivity,
                )
                rows.append(
                    {
                        "method": "noisy_input",
                        **_row_metadata(subject_key, trial),
                        "elapsed_s": round(time.time() - t_input, 2),
                        **input_metrics,
                    }
                )

            t0 = time.time()
            denoised = _run_patch2self(
                noisy,
                record.bvals,
                p2s_cfg,
                quiet=not args.verbose_patch2self,
            )
            p2s_metrics, fitted_dti6d = _fit_dti_and_score(
                denoised,
                record,
                fit_method=p2s_cfg.dti_fit_method,
                b0_threshold=args.b0_threshold,
                max_diffusivity=args.max_diffusivity,
            )
            row = {
                "method": "patch2self",
                **_row_metadata(subject_key, trial),
                **_config_metadata(p2s_cfg),
                "elapsed_s": round(time.time() - t0, 2),
                **p2s_metrics,
            }
            rows.append(row)

            if args.save_fitted_tensors:
                _save_fitted_tensor(out_dir, subject_key, trial, p2s_cfg, fitted_dti6d)

            log.info(
                "[%d/%d] %-14s repeat=%02d keep=%.3f noise=%.3f "
                "tensor_rmse=%.8f fa_rmse=%.6f adc_rmse=%.2e elapsed=%.1fs",
                done,
                total,
                subject_key,
                trial.repeat,
                trial.keep_fraction,
                trial.noise_level,
                p2s_metrics["tensor_rmse"],
                p2s_metrics["fa_rmse"],
                p2s_metrics["adc_rmse"],
                row["elapsed_s"],
            )

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr_path", default="dataset/default_clean.zarr")
    parser.add_argument("--out_dir", default="baselines/patch2self_eval/results")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Biological subject IDs or exact Zarr keys. Default: all Zarr groups.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval_seed", type=int, default=cfg.EVAL_DEGRADE_SEED)
    parser.add_argument(
        "--keep_fraction_min",
        type=float,
        default=cfg.EVAL_KEEP_FRACTION_MIN,
    )
    parser.add_argument(
        "--keep_fraction_max",
        type=float,
        default=cfg.EVAL_KEEP_FRACTION_MAX,
    )
    parser.add_argument("--noise_min", type=float, default=cfg.EVAL_NOISE_MIN)
    parser.add_argument("--noise_max", type=float, default=cfg.EVAL_NOISE_MAX)
    parser.add_argument("--b0_threshold", type=float, default=cfg.B0_THRESHOLD)
    parser.add_argument("--dti_fit_method", choices=["WLS", "OLS", "NLLS"], default=cfg.DTI_FIT_METHOD)
    parser.add_argument("--max_diffusivity", type=float, default=cfg.MAX_DIFFUSIVITY)

    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument(
        "--tune_subjects",
        nargs="*",
        default=cfg.VAL_SUBJECTS,
        help="Subjects used for Patch2Self tuning. Default: validation subjects from config.py.",
    )
    parser.add_argument("--tune_repeats", type=int, default=2)
    parser.add_argument(
        "--tune_metric",
        choices=METRIC_COLS,
        default="fa_rmse",
        help="Metric used to select the tuned Patch2Self config.",
    )
    parser.add_argument("--tune_grid", choices=["compact", "broad"], default="compact")
    parser.add_argument(
        "--tune_alphas",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0],
        help="Ridge/lasso alpha values used by the tuning grid.",
    )

    parser.add_argument("--p2s_model", choices=["ols", "ridge", "lasso"], default="ridge")
    parser.add_argument("--p2s_alpha", type=float, default=0.1)
    parser.add_argument("--p2s_b0_denoising", action="store_true", default=False)
    parser.add_argument("--p2s_no_b0_denoising", dest="p2s_b0_denoising", action="store_false")
    parser.add_argument("--p2s_clip_negative", action="store_true", default=True)
    parser.add_argument("--p2s_no_clip_negative", dest="p2s_clip_negative", action="store_false")
    parser.add_argument("--p2s_shift_intensity", action="store_true", default=True)
    parser.add_argument("--p2s_no_shift_intensity", dest="p2s_shift_intensity", action="store_false")

    parser.add_argument(
        "--include_noisy_baseline",
        action="store_true",
        help="Also fit/evaluate DTI directly on the degraded input.",
    )
    parser.add_argument(
        "--save_fitted_tensors",
        action="store_true",
        help="Save fitted Patch2Self 6D tensors as compressed .npz files.",
    )
    parser.add_argument(
        "--verbose_patch2self",
        action="store_true",
        help="Allow DIPY Patch2Self progress output.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = _parse_args() if args is None else args
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(args.zarr_path, mode="r")
    all_keys = sorted(store.keys())
    subjects = _expand_subjects(args.subjects, all_keys) if args.subjects else all_keys
    if not subjects:
        raise ValueError("No final evaluation subjects selected.")

    tune_subjects = _expand_subjects(args.tune_subjects, all_keys)
    if not tune_subjects and not args.skip_tuning:
        log.warning("No tuning subjects matched; falling back to final subjects for tuning.")
        tune_subjects = subjects

    trials = _make_degradation_plan(
        args.repeats,
        seed=args.eval_seed,
        keep_fraction_range=(args.keep_fraction_min, args.keep_fraction_max),
        noise_range=(args.noise_min, args.noise_max),
    )
    pd.DataFrame([asdict(trial) for trial in trials]).to_csv(
        out_dir / "degradation_plan.csv",
        index=False,
    )

    log.info("Final subjects: %d", len(subjects))
    log.info("Final repeats: %d", len(trials))
    log.info(
        "Degradation ranges: keep=[%.3f, %.3f] noise=[%.3f, %.3f] seed=%d",
        args.keep_fraction_min,
        args.keep_fraction_max,
        args.noise_min,
        args.noise_max,
        args.eval_seed,
    )

    if args.skip_tuning:
        p2s_cfg = _default_tuned_config(args)
        log.info("Skipping tuning; using config: %s", p2s_cfg)
    else:
        if args.tune_repeats > args.repeats:
            raise ValueError("--tune_repeats cannot exceed --repeats")
        p2s_cfg = tune_patch2self(
            args,
            subjects=tune_subjects,
            trials=trials,
            out_dir=out_dir,
        )

    _write_json(out_dir / "best_config.json", asdict(p2s_cfg))

    rows = run_final_eval(
        args,
        subjects=subjects,
        trials=trials,
        p2s_cfg=p2s_cfg,
        out_dir=out_dir,
    )
    _save_final_outputs(rows, out_dir, args.tune_metric)

    final_df = pd.DataFrame(rows)
    p2s_df = final_df[final_df["method"] == "patch2self"]
    print()
    print("=" * 72)
    print("  Patch2Self Final Evaluation")
    print("=" * 72)
    print(
        "config={name} model={model} alpha={alpha:g} b0_denoising={b0} "
        "clip_negative={clip} shift_intensity={shift}".format(
            name=p2s_cfg.name,
            model=p2s_cfg.model,
            alpha=p2s_cfg.alpha,
            b0=p2s_cfg.b0_denoising,
            clip=p2s_cfg.clip_negative,
            shift=p2s_cfg.shift_intensity,
        )
    )
    means = p2s_df[METRIC_COLS].mean(numeric_only=True)
    print("  " + "  ".join(f"{metric}={means[metric]:.6g}" for metric in METRIC_COLS))
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
