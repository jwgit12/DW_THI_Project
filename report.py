#!/usr/bin/env python3
"""Generate a self-contained HTML report for a training run.

Reads ``config.json`` and ``history.json`` from a run directory (written by
``train.py``) and renders a single HTML file that explains the data flow of the
DWI -> DTI pipeline and embeds the training curves of that specific run.

    python3 report.py --run_dir runs/production
    python3 report.py --run_dir runs/production --out report.html

The output is fully self-contained (charts embedded as base64 PNG), so it can be
opened directly in a browser or shared as a single file.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {run_dir}")
    config = json.loads(config_path.read_text())
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    return config, history


# ─────────────────────────────────────────────────────────────────────────────
# Charts (embedded as base64 PNG so the report stays a single file)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _line_chart(epochs, series, title, ylabel, *, logy=False, best_epoch=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, values, style in series:
        ax.plot(epochs, values, style, label=label, linewidth=1.6)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#888", linestyle=":", linewidth=1.2,
                   label=f"best (epoch {best_epoch})")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def build_charts(history: list[dict], best_epoch: int | None) -> dict[str, str]:
    if not history:
        return {}
    epochs = [h["epoch"] for h in history]

    def col(name):
        return [h.get(name) for h in history]

    charts = {}
    charts["loss"] = _line_chart(
        epochs,
        [("train", col("train_loss"), "-"), ("val", col("val_loss"), "-")],
        "Total loss", "loss", logy=True, best_epoch=best_epoch,
    )
    charts["fa_mae"] = _line_chart(
        epochs,
        [("train", col("train_fa_mae"), "-"), ("val", col("val_fa_mae"), "-")],
        "FA mean absolute error", "FA MAE", best_epoch=best_epoch,
    )
    charts["md_mae"] = _line_chart(
        epochs,
        [("train", col("train_md_mae"), "-"), ("val", col("val_md_mae"), "-")],
        "MD mean absolute error", "MD MAE", best_epoch=best_epoch,
    )
    charts["lr"] = _line_chart(
        epochs, [("learning rate", col("lr"), "-")],
        "Learning-rate schedule", "lr",
    )
    return charts


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────
def _esc(value) -> str:
    return html.escape(str(value))


def _kv_table(rows: list[tuple[str, object]]) -> str:
    body = "\n".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{body}</table>"


def _pipeline_section(config: dict) -> str:
    """Narrative of what happens to the data, grounded in this run's config."""
    max_n = config.get("max_n", "N")
    hw = config.get("canonical_hw", ["H", "W"])
    feat_dim = config.get("feat_dim", cfg.FEAT_DIM)
    channels = config.get("channels", cfg.UNET_CHANNELS)
    cholesky = config.get("cholesky", True)
    dti_scale = config.get("dti_scale")

    noise_dist = getattr(cfg, "NOISE_DISTRIBUTION", "rician")
    noise_lo, noise_hi = cfg.NOISE_MIN, cfg.NOISE_MAX
    keep_lo, keep_hi = cfg.KEEP_FRACTION_MIN, cfg.KEEP_FRACTION_MAX

    head = "L @ Lᵀ (Cholesky → guaranteed PSD)" if cholesky else "direct 6 channels"
    scale_str = f"{dti_scale:.3g}" if isinstance(dti_scale, (int, float)) else "n/a"

    steps = [
        ("1 · Dataset build",
         "build_dataset.py → preprocessing.py",
         "NIfTI DWI volumes are loaded; a brain mask is computed from the mean b0 "
         "with DIPY <code>median_otsu</code>; a diffusion tensor is fit per voxel "
         f"with <code>TensorModel</code> ({cfg.DTI_FIT_METHOD}). Only <b>clean</b> "
         "targets are written to the Zarr store: <code>target_dwi (X,Y,Z,N)</code>, "
         "<code>target_dti_6d (X,Y,Z,6)</code>, <code>brain_mask</code>, "
         "<code>bvals</code>, <code>bvecs</code>. No noisy data is stored."),
        ("2 · On-the-fly degradation",
         "dataset.py + augment.py",
         f"Each epoch every 2D slice is corrupted fresh: a central k-space cutout "
         f"keeps a random fraction ∈ [{keep_lo}, {keep_hi}] of low frequencies "
         f"(blurring), then <b>{noise_dist}</b> magnitude noise at relative level "
         f"∈ [{noise_lo}, {noise_hi}] is added. The input is normalised by its "
         "own mean-b0. Training augmentations: physical-mirror flips (signal + "
         "tensor + bvecs flipped together), intensity jitter, per-volume dropout, "
         "and random gradient-direction masking."),
        ("3 · Q-space encoder",
         "model.py · QSpaceEncoder",
         f"The variable-N stack of degraded volumes is aggregated into "
         f"<code>{feat_dim}</code> feature maps by a learned, gradient-conditioned "
         "weighted sum (a per-volume MLP embedding of <code>(bval, bvec)</code>). "
         "This is permutation-invariant and independent of how many volumes the "
         f"subject has or of padding up to <code>max_n={max_n}</code>."),
        ("4 · 2D U-Net head",
         "model.py · UNet2D",
         f"A 2D U-Net with encoder channels <code>{list(channels)}</code> maps the "
         f"feature maps to a <code>(6, {hw[0]}, {hw[1]})</code> output, "
         f"parameterised as {head}. Output order: "
         "<code>[Dxx, Dxy, Dyy, Dxz, Dyz, Dzz]</code>."),
        ("5 · Loss",
         "loss.py · DTILoss",
         "Brain-masked Charbonnier penalty on the 6 tensor channels, plus FA and "
         f"MD mean-absolute-error (weight λ={config.get('lambda_scalar', cfg.LAMBDA_SCALAR)}) "
         f"and an FA spatial-gradient/edge term (λ={config.get('lambda_edge', cfg.LAMBDA_EDGE)}). "
         "FA/MD are computed from Frobenius norms (no eigendecomposition) so the "
         f"loss runs on MPS. Targets are scaled by dti_scale={scale_str}."),
        ("6 · Optimisation",
         "train.py",
         f"AdamW (lr={config.get('lr')}, weight decay={config.get('weight_decay')}) "
         f"with {config.get('warmup_epochs')}-epoch linear warmup then cosine "
         f"annealing, gradient-norm clipping, and early stopping "
         f"(patience={config.get('patience')}). The train/val/test split is by "
         "biological subject so no subject leaks across splits."),
    ]
    cards = "\n".join(
        f"<div class='step'><div class='step-head'><span class='step-title'>{_esc(t)}</span>"
        f"<span class='step-file'>{_esc(f)}</span></div><p>{body}</p></div>"
        for t, f, body in steps
    )
    return f"<div class='steps'>{cards}</div>"


def build_html(config: dict, history: list[dict], run_dir: Path) -> str:
    best = min(history, key=lambda h: h["val_loss"]) if history else None
    best_epoch = best["epoch"] if best else None
    charts = build_charts(history, best_epoch)

    n_params = config.get("n_params")
    n_params_str = f"{n_params:,}" if isinstance(n_params, int) else _esc(n_params)

    summary_rows = [
        ("Run directory", config.get("out_dir", str(run_dir))),
        ("Device", config.get("device")),
        ("Epochs trained", f"{len(history)} / {config.get('epochs')}"),
        ("Best val loss",
         f"{best['val_loss']:.6f} @ epoch {best_epoch}" if best else "n/a"),
        ("Parameters", n_params_str),
        ("Trainable slices (train / val)",
         f"{config.get('train_slices')} / {config.get('val_slices')}"),
    ]

    arch_rows = [
        ("max_n (padded volumes)", config.get("max_n")),
        ("canonical H×W", "×".join(str(x) for x in config.get("canonical_hw", []))),
        ("feat_dim", config.get("feat_dim")),
        ("U-Net channels", config.get("channels")),
        ("Cholesky (PSD)", config.get("cholesky")),
        ("dti_scale", config.get("dti_scale")),
        ("max_bval", config.get("max_bval")),
    ]

    train_rows = [
        ("Batch size", config.get("batch_size")),
        ("Learning rate", config.get("lr")),
        ("Weight decay", config.get("weight_decay")),
        ("Warmup epochs", config.get("warmup_epochs")),
        ("Patience", config.get("patience")),
        ("λ scalar (FA/MD)", config.get("lambda_scalar")),
        ("λ edge (FA grad)", config.get("lambda_edge")),
        ("AMP", f"{config.get('amp')} ({config.get('amp_dtype')})"),
        ("torch.compile", config.get("compile_enabled")),
        ("Brain mask", config.get("use_brain_mask")),
    ]

    def subj_list(key):
        return ", ".join(config.get(key, [])) or "—"

    split_rows = [
        ("Train subjects", subj_list("train_subjects")),
        ("Val subjects", subj_list("val_subjects")),
        ("Test subjects", subj_list("test_subjects")),
    ]

    def chart_img(key, alt):
        if key not in charts:
            return ""
        return f"<figure><img src='{charts[key]}' alt='{_esc(alt)}'></figure>"

    charts_html = (
        "<div class='charts'>"
        + chart_img("loss", "training and validation loss")
        + chart_img("fa_mae", "FA MAE")
        + chart_img("md_mae", "MD MAE")
        + chart_img("lr", "learning rate")
        + "</div>"
        if charts
        else "<p class='empty'>No history.json found — run has no recorded epochs yet.</p>"
    )

    pipeline_html = _pipeline_section(config)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DWI → DTI training report — {_esc(run_dir.name)}</title>
<style>
  :root {{ --bg:#0f1117; --card:#181b24; --line:#262b38; --txt:#e6e8ee;
          --muted:#9aa3b2; --accent:#5b9dff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:36px 0 14px; border-bottom:1px solid var(--line);
        padding-bottom:6px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; }}
  code {{ background:#0b0d13; padding:1px 5px; border-radius:4px; font-size:.9em;
          color:#cdd6f4; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
           gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:16px 18px; }}
  .card h3 {{ margin:0 0 10px; font-size:13px; text-transform:uppercase;
              letter-spacing:.06em; color:var(--muted); }}
  table.kv {{ width:100%; border-collapse:collapse; }}
  table.kv th {{ text-align:left; font-weight:500; color:var(--muted);
                 padding:4px 10px 4px 0; vertical-align:top; white-space:nowrap; }}
  table.kv td {{ text-align:right; padding:4px 0; font-variant-numeric:tabular-nums;
                 word-break:break-word; }}
  .steps {{ display:flex; flex-direction:column; gap:12px; }}
  .step {{ background:var(--card); border:1px solid var(--line);
           border-left:3px solid var(--accent); border-radius:8px; padding:14px 16px; }}
  .step-head {{ display:flex; justify-content:space-between; align-items:baseline;
                gap:12px; margin-bottom:6px; flex-wrap:wrap; }}
  .step-title {{ font-weight:600; }}
  .step-file {{ color:var(--muted); font-size:12px; font-family:ui-monospace,monospace; }}
  .step p {{ margin:0; color:#c8ceda; }}
  .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
             gap:16px; }}
  figure {{ margin:0; background:#fff; border-radius:8px; padding:6px; }}
  figure img {{ width:100%; display:block; }}
  .empty {{ color:var(--muted); }}
  footer {{ margin-top:48px; color:var(--muted); font-size:12px;
            border-top:1px solid var(--line); padding-top:16px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>DWI → DTI training report</h1>
  <p class="sub">Run <code>{_esc(run_dir.name)}</code> · supervised denoising of
     diffusion-weighted images into 6-component diffusion tensors.</p>

  <h2>Run summary</h2>
  <div class="card">{_kv_table(summary_rows)}</div>

  <h2>What happens to the data</h2>
  {pipeline_html}

  <h2>Training curves</h2>
  {charts_html}

  <h2>Configuration</h2>
  <div class="grid">
    <div class="card"><h3>Architecture</h3>{_kv_table(arch_rows)}</div>
    <div class="card"><h3>Optimisation</h3>{_kv_table(train_rows)}</div>
    <div class="card"><h3>Subject split</h3>{_kv_table(split_rows)}</div>
  </div>

  <footer>Generated by report.py from
     <code>{_esc(run_dir.name)}/config.json</code> and
     <code>history.json</code>.</footer>
</div>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a self-contained HTML report for a training run."
    )
    parser.add_argument("--run_dir", default=cfg.TRAIN_OUT_DIR,
                        help="Run directory containing config.json / history.json.")
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default: <run_dir>/report.html).")
    return parser


def main(args: argparse.Namespace) -> None:
    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() \
        else Path(args.run_dir)
    config, history = load_run(run_dir)
    out_path = Path(args.out) if args.out else run_dir / "report.html"
    out_path.write_text(build_html(config, history, run_dir), encoding="utf-8")
    print(f"Wrote {out_path}  ({len(history)} epochs)")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
