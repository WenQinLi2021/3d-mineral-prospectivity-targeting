from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import LOG, mark_complete, output_dir, stage_dir, write_json


def _plot_pv(modeling_dir: Path, figure_dir: Path) -> Path:
    table = pd.read_csv(modeling_dir / "pv_aggregate.csv")
    path = figure_dir / "pv_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kappa, group in table.groupby("kappa"):
        label = f"kappa={kappa:g}"
        axes[0].plot(group["volume_rate"], group["H_all"], marker="o", label=label)
        axes[0].plot(
            group["volume_rate"], group["H_strong"], linestyle="--", alpha=0.7
        )
        axes[1].plot(group["volume_rate"], group["Lift_all"], marker="o", label=label)
    axes[0].plot([0, 0.30], [0, 0.30], color="black", linestyle=":", label="V")
    axes[0].set(xlabel="Volume rate", ylabel="Recovery rate", title="Prediction-Volume")
    axes[1].set(xlabel="Volume rate", ylabel="Lift", title="Volume enrichment")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _authoritative_target_summary(target_dir: Path) -> Path:
    corrected = target_dir / "target_summary_corrected.csv"
    return corrected if corrected.exists() else target_dir / "target_summary.csv"


def _plot_targets(target_dir: Path, figure_dir: Path) -> list[Path]:
    summary_path = _authoritative_target_summary(target_dir)
    summary = pd.read_csv(summary_path)
    paths: list[Path] = []
    if summary.empty:
        return paths
    top = summary.nsmallest(min(20, len(summary)), "rank").sort_values("rank", ascending=False)
    path = figure_dir / "target_ranking.png"
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["target_id"].astype(str), top["total_score"], color="#287D8E")
    axis.set(xlabel="Total score", ylabel="Target ID", title="Top ranked targets")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figure_dir / "target_depth_volume.png"
    fig, axis = plt.subplots(figsize=(9, 6))
    relative_top_elevation = summary["z_top"].max() - summary["z_top"]
    scatter = axis.scatter(
        relative_top_elevation,
        summary["volume_m3"],
        c=summary["mean_score"],
        s=20,
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_yscale("log")
    axis.set(
        xlabel="Relative top-elevation offset (m)",
        ylabel="Volume (m3)",
        title="Relative vertical position and candidate volume",
    )
    fig.colorbar(scatter, ax=axis, label="Mean score")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_sensitivity(target_dir: Path, figure_dir: Path) -> Path:
    table = pd.read_csv(target_dir / "sensitivity_summary.csv")
    path = figure_dir / "sensitivity_targets.png"
    pivot = table.pivot_table(
        index="case_id", columns="min_cells", values="n_targets", aggfunc="first"
    )
    axis = pivot.plot(kind="bar", figsize=(11, 6))
    axis.set(xlabel="Case", ylabel="Target count", title="Parameter sensitivity")
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def run_reporting(cfg: dict[str, Any]) -> dict[str, Path]:
    out = stage_dir(cfg, "03_report")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    modeling_dir = output_dir(cfg) / "01_modeling"
    target_dir = output_dir(cfg) / "02_targets"
    target_summary = _authoritative_target_summary(target_dir)
    generated = [_plot_pv(modeling_dir, figures)]
    generated.extend(_plot_targets(target_dir, figures))
    generated.append(_plot_sensitivity(target_dir, figures))

    bundle = out / "analysis_bundle.zip"
    small_files = [
        modeling_dir / "pv_folds.csv",
        modeling_dir / "pv_aggregate.csv",
        modeling_dir / "threshold_summary.csv",
        target_summary,
        target_dir / "sensitivity_summary.csv",
        target_dir / "persistence.csv",
        output_dir(cfg) / "00_audit" / "audit_report.json",
        output_dir(cfg) / "00_audit" / "fold_counts.csv",
        *generated,
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in small_files:
            if path.exists():
                arcname = (
                    "02_targets/target_summary.csv"
                    if path == target_summary
                    else str(path.relative_to(output_dir(cfg)))
                )
                archive.write(path, arcname=arcname)
    manifest = {
        "analysis_bundle": str(bundle),
        "target_grid": str(target_dir / "target_id_grid.parquet"),
        "robust_scores": str(modeling_dir / "robust_scores.parquet"),
        "figures": [str(path) for path in generated],
    }
    write_json(out / "result_manifest.json", manifest)
    mark_complete(out / "_SUCCESS.json", manifest)
    LOG.info("Analysis bundle ready: %s", bundle)
    return {"analysis_bundle": bundle, "manifest": out / "result_manifest.json"}
