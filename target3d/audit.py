from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import (
    LOG,
    completed,
    feature_columns,
    iter_synced_batches,
    mark_complete,
    metadata_min_max,
    parquet_num_rows,
    resolve_path,
    stage_dir,
    write_json,
)


def _contiguous_fold_edges(
    values: np.ndarray, n_splits: int, balance_weights: np.ndarray | None = None
) -> list[float]:
    unique, inverse = np.unique(values, return_inverse=True)
    if balance_weights is None:
        counts = np.bincount(inverse)
    else:
        counts = np.bincount(inverse, weights=balance_weights)
        if counts.sum() == 0:
            counts = np.bincount(inverse)
    cumulative = np.cumsum(counts)
    targets = np.arange(1, n_splits) * counts.sum() / n_splits
    edges: list[float] = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(idx, len(unique) - 2)
        edges.append(float(unique[idx] + unique[idx + 1]) / 2.0)
    edges = sorted(set(edges))
    if len(edges) != n_splits - 1:
        raise ValueError("Not enough unique coordinates to create spatial folds")
    return edges


def assign_folds(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), values, side="right").astype(np.int8)


def run_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    out = stage_dir(cfg, "00_audit")
    done = out / "_SUCCESS.json"
    report_path = out / "audit_report.json"
    if completed(done, cfg):
        LOG.info("Audit already complete")
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    feature_path = resolve_path(cfg, cfg["paths"]["feature_parquet"])
    attr_path = resolve_path(cfg, cfg["paths"]["attribute_parquet"])
    for path in (feature_path, attr_path):
        if not path.exists():
            raise FileNotFoundError(path)

    feature_pf = pq.ParquetFile(feature_path)
    attr_pf = pq.ParquetFile(attr_path)
    feature_rows = parquet_num_rows(feature_path)
    attr_rows = parquet_num_rows(attr_path)
    if feature_rows != attr_rows:
        raise ValueError(f"Row count mismatch: {feature_rows} != {attr_rows}")

    required_feature = {"ix", "iy", "iz", "label", "label_xy"}
    missing = required_feature.difference(feature_pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Feature Parquet is missing columns: {sorted(missing)}")
    required_attr = {"index", "x", "y", "z", "label"}
    missing = required_attr.difference(attr_pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Attribute Parquet is missing columns: {sorted(missing)}")

    feat_cols = feature_columns(feature_pf.schema_arrow.names)
    if not feat_cols:
        raise ValueError("No convolution feature columns found")

    x_min, x_max = metadata_min_max(attr_path, "x")
    y_min, y_max = metadata_min_max(attr_path, "y")
    z_min, z_max = metadata_min_max(attr_path, "z")
    grid = cfg["grid"]
    dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])
    rebin_z = int(grid["model_rebin_z"])
    nx = int(round((x_max - x_min) / dx)) + 1
    ny = int(round((y_max - y_min) / dy)) + 1
    nz = int(round((z_max - z_min) / dz)) + 1

    runtime_rows = cfg.get("runtime", {}).get("max_rows")
    max_rows = min(feature_rows, int(runtime_rows)) if runtime_rows else feature_rows
    batch_size = int(cfg["runtime"]["batch_size"])
    full_alignment = bool(cfg.get("audit", {}).get("full_alignment_check", True))
    alignment_rows = max_rows if full_alignment else min(max_rows, batch_size * 3)

    positive_parts: list[pd.DataFrame] = []
    label_counts: dict[int, int] = {}
    label_xy_counts: dict[int, int] = {}
    duplicate_index_sample = 0

    LOG.info("Auditing row alignment and collecting validation positives")
    columns_left = ["ix", "iy", "iz", "label", "label_xy"]
    columns_right = ["index", "x", "y", "z", "label"]
    for left, right, offset in iter_synced_batches(
        feature_path,
        attr_path,
        columns_left,
        columns_right,
        batch_size,
        max_rows=max_rows,
    ):
        f = left.to_pandas()
        a = right.to_pandas()
        labels = a["label"].to_numpy(np.int8, copy=False)
        for key, value in zip(*np.unique(labels, return_counts=True)):
            label_counts[int(key)] = label_counts.get(int(key), 0) + int(value)
        labels_xy = f["label_xy"].to_numpy(np.int8, copy=False)
        for key, value in zip(*np.unique(labels_xy, return_counts=True)):
            label_xy_counts[int(key)] = label_xy_counts.get(int(key), 0) + int(value)

        mask = np.isin(labels, np.asarray(cfg["validation"]["positive_labels"]))
        if mask.any():
            rows = np.arange(offset, offset + len(a), dtype=np.int64)[mask]
            positive_parts.append(
                pd.DataFrame(
                    {
                        "row_id": rows,
                        "index": a.loc[mask, "index"].to_numpy(np.int64),
                        "ix": f.loc[mask, "ix"].to_numpy(np.int32),
                        "iy": f.loc[mask, "iy"].to_numpy(np.int32),
                        "iz_model": f.loc[mask, "iz"].to_numpy(np.int32),
                        "label": labels[mask],
                        "x": a.loc[mask, "x"].to_numpy(np.float64),
                        "y": a.loc[mask, "y"].to_numpy(np.float64),
                        "z": a.loc[mask, "z"].to_numpy(np.float64),
                    }
                )
            )

        if offset < alignment_rows:
            ix = np.rint((a["x"].to_numpy() - x_min) / dx).astype(np.int32)
            iy = np.rint((a["y"].to_numpy() - y_min) / dy).astype(np.int32)
            iz_raw = np.rint((z_max - a["z"].to_numpy()) / dz).astype(np.int32)
            iz_model = (iz_raw // rebin_z).astype(np.int32)
            if not np.array_equal(ix, f["ix"].to_numpy(np.int32)):
                raise ValueError(f"ix alignment failed near row {offset}")
            if not np.array_equal(iy, f["iy"].to_numpy(np.int32)):
                raise ValueError(f"iy alignment failed near row {offset}")
            if not np.array_equal(iz_model, f["iz"].to_numpy(np.int32)):
                raise ValueError(f"iz alignment failed near row {offset}")
            if not np.array_equal(labels, f["label"].to_numpy(np.int8)):
                raise ValueError(f"label alignment failed near row {offset}")
            duplicate_index_sample += int(a["index"].duplicated().sum())

    positives = pd.concat(positive_parts, ignore_index=True)
    n_splits = int(cfg["model"]["n_splits"])
    split_axis = cfg["model"].get("split_axis", "auto")
    if split_axis == "auto":
        x_span = float(positives["ix"].max() - positives["ix"].min())
        y_span = float(positives["iy"].max() - positives["iy"].min())
        split_axis = "ix" if x_span >= y_span else "iy"
    positives["is_strong"] = positives["label"].isin(
        cfg["validation"]["strong_positive_labels"]
    )
    # Preserve spatially contiguous folds while choosing band boundaries from
    # strong-positive density. The former implementation balanced all
    # mineralization points and could leave a validation fold with no high-grade
    # samples at all.
    try:
        edges = _contiguous_fold_edges(
            positives[split_axis].to_numpy(),
            n_splits,
            positives["is_strong"].to_numpy(np.float64),
        )
    except ValueError:
        LOG.warning(
            "Strong positives cannot form %d contiguous spatial folds; "
            "falling back to all mineralization points for fold boundaries",
            n_splits,
        )
        edges = _contiguous_fold_edges(
            positives[split_axis].to_numpy(), n_splits
        )
    positives["fold"] = assign_folds(positives[split_axis].to_numpy(), edges)
    positives.to_csv(out / "validation_positives.csv", index=False, encoding="utf-8-sig")

    fold_counts = (
        positives.groupby("fold")
        .agg(n_positive=("row_id", "size"), n_strong=("is_strong", "sum"))
        .reset_index()
    )
    fold_counts.to_csv(out / "fold_counts.csv", index=False)

    report = {
        "feature_rows": feature_rows,
        "attribute_rows": attr_rows,
        "processed_rows": max_rows,
        "feature_count": len(feat_cols),
        "feature_columns": feat_cols,
        "label_counts": label_counts,
        "label_xy_counts": label_xy_counts,
        "bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        },
        "grid_shape_zyx": [nz, ny, nx],
        "theoretical_cells": int(nx * ny * nz),
        "observed_fraction": float(max_rows / (nx * ny * nz)),
        "dense_bool_gb": float(nx * ny * nz / 1e9),
        "dense_int32_gb": float(nx * ny * nz * 4 / 1e9),
        "split_axis": split_axis,
        "fold_edges": edges,
        "fold_counts": fold_counts.to_dict(orient="records"),
        "sample_duplicate_indices": duplicate_index_sample,
    }
    write_json(report_path, report)
    mark_complete(done, {"report": str(report_path)})
    LOG.info("Audit complete: %s", report_path)
    return report
