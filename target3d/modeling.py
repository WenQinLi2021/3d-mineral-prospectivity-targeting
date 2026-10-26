from __future__ import annotations

import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from .audit import assign_folds
from .common import (
    LOG,
    available_device,
    completed,
    mark_complete,
    open_npy,
    parquet_num_rows,
    read_json,
    resolve_path,
    stage_dir,
    write_json,
)
from .pv import aggregate_pv_tables, compute_pv_table, select_thresholds


def _fold_training_mask(
    axis_values: np.ndarray,
    fold: int,
    edges: list[float],
    buffer_cells: int,
) -> np.ndarray:
    lower = -np.inf if fold == 0 else edges[fold - 1]
    upper = np.inf if fold == len(edges) else edges[fold]
    lower -= buffer_cells
    upper += buffer_cells
    return (axis_values < lower) | (axis_values > upper)


def _sample_training_data(
    cfg: dict[str, Any],
    fold: int,
    seed: int,
    audit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    feature_path = resolve_path(cfg, cfg["paths"]["feature_parquet"])
    feature_cols = audit["feature_columns"]
    split_axis = audit["split_axis"]
    edges = [float(x) for x in audit["fold_edges"]]
    batch_size = int(cfg["runtime"]["batch_size"])
    max_rows = cfg.get("runtime", {}).get("max_rows")
    u_per_batch = int(cfg["model"]["u_sample_per_batch"])
    buffer_cells = int(cfg["model"].get("fold_buffer_cells", 4))
    rng = np.random.default_rng(seed + 1009 * fold)

    xs: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    counts: dict[int, int] = {}
    offset = 0
    parquet = pq.ParquetFile(feature_path)
    columns = [split_axis, "label_xy"] + feature_cols
    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns)
    ):
        if max_rows is not None and offset >= int(max_rows):
            break
        df = batch.to_pandas()
        if max_rows is not None and offset + len(df) > int(max_rows):
            df = df.iloc[: int(max_rows) - offset]
        axis = df[split_axis].to_numpy(np.int32, copy=False)
        train_region = _fold_training_mask(axis, fold, edges, buffer_cells)
        labels = df["label_xy"].to_numpy(np.int8, copy=False)
        important = train_region & np.isin(labels, [1, 0, -2])
        unlabeled = np.flatnonzero(train_region & (labels == -1))
        if len(unlabeled) > u_per_batch:
            unlabeled = rng.choice(unlabeled, size=u_per_batch, replace=False)
        keep = np.flatnonzero(important)
        if len(unlabeled):
            keep = np.concatenate([keep, unlabeled])
        if len(keep):
            selected_labels = labels[keep].copy()
            xs.append(df.iloc[keep][feature_cols].to_numpy(np.float32, copy=True))
            raw_labels.append(selected_labels)
            for key, value in zip(*np.unique(selected_labels, return_counts=True)):
                counts[int(key)] = counts.get(int(key), 0) + int(value)
        offset += len(df)
        if (batch_index + 1) % 25 == 0:
            LOG.info("Fold %d sampling batch %d", fold, batch_index + 1)

    if not xs:
        raise RuntimeError(f"Fold {fold} produced no training rows")
    x_train = np.concatenate(xs, axis=0)
    raw = np.concatenate(raw_labels)
    del xs, raw_labels
    y_train = (raw == 1).astype(np.int8)
    weight_map = {int(k): float(v) for k, v in cfg["model"]["weights"].items()}
    weights = np.asarray([weight_map[int(value)] for value in raw], dtype=np.float32)
    if y_train.sum() == 0:
        raise RuntimeError(f"Fold {fold} has no strong positive training rows")
    return x_train, y_train, weights, counts


def _fit_or_load_model(
    cfg: dict[str, Any],
    fold: int,
    seed: int,
    audit: dict[str, Any],
    model_path: Path,
    device: str,
) -> xgb.XGBClassifier:
    if model_path.exists() and cfg.get("runtime", {}).get("resume", True):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        LOG.info("Loaded existing model %s", model_path.name)
        return model

    x_train, y_train, weights, counts = _sample_training_data(
        cfg, fold, seed, audit
    )
    params = dict(cfg["model"]["xgboost"])
    params["random_state"] = seed
    params["device"] = device
    params["n_jobs"] = int(cfg["model"].get("n_jobs", -1))
    LOG.info(
        "Training fold=%d seed=%d rows=%s positives=%s device=%s labels=%s",
        fold,
        seed,
        f"{len(y_train):,}",
        f"{int(y_train.sum()):,}",
        device,
        counts,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    model.save_model(model_path)
    del x_train, y_train, weights
    gc.collect()
    return model


def _predict_model(
    cfg: dict[str, Any],
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    output_path: Path,
    n_rows: int,
) -> np.ndarray:
    scores = open_npy(output_path, "w+", shape=(n_rows,))
    feature_path = resolve_path(cfg, cfg["paths"]["feature_parquet"])
    batch_size = int(cfg["runtime"]["prediction_batch_size"])
    offset = 0
    for batch_index, batch in enumerate(
        pq.ParquetFile(feature_path).iter_batches(
            batch_size=batch_size, columns=feature_cols
        )
    ):
        if offset >= n_rows:
            break
        frame = batch.to_pandas()
        keep = min(len(frame), n_rows - offset)
        x = frame.iloc[:keep].to_numpy(np.float32, copy=False)
        scores[offset : offset + keep] = model.predict_proba(x)[:, 1].astype(
            np.float32
        )
        offset += keep
        if (batch_index + 1) % 25 == 0:
            scores.flush()
            LOG.info("Predicted %s / %s rows", f"{offset:,}", f"{n_rows:,}")
    scores.flush()
    if offset != n_rows:
        raise RuntimeError(f"Prediction row count mismatch: {offset} != {n_rows}")
    return scores


def _run_fold(
    cfg: dict[str, Any],
    fold: int,
    audit: dict[str, Any],
    positives: pd.DataFrame,
    device: str,
    out: Path,
) -> None:
    fold_dir = out / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done = fold_dir / "_SUCCESS.json"
    if completed(done, cfg):
        LOG.info("Fold %d already complete", fold)
        return

    n_rows = int(audit["processed_rows"])
    feature_cols = audit["feature_columns"]
    fold_sum = open_npy(fold_dir / "prediction_sum.npy", "w+", shape=(n_rows,))
    fold_sumsq = open_npy(
        fold_dir / "prediction_sumsq.npy", "w+", shape=(n_rows,)
    )
    fold_sum[:] = 0.0
    fold_sumsq[:] = 0.0
    seeds = [int(seed) for seed in cfg["model"]["seeds"]]

    for seed in seeds:
        model_path = fold_dir / f"model_seed_{seed}.json"
        temp_path = fold_dir / f"prediction_seed_{seed}.npy"
        model = _fit_or_load_model(
            cfg, fold, seed, audit, model_path, device
        )
        scores = _predict_model(
            cfg, model, feature_cols, temp_path, n_rows
        )
        chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            values = np.array(scores[start:stop], dtype=np.float32, copy=True)
            fold_sum[start:stop] += values
            fold_sumsq[start:stop] += values * values
        fold_sum.flush()
        fold_sumsq.flush()
        del values, scores, model
        gc.collect()
        temp_path.unlink(missing_ok=True)

    fold_mean = open_npy(fold_dir / "prediction_mean.npy", "w+", shape=(n_rows,))
    fold_u = open_npy(fold_dir / "prediction_u.npy", "w+", shape=(n_rows,))
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        mean = fold_sum[start:stop] / len(seeds)
        variance = np.maximum(fold_sumsq[start:stop] / len(seeds) - mean * mean, 0.0)
        fold_mean[start:stop] = mean
        fold_u[start:stop] = np.sqrt(variance)
    fold_mean.flush()
    fold_u.flush()

    validation = positives[positives["fold"] == fold].copy()
    volume_rates = cfg["validation"]["volume_rates"]
    pv_parts: list[pd.DataFrame] = []
    for kappa in cfg["score"]["kappas"]:
        work_path = fold_dir / f"work_kappa_{float(kappa):g}.npy"
        work = open_npy(work_path, "w+", shape=(n_rows,))
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            work[start:stop] = (
                fold_mean[start:stop] - float(kappa) * fold_u[start:stop]
            )
        work.flush()
        pv_parts.append(
            compute_pv_table(work, validation, volume_rates, float(kappa), fold)
        )
        del work
        work_path.unlink(missing_ok=True)
    pd.concat(pv_parts, ignore_index=True).to_csv(fold_dir / "pv_table.csv", index=False)
    write_json(fold_dir / "metadata.json", {"fold": fold, "seeds": seeds})
    mark_complete(done, {"fold": fold, "seeds": seeds})


def _aggregate_predictions(
    cfg: dict[str, Any], audit: dict[str, Any], out: Path
) -> tuple[Path, Path]:
    n_rows = int(audit["processed_rows"])
    n_splits = int(cfg["model"]["n_splits"])
    seeds = cfg["model"]["seeds"]
    total_models = n_splits * len(seeds)
    mu_path = out / "mu.npy"
    u_path = out / "u.npy"
    if completed(out / "_PREDICTIONS_SUCCESS.json", cfg):
        return mu_path, u_path
    mu = open_npy(mu_path, "w+", shape=(n_rows,))
    second = open_npy(out / "second_moment.npy", "w+", shape=(n_rows,))
    mu[:] = 0.0
    second[:] = 0.0
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    for fold in range(n_splits):
        fold_sum = np.load(out / f"fold_{fold}" / "prediction_sum.npy", mmap_mode="r")
        fold_sumsq = np.load(
            out / f"fold_{fold}" / "prediction_sumsq.npy", mmap_mode="r"
        )
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            mu[start:stop] += fold_sum[start:stop]
            second[start:stop] += fold_sumsq[start:stop]
    u = open_npy(u_path, "w+", shape=(n_rows,))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        mean = mu[start:stop] / total_models
        variance = np.maximum(second[start:stop] / total_models - mean * mean, 0.0)
        mu[start:stop] = mean
        u[start:stop] = np.sqrt(variance)
    mu.flush()
    u.flush()
    del second
    (out / "second_moment.npy").unlink(missing_ok=True)
    mark_complete(
        out / "_PREDICTIONS_SUCCESS.json", {"n_models": total_models}
    )
    return mu_path, u_path


def _write_robust_scores(
    cfg: dict[str, Any], audit: dict[str, Any], mu_path: Path, u_path: Path, out: Path
) -> Path:
    destination = out / "robust_scores.parquet"
    done = out / "_ROBUST_SCORES_SUCCESS.json"
    if completed(done, cfg):
        return destination
    attr_path = resolve_path(cfg, cfg["paths"]["attribute_parquet"])
    mu = np.load(mu_path, mmap_mode="r")
    uncertainty = np.load(u_path, mmap_mode="r")
    bounds = audit["bounds"]
    dx = float(cfg["grid"]["dx"])
    dy = float(cfg["grid"]["dy"])
    dz = float(cfg["grid"]["dz"])
    batch_size = int(cfg["runtime"]["batch_size"])
    n_rows = int(audit["processed_rows"])
    writer: pq.ParquetWriter | None = None
    offset = 0
    columns = ["index", "x", "y", "z", "label"]
    try:
        for batch in pq.ParquetFile(attr_path).iter_batches(
            batch_size=batch_size, columns=columns
        ):
            if offset >= n_rows:
                break
            frame = batch.to_pandas()
            keep = min(len(frame), n_rows - offset)
            frame = frame.iloc[:keep].copy()
            x = frame["x"].to_numpy()
            y = frame["y"].to_numpy()
            z = frame["z"].to_numpy()
            values_mu = np.asarray(mu[offset : offset + keep], dtype=np.float32)
            values_u = np.asarray(uncertainty[offset : offset + keep], dtype=np.float32)
            result = pd.DataFrame(
                {
                    "row_id": np.arange(offset, offset + keep, dtype=np.int64),
                    "index": frame["index"].to_numpy(np.int64),
                    "x": x.astype(np.float64),
                    "y": y.astype(np.float64),
                    "z": z.astype(np.float64),
                    "ix": np.rint((x - bounds["x_min"]) / dx).astype(np.int32),
                    "iy": np.rint((y - bounds["y_min"]) / dy).astype(np.int32),
                    # This must match prepare_features.py: index 0 is the
                    # shallowest layer (z_max), and z increases with depth.
                    "iz": np.rint((bounds["z_max"] - z) / dz).astype(np.int32),
                    "label": frame["label"].to_numpy(np.int8),
                    "mu": values_mu,
                    "uncertainty": values_u,
                    "score_k0": values_mu,
                    "score_k05": values_mu - 0.5 * values_u,
                    "score_k1": values_mu - values_u,
                }
            )
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            offset += keep
            if offset % (batch_size * 10) == 0:
                LOG.info("Wrote robust scores: %s rows", f"{offset:,}")
    finally:
        if writer is not None:
            writer.close()
    if offset != n_rows:
        raise RuntimeError(f"Robust score row mismatch: {offset} != {n_rows}")
    mark_complete(done, {"path": str(destination), "rows": n_rows})
    return destination


def run_modeling(cfg: dict[str, Any], audit: dict[str, Any]) -> dict[str, Path]:
    out = stage_dir(cfg, "01_modeling")
    positives = pd.read_csv(
        stage_dir(cfg, "00_audit") / "validation_positives.csv"
    )
    device = available_device(cfg["model"].get("device", "auto"))
    LOG.info("Selected XGBoost device: %s", device)
    for fold in range(int(cfg["model"]["n_splits"])):
        _run_fold(cfg, fold, audit, positives, device, out)

    fold_tables = [
        pd.read_csv(out / f"fold_{fold}" / "pv_table.csv")
        for fold in range(int(cfg["model"]["n_splits"]))
    ]
    fold_pv = pd.concat(fold_tables, ignore_index=True)
    fold_pv.to_csv(out / "pv_folds.csv", index=False)
    aggregate = aggregate_pv_tables(fold_pv)
    aggregate.to_csv(out / "pv_aggregate.csv", index=False)
    selected = select_thresholds(aggregate)
    selected.to_csv(out / "threshold_summary.csv", index=False)

    mu_path, u_path = _aggregate_predictions(cfg, audit, out)
    robust_path = _write_robust_scores(cfg, audit, mu_path, u_path, out)
    mark_complete(out / "_SUCCESS.json", {"robust_scores": str(robust_path)})
    return {
        "mu": mu_path,
        "u": u_path,
        "robust_scores": robust_path,
        "pv": out / "pv_aggregate.csv",
        "thresholds": out / "threshold_summary.csv",
    }
