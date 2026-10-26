from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage

from .common import (
    LOG,
    completed,
    mark_complete,
    open_npy,
    stage_dir,
    write_json,
)


def _prepare_grid_arrays(
    cfg: dict[str, Any], robust_path: Path, n_rows: int, out: Path
) -> tuple[Path, Path, Path]:
    paths = tuple(out / f"{name}.npy" for name in ("ix", "iy", "iz"))
    if all(path.exists() for path in paths):
        return paths  # type: ignore[return-value]
    arrays = [open_npy(path, "w+", shape=(n_rows,)) for path in paths]
    offset = 0
    batch_size = int(cfg["runtime"]["batch_size"])
    for batch in pq.ParquetFile(robust_path).iter_batches(
        batch_size=batch_size, columns=["ix", "iy", "iz"]
    ):
        keep = min(batch.num_rows, n_rows - offset)
        if keep <= 0:
            break
        for idx, name in enumerate(("ix", "iy", "iz")):
            arrays[idx][offset : offset + keep] = (
                batch.column(name).slice(0, keep).to_numpy().astype(np.float32)
            )
        offset += keep
    if offset != n_rows:
        raise RuntimeError(f"Grid array row mismatch: {offset} != {n_rows}")
    for array in arrays:
        array.flush()
    return paths  # type: ignore[return-value]


def _work_score(
    cfg: dict[str, Any], modeling_dir: Path, target_dir: Path, kappa: float, n_rows: int
) -> Path:
    path = target_dir / f"score_kappa_{kappa:g}.npy"
    if path.exists():
        return path
    mu = np.load(modeling_dir / "mu.npy", mmap_mode="r")
    uncertainty = np.load(modeling_dir / "u.npy", mmap_mode="r")
    score = open_npy(path, "w+", shape=(n_rows,))
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        score[start:stop] = mu[start:stop] - kappa * uncertainty[start:stop]
    score.flush()
    return path


def _select_top_rows(score: np.ndarray, volume_rate: float) -> tuple[np.ndarray, float]:
    """Return exactly ceil(volume_rate * n) rows with deterministic tie handling."""
    n_rows = len(score)
    n_selected = int(np.ceil(n_rows * volume_rate))
    if not 0 < n_selected <= n_rows:
        raise ValueError(f"Invalid volume rate {volume_rate!r}")
    # np.partition finds the cutoff without sorting all 154 million rows.
    cutoff_index = n_rows - n_selected
    threshold = float(np.partition(np.asarray(score), cutoff_index)[cutoff_index])
    above = np.flatnonzero(score > threshold)
    remaining = n_selected - len(above)
    if remaining < 0:
        raise RuntimeError("Top-volume selection produced too many strict-above rows")
    # np.flatnonzero is in ascending row-id order, which deterministically
    # resolves any equal-score tie at the cutoff.
    tied = np.flatnonzero(score == threshold)
    if len(tied) < remaining:
        raise RuntimeError("Top-volume selection could not fill the requested rows")
    selected = np.concatenate((above, tied[:remaining])).astype(np.int64, copy=False)
    if len(selected) != n_selected:
        raise RuntimeError("Top-volume selection count mismatch")
    return selected, threshold


def _fill_binary(
    cfg: dict[str, Any],
    coordinate_paths: tuple[Path, Path, Path],
    selected_rows: np.ndarray,
    shape: tuple[int, int, int],
    binary_path: Path,
) -> np.memmap:
    binary = np.memmap(binary_path, dtype=np.bool_, mode="w+", shape=shape)
    binary[:] = False
    ix = np.load(coordinate_paths[0], mmap_mode="r")
    iy = np.load(coordinate_paths[1], mmap_mode="r")
    iz = np.load(coordinate_paths[2], mmap_mode="r")
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    for start in range(0, len(selected_rows), chunk):
        rows = selected_rows[start : start + chunk]
        binary[
            np.asarray(iz[rows], dtype=np.int32),
            np.asarray(iy[rows], dtype=np.int32),
            np.asarray(ix[rows], dtype=np.int32),
        ] = True
    binary.flush()
    return binary


def _component_case(
    cfg: dict[str, Any],
    case_id: str,
    score_path: Path,
    coordinate_paths: tuple[Path, Path, Path],
    shape: tuple[int, int, int],
    volume_rate: float,
    connectivity: int,
    closing: bool,
    out: Path,
) -> dict[str, Any]:
    case_dir = out / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists() and cfg.get("runtime", {}).get("resume", True):
        metadata = json.load(metadata_path.open("r", encoding="utf-8"))
        if Path(metadata["labels_path"]).exists():
            return metadata

    score = np.load(score_path, mmap_mode="r")
    selected_rows, threshold = _select_top_rows(score, volume_rate)
    binary_path = case_dir / "binary.dat"
    labels_path = case_dir / "labels.dat"
    binary = _fill_binary(
        cfg, coordinate_paths, selected_rows, shape, binary_path
    )
    if closing:
        closed_path = case_dir / "closed.dat"
        closed = np.memmap(closed_path, dtype=np.bool_, mode="w+", shape=shape)
        structure_xy = np.ones((1, 3, 3), dtype=np.bool_)
        ndimage.binary_closing(binary, structure=structure_xy, iterations=1, output=closed)
        closed.flush()
        del binary
        binary_path.unlink(missing_ok=True)
        binary = closed
        binary_path = closed_path

    conn_map = {6: 1, 18: 2, 26: 3}
    structure = ndimage.generate_binary_structure(3, conn_map[connectivity])
    labels = np.memmap(labels_path, dtype=np.int32, mode="w+", shape=shape)
    n_raw = int(ndimage.label(binary, structure=structure, output=labels))
    labels.flush()
    counts = np.bincount(labels.reshape(-1))
    counts_path = case_dir / "component_counts.npy"
    np.save(counts_path, counts)
    min_values = [int(x) for x in cfg["targets"]["min_component_cells"]]
    metrics: dict[str, Any] = {}
    candidate_cells = int(len(selected_rows))
    for minimum in min_values:
        kept = counts[1:] >= minimum
        kept_counts = counts[1:][kept]
        metrics[str(minimum)] = {
            "n_targets": int(kept.sum()),
            "kept_cells": int(kept_counts.sum()),
            "fragment_ratio": float(
                1.0 - kept_counts.sum() / candidate_cells if candidate_cells else 0.0
            ),
            "mean_target_cells": float(kept_counts.mean()) if len(kept_counts) else 0.0,
            "median_target_cells": float(np.median(kept_counts))
            if len(kept_counts)
            else 0.0,
        }
    metadata = {
        "case_id": case_id,
        "volume_rate": float(volume_rate),
        "score_threshold": threshold,
        "selected_cells": candidate_cells,
        "connectivity": connectivity,
        "closing": closing,
        "n_raw_targets": n_raw,
        "candidate_cells": candidate_cells,
        "shape": list(shape),
        "binary_path": str(binary_path),
        "labels_path": str(labels_path),
        "counts_path": str(counts_path),
        "metrics": metrics,
    }
    write_json(metadata_path, metadata)
    del selected_rows, labels, binary, score
    gc.collect()
    return metadata


def _remap_for_minimum(case: dict[str, Any], minimum: int) -> np.ndarray:
    counts = np.load(case["counts_path"])
    remap = np.zeros(len(counts), dtype=np.int32)
    kept = np.flatnonzero(counts >= minimum)
    kept = kept[kept != 0]
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    return remap


def _target_counts(case: dict[str, Any], remap: np.ndarray) -> np.ndarray:
    raw = np.load(case["counts_path"])
    result = np.zeros(int(remap.max()) + 1, dtype=np.int64)
    kept = np.flatnonzero(remap)
    result[remap[kept]] = raw[kept]
    return result


def _persistence_iou(
    cfg: dict[str, Any],
    main_case: dict[str, Any],
    alt_case: dict[str, Any],
    coordinate_paths: tuple[Path, Path, Path],
    minimum: int,
) -> dict[int, float]:
    shape = tuple(int(x) for x in main_case["shape"])
    main_labels = np.memmap(
        main_case["labels_path"], dtype=np.int32, mode="r", shape=shape
    )
    alt_labels = np.memmap(
        alt_case["labels_path"], dtype=np.int32, mode="r", shape=shape
    )
    main_remap = _remap_for_minimum(main_case, minimum)
    alt_remap = _remap_for_minimum(alt_case, minimum)
    main_counts = _target_counts(main_case, main_remap)
    alt_counts = _target_counts(alt_case, alt_remap)
    ix = np.load(coordinate_paths[0], mmap_mode="r")
    iy = np.load(coordinate_paths[1], mmap_mode="r")
    iz = np.load(coordinate_paths[2], mmap_mode="r")
    intersections: dict[int, int] = {}
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    for start in range(0, len(ix), chunk):
        stop = min(start + chunk, len(ix))
        z = np.asarray(iz[start:stop], dtype=np.int32)
        y = np.asarray(iy[start:stop], dtype=np.int32)
        x = np.asarray(ix[start:stop], dtype=np.int32)
        main_id = main_remap[main_labels[z, y, x]]
        alt_id = alt_remap[alt_labels[z, y, x]]
        valid = (main_id > 0) & (alt_id > 0)
        if not valid.any():
            continue
        codes = (main_id[valid].astype(np.uint64) << np.uint64(32)) | alt_id[
            valid
        ].astype(np.uint64)
        unique, counts = np.unique(codes, return_counts=True)
        for code, count in zip(unique.tolist(), counts.tolist()):
            intersections[int(code)] = intersections.get(int(code), 0) + int(count)

    best: dict[int, float] = {idx: 0.0 for idx in range(1, len(main_counts))}
    for code, intersection in intersections.items():
        main_id = code >> 32
        alt_id = code & 0xFFFFFFFF
        union = main_counts[main_id] + alt_counts[alt_id] - intersection
        iou = intersection / union if union else 0.0
        if iou > best[main_id]:
            best[main_id] = float(iou)
    return best


def _write_target_grid(
    cfg: dict[str, Any],
    robust_path: Path,
    score_path: Path,
    uncertainty_path: Path,
    main_case: dict[str, Any],
    minimum: int,
    destination: Path,
) -> None:
    if destination.exists() and cfg.get("runtime", {}).get("resume", True):
        return
    shape = tuple(int(x) for x in main_case["shape"])
    labels = np.memmap(
        main_case["labels_path"], dtype=np.int32, mode="r", shape=shape
    )
    remap = _remap_for_minimum(main_case, minimum)
    score = np.load(score_path, mmap_mode="r")
    uncertainty = np.load(uncertainty_path, mmap_mode="r")
    writer: pq.ParquetWriter | None = None
    offset = 0
    columns = ["row_id", "index", "x", "y", "z", "ix", "iy", "iz", "label"]
    try:
        for batch in pq.ParquetFile(robust_path).iter_batches(
            batch_size=int(cfg["runtime"]["batch_size"]), columns=columns
        ):
            frame = batch.to_pandas()
            n = len(frame)
            z = frame["iz"].to_numpy(np.int32)
            y = frame["iy"].to_numpy(np.int32)
            x = frame["ix"].to_numpy(np.int32)
            target_id = remap[labels[z, y, x]]
            frame["score"] = np.asarray(score[offset : offset + n], dtype=np.float32)
            frame["uncertainty"] = np.asarray(
                uncertainty[offset : offset + n], dtype=np.float32
            )
            frame["target_id"] = target_id
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            offset += n
    finally:
        if writer is not None:
            writer.close()


def _summarize_targets(
    cfg: dict[str, Any],
    grid_path: Path,
    persistence: dict[int, float],
    output_path: Path,
) -> pd.DataFrame:
    grid_literal = str(grid_path).replace("'", "''")
    connection = duckdb.connect()
    core_threshold = float(
        connection.execute(
            f"SELECT quantile_cont(score, 0.95) FROM read_parquet('{grid_literal}')"
        ).fetchone()[0]
    )
    query = f"""
        SELECT
            target_id,
            COUNT(*)::BIGINT AS n_cells,
            MIN(ix) AS ix_min, MAX(ix) AS ix_max,
            MIN(iy) AS iy_min, MAX(iy) AS iy_max,
            MIN(iz) AS iz_min, MAX(iz) AS iz_max,
            MIN(x) AS x_min, MAX(x) AS x_max,
            MIN(y) AS y_min, MAX(y) AS y_max,
            MAX(z) AS z_top, MIN(z) AS z_bottom,
            AVG(score) AS mean_score,
            quantile_cont(score, 0.9) AS p90_score,
            MAX(score) AS max_score,
            AVG(uncertainty) AS mean_uncertainty,
            quantile_cont(uncertainty, 0.9) AS p90_uncertainty,
            AVG(CASE WHEN score >= {core_threshold} THEN 1.0 ELSE 0.0 END) AS core_ratio,
            COUNT(DISTINCT iz)::BIGINT AS n_layers,
            COUNT(DISTINCT (CAST(ix AS BIGINT) * 1000000 + iy))::BIGINT AS n_xy,
            SUM(CASE WHEN label IN (0, 1) THEN 1 ELSE 0 END)::BIGINT AS n_positive_cells,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)::BIGINT AS n_strong_positive_cells
        FROM read_parquet('{grid_literal}')
        WHERE target_id > 0
        GROUP BY target_id
        ORDER BY target_id
    """
    frame = connection.execute(query).df()
    connection.close()
    dx, dy, dz = (
        float(cfg["grid"]["dx"]),
        float(cfg["grid"]["dy"]),
        float(cfg["grid"]["dz"]),
    )
    frame["volume_m3"] = frame["n_cells"] * dx * dy * dz
    frame["area_xy_m2"] = frame["n_xy"] * dx * dy
    frame["thickness_m"] = frame["z_top"] - frame["z_bottom"] + dz
    bbox_cells = (
        (frame["ix_max"] - frame["ix_min"] + 1)
        * (frame["iy_max"] - frame["iy_min"] + 1)
        * (frame["iz_max"] - frame["iz_min"] + 1)
    )
    frame["bbox_fill"] = frame["n_cells"] / bbox_cells
    frame["persist_iou"] = frame["target_id"].map(persistence).fillna(0.0)

    def normalize(series: pd.Series) -> pd.Series:
        low, high = float(series.min()), float(series.max())
        if high <= low:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - low) / (high - low)

    normalized = pd.DataFrame(
        {
            "mean_score": normalize(frame["mean_score"]),
            "p90_score": normalize(frame["p90_score"]),
            "core_ratio": normalize(frame["core_ratio"]),
            "persist_iou": normalize(frame["persist_iou"]),
            "bbox_fill": normalize(frame["bbox_fill"]),
            "stable_uncertainty": 1.0 - normalize(frame["mean_uncertainty"]),
        }
    )
    base_weights = np.asarray([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
    score = normalized.to_numpy() @ base_weights
    frame["total_score"] = score
    top_n = min(5, len(frame))
    stability_hits = np.zeros(len(frame), dtype=np.int32)
    scenarios = 0
    for weight_index in range(len(base_weights)):
        for delta in (-0.05, 0.05):
            weights = base_weights.copy()
            weights[weight_index] = max(0.0, weights[weight_index] + delta)
            weights /= weights.sum()
            scenario_score = normalized.to_numpy() @ weights
            top_indices = np.argpartition(scenario_score, -top_n)[-top_n:]
            stability_hits[top_indices] += 1
            scenarios += 1
    frame["parameter_stability"] = stability_hits / scenarios
    frame = frame.sort_values("total_score", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    rank_fraction = frame["rank"] / max(len(frame), 1)
    uncertainty_median = float(frame["mean_uncertainty"].median())
    score_median = float(frame["mean_score"].median())
    level = np.full(len(frame), "三级", dtype=object)
    level[rank_fraction <= 0.50] = "二级"
    level[(rank_fraction <= 0.20) & (frame["mean_uncertainty"] <= uncertainty_median)] = "一级"
    level[(frame["mean_score"] >= score_median) & (frame["mean_uncertainty"] > uncertainty_median)] = "验证型"
    frame["level"] = level
    action_map = {
        "一级": "优先核查",
        "二级": "补充资料后评价",
        "三级": "保留观察",
        "验证型": "重点验证不确定性",
    }
    frame["recommended_action"] = frame["level"].map(action_map)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def run_targets(
    cfg: dict[str, Any], audit: dict[str, Any], modeling: dict[str, Path]
) -> dict[str, Path]:
    out = stage_dir(cfg, "02_targets")
    done = out / "_SUCCESS.json"
    summary_path = out / "target_summary.csv"
    grid_path = out / "target_id_grid.parquet"
    if completed(done, cfg):
        return {"target_summary": summary_path, "target_grid": grid_path}

    n_rows = int(audit["processed_rows"])
    shape = tuple(int(x) for x in audit["grid_shape_zyx"])
    robust_path = Path(modeling["robust_scores"])
    coordinate_paths = _prepare_grid_arrays(cfg, robust_path, n_rows, out)
    modeling_dir = Path(modeling["mu"]).parent
    thresholds = pd.read_csv(modeling["thresholds"])
    main_kappa = float(cfg["score"]["main_kappa"])
    main_rule = cfg["validation"]["threshold_rule"]

    def selected_rate(kappa: float) -> float:
        match = thresholds[
            np.isclose(thresholds["kappa"], kappa) & (thresholds["rule"] == main_rule)
        ]
        if match.empty:
            raise ValueError(f"No selected threshold for kappa={kappa}")
        return float(match.iloc[0]["volume_rate"])

    main_rate = selected_rate(main_kappa)
    main_score = _work_score(cfg, modeling_dir, out, main_kappa, n_rows)
    main_case = _component_case(
        cfg,
        "main",
        main_score,
        coordinate_paths,
        shape,
        main_rate,
        int(cfg["targets"]["main_connectivity"]),
        False,
        out,
    )

    sensitivity: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {"main": main_case}
    for connectivity in cfg["targets"]["compare_connectivity"]:
        case_id = f"connectivity_{int(connectivity)}"
        if int(connectivity) == int(cfg["targets"]["main_connectivity"]):
            case = main_case
        else:
            case = _component_case(
                cfg,
                case_id,
                main_score,
                coordinate_paths,
                shape,
                main_rate,
                int(connectivity),
                False,
                out,
            )
        cases[case_id] = case

    # Use multiplicatively symmetric neighbours. An additive delta was clipped
    # at the 1% search boundary in the first run, which made persistence
    # asymmetric and artificially favoured the main case.
    persistence_ratio = float(cfg["targets"].get("persistence_ratio", 1.5))
    if persistence_ratio <= 1.0:
        raise ValueError("targets.persistence_ratio must be greater than 1")
    for name, rate in (
        ("v_minus", max(main_rate / persistence_ratio, 1e-6)),
        ("v_plus", min(main_rate * persistence_ratio, 0.99)),
    ):
        cases[name] = _component_case(
            cfg,
            name,
            main_score,
            coordinate_paths,
            shape,
            rate,
            int(cfg["targets"]["main_connectivity"]),
            False,
            out,
        )

    for kappa in cfg["score"]["kappas"]:
        kappa = float(kappa)
        if np.isclose(kappa, main_kappa):
            continue
        score_path = _work_score(cfg, modeling_dir, out, kappa, n_rows)
        cases[f"kappa_{kappa:g}"] = _component_case(
            cfg,
            f"kappa_{kappa:g}",
            score_path,
            coordinate_paths,
            shape,
            selected_rate(kappa),
            int(cfg["targets"]["main_connectivity"]),
            False,
            out,
        )

    if cfg["targets"].get("compare_closing", True):
        cases["closing_xy"] = _component_case(
            cfg,
            "closing_xy",
            main_score,
            coordinate_paths,
            shape,
            main_rate,
            int(cfg["targets"]["main_connectivity"]),
            True,
            out,
        )

    for case_id, case in cases.items():
        for minimum, values in case["metrics"].items():
            sensitivity.append(
                {
                    "case_id": case_id,
                    "volume_rate": case["volume_rate"],
                    "connectivity": case["connectivity"],
                    "closing": case["closing"],
                    "min_cells": int(minimum),
                    **values,
                }
            )
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity_summary.csv", index=False)

    minimum = int(cfg["targets"]["main_min_component_cells"])
    persist_minus = _persistence_iou(
        cfg, main_case, cases["v_minus"], coordinate_paths, minimum
    )
    persist_plus = _persistence_iou(
        cfg, main_case, cases["v_plus"], coordinate_paths, minimum
    )
    persistence = {
        target_id: (persist_minus.get(target_id, 0.0) + persist_plus.get(target_id, 0.0))
        / 2.0
        for target_id in set(persist_minus) | set(persist_plus)
    }
    pd.DataFrame(
        {"target_id": list(persistence), "persist_iou": list(persistence.values())}
    ).to_csv(out / "persistence.csv", index=False)

    _write_target_grid(
        cfg,
        robust_path,
        main_score,
        Path(modeling["u"]),
        main_case,
        minimum,
        grid_path,
    )
    _summarize_targets(cfg, grid_path, persistence, summary_path)

    if not cfg.get("runtime", {}).get("keep_sensitivity_labels", False):
        main_labels_path = Path(main_case["labels_path"])
        for case_id, case in cases.items():
            if Path(case["labels_path"]) == main_labels_path:
                continue
            Path(case["labels_path"]).unlink(missing_ok=True)
            Path(case["binary_path"]).unlink(missing_ok=True)

    mark_complete(
        done,
        {
            "target_summary": str(summary_path),
            "target_grid": str(grid_path),
            "main_volume_rate": main_rate,
        },
    )
    return {"target_summary": summary_path, "target_grid": grid_path}
