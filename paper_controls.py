from __future__ import annotations

import argparse
import json
import math
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage
import xgboost as xgb


def _peak_memory(stop: threading.Event, state: dict[str, int]) -> None:
    try:
        import psutil
    except ImportError:
        return
    process = psutil.Process()
    while not stop.wait(0.2):
        state["peak_rss_bytes"] = max(state["peak_rss_bytes"], process.memory_info().rss)


def _metric(mask: np.ndarray, positives: pd.DataFrame, rate: float) -> dict[str, float | int]:
    rows = positives["row_id"].to_numpy(np.int64)
    strong = positives["is_strong"].to_numpy(bool)
    hit = np.asarray(mask[rows], dtype=bool)
    hits_all = int(hit.sum())
    hits_strong = int((hit & strong).sum())
    n_all = len(rows)
    n_strong = int(strong.sum())
    h_all = hits_all / n_all if n_all else float("nan")
    h_strong = hits_strong / n_strong if n_strong else float("nan")
    return {
        "n_positive": n_all,
        "n_strong": n_strong,
        "hits_all": hits_all,
        "hits_strong": hits_strong,
        "H_all": h_all,
        "H_strong": h_strong,
        "Lift_all": h_all / rate if rate else float("nan"),
        "Lift_strong": h_strong / rate if rate else float("nan"),
    }


def _exact_top_n(scores: np.ndarray, rate: float, selected_path: Path) -> tuple[np.ndarray, float, int]:
    n_rows = len(scores)
    n_cells = int(math.ceil(n_rows * rate))
    cutoff = float(np.quantile(scores, 1.0 - rate, method="higher"))
    selected = np.lib.format.open_memmap(selected_path, mode="w+", dtype=np.bool_, shape=(n_rows,))
    chunk = 5_000_000
    greater = 0
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        values = scores[start:stop]
        chosen = values > cutoff
        selected[start:stop] = chosen
        greater += int(chosen.sum())
    ties_needed = n_cells - greater
    if ties_needed < 0:
        raise RuntimeError("invalid exact Top-N cutoff")
    remaining = ties_needed
    for start in range(0, n_rows, chunk):
        if remaining == 0:
            break
        stop = min(start + chunk, n_rows)
        ties = np.flatnonzero(scores[start:stop] == cutoff)
        take = ties[:remaining]
        selected[start + take] = True
        remaining -= len(take)
    if remaining:
        raise RuntimeError("not enough tied rows at the exact Top-N cutoff")
    selected.flush()
    return selected, cutoff, n_cells


def _b_target_summary(
    robust_path: Path,
    selected: np.ndarray,
    shape: tuple[int, int, int],
    temp_dir: Path,
) -> dict[str, float | int]:
    binary = np.memmap(temp_dir / "b_binary.dat", dtype=np.bool_, mode="w+", shape=shape)
    binary[:] = False
    offset = 0
    for batch in pq.ParquetFile(robust_path).iter_batches(batch_size=500_000, columns=["iz", "iy", "ix"]):
        frame = batch.to_pandas()
        stop = offset + len(frame)
        keep = np.flatnonzero(selected[offset:stop])
        if len(keep):
            binary[
                frame["iz"].to_numpy(np.int32)[keep],
                frame["iy"].to_numpy(np.int32)[keep],
                frame["ix"].to_numpy(np.int32)[keep],
            ] = True
        offset = stop
    binary.flush()
    structure = ndimage.generate_binary_structure(3, 3)
    labels = np.memmap(temp_dir / "b_labels.dat", dtype=np.int32, mode="w+", shape=shape)
    n_raw = int(ndimage.label(binary, structure=structure, output=labels))
    labels.flush()
    counts = np.bincount(labels.reshape(-1), minlength=n_raw + 1)
    sizes = counts[1:]
    kept = sizes[sizes >= 30]
    discarded = int(sizes[sizes < 30].sum())
    selected_cells = int(sizes.sum())
    return {
        "connectivity": 26,
        "min_cells": 30,
        "n_raw_targets": n_raw,
        "n_targets": int(len(kept)),
        "selected_cells": selected_cells,
        "kept_cells": int(kept.sum()),
        "fragment_ratio": discarded / selected_cells if selected_cells else float("nan"),
        "mean_target_cells": float(kept.mean()) if len(kept) else float("nan"),
        "median_target_cells": float(np.median(kept)) if len(kept) else float("nan"),
    }


def _oof_validation_score(root: Path, positives: pd.DataFrame, audit: dict[str, object]) -> np.ndarray:
    feature_path = root / "data" / "feature_stack_3D_event_xylabel.parquet"
    feature_cols = list(audit["feature_columns"])
    output = np.full(len(positives), np.nan, dtype=np.float32)
    work = positives.reset_index(drop=True).copy()
    models: dict[int, list[xgb.XGBClassifier]] = {}
    for fold in sorted(work["fold"].unique()):
        models[int(fold)] = []
        for seed in (42, 52, 62):
            model = xgb.XGBClassifier()
            model.load_model(root / "outputs" / "01_modeling" / f"fold_{fold}" / f"model_seed_{seed}.json")
            models[int(fold)].append(model)
    offset = 0
    for batch in pq.ParquetFile(feature_path).iter_batches(batch_size=500_000, columns=feature_cols):
        stop = offset + batch.num_rows
        subset = work[(work["row_id"] >= offset) & (work["row_id"] < stop)]
        if len(subset):
            frame = batch.to_pandas()
            for fold, group in subset.groupby("fold", sort=False):
                local = group["row_id"].to_numpy(np.int64) - offset
                features = frame.iloc[local][feature_cols].to_numpy(np.float32, copy=False)
                predictions = np.column_stack([model.predict_proba(features)[:, 1] for model in models[int(fold)]])
                mean = predictions.mean(axis=1)
                uncertainty = predictions.std(axis=1)
                output[group.index.to_numpy(np.int64)] = mean - 0.5 * uncertainty
        offset = stop
    if np.isnan(output).any():
        raise RuntimeError("some validation rows were not found in the feature parquet")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    robust_path = root / "outputs" / "01_modeling" / "robust_scores.parquet"
    positives_path = root / "outputs" / "00_audit" / "validation_positives.csv"
    audit_path = root / "outputs" / "00_audit" / "audit_report.json"
    out = root / "paper_controls"
    temp = out / "_tmp"
    out.mkdir(exist_ok=True)
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    start = time.perf_counter()
    monitor_stop = threading.Event()
    memory = {"peak_rss_bytes": 0}
    monitor = threading.Thread(target=_peak_memory, args=(monitor_stop, memory), daemon=True)
    monitor.start()
    try:
        n_rows = pq.ParquetFile(robust_path).metadata.num_rows
        scores = np.lib.format.open_memmap(temp / "score_k05.npy", mode="w+", dtype=np.float32, shape=(n_rows,))
        offset = 0
        for batch in pq.ParquetFile(robust_path).iter_batches(batch_size=500_000, columns=["score_k05"]):
            values = batch.column(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            scores[offset : offset + len(values)] = values
            offset += len(values)
        scores.flush()
        if offset != n_rows:
            raise RuntimeError(f"row count mismatch: {offset} != {n_rows}")
        positives = pd.read_csv(positives_path)
        a_mask = np.asarray(scores >= 0.7, dtype=bool)
        a_rate = float(a_mask.mean())
        oof_scores = _oof_validation_score(root, positives, json.loads(audit_path.read_text(encoding="utf-8")))
        oof_hit = oof_scores >= 0.7
        strong = positives["is_strong"].to_numpy(bool)
        a_row: dict[str, float | int | str] = {
            "method": "A_fixed_score_ge_0.7",
            "score_cutoff": 0.7,
            "volume_rate": a_rate,
            "selected_cells": int(a_mask.sum()),
            "n_positive": len(positives),
            "n_strong": int(strong.sum()),
            "hits_all": int(oof_hit.sum()),
            "hits_strong": int((oof_hit & strong).sum()),
            "H_all": float(oof_hit.mean()),
            "H_strong": float((oof_hit & strong).sum() / strong.sum()),
            "Lift_all": float("nan"),
            "Lift_strong": float("nan"),
        }
        selected, cutoff, n_cells = _exact_top_n(scores, 0.10, temp / "b_selected.npy")
        b_row: dict[str, float | int | str] = {
            "method": "B_exact_top_10_percent",
            "score_cutoff": cutoff,
            "volume_rate": 0.10,
            "selected_cells": n_cells,
        }
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        pv = pd.read_csv(root / "outputs" / "01_modeling" / "pv_aggregate.csv")
        b_pv = pv[np.isclose(pv["kappa"], 0.5) & np.isclose(pv["volume_rate"], 0.10)]
        if len(b_pv) != 1:
            raise RuntimeError("missing exact 10% OOF P-V result")
        b_row.update(b_pv.iloc[0][["n_positive", "n_strong", "hits_all", "hits_strong", "H_all", "H_strong", "Lift_all", "Lift_strong"]].to_dict())
        b_row.update(_b_target_summary(robust_path, selected, tuple(audit["grid_shape_zyx"]), temp))
        pd.DataFrame([a_row, b_row]).to_csv(out / "control_experiments.csv", index=False, encoding="utf-8-sig")
    finally:
        monitor_stop.set()
        monitor.join(timeout=1)
        elapsed = time.perf_counter() - start
        temp_size = sum(path.stat().st_size for path in temp.rglob("*") if path.is_file())
        runtime = {
            "scope": "A/B threshold controls and B 26-neighbour, min-30 component filtering; no model training",
            "wall_time_seconds": elapsed,
            "peak_rss_gb": memory["peak_rss_bytes"] / 1024**3 if memory["peak_rss_bytes"] else None,
            "temporary_disk_gb": temp_size / 1024**3,
        }
        (out / "runtime_memory.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.keep_temp:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
