from __future__ import annotations

"""Recompute exact Top-N P-V validation from saved models without training."""

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from target3d.audit import run_audit
from target3d.common import load_config, open_npy, setup_logging, stage_dir
from target3d.modeling import _predict_model
from target3d.pv import aggregate_pv_tables, compute_pv_table, select_thresholds
from target3d.reporting import run_reporting
from target3d.targets import run_targets

LOWER_VOLUME_RATES = [0.0001, 0.0002, 0.0005, 0.00075]


def _require_models(model_dir: Path, n_splits: int, seeds: list[int]) -> None:
    missing = [
        str(model_dir / f"fold_{fold}" / f"model_seed_{seed}.json")
        for fold in range(n_splits)
        for seed in seeds
        if not (model_dir / f"fold_{fold}" / f"model_seed_{seed}.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Saved model files are incomplete. This script will not retrain. Missing:\n"
            + "\n".join(missing)
        )


def main() -> None:
    setup_logging()
    cfg = load_config("config.yaml")
    cfg["validation"]["volume_rates"] = sorted(
        set(LOWER_VOLUME_RATES + list(cfg["validation"]["volume_rates"]))
    )
    audit = run_audit(cfg)
    model_dir = stage_dir(cfg, "01_modeling")
    n_splits = int(cfg["model"]["n_splits"])
    seeds = [int(seed) for seed in cfg["model"]["seeds"]]
    _require_models(model_dir, n_splits, seeds)
    required = ("mu.npy", "u.npy", "robust_scores.parquet")
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing modeling artifacts: {missing}")

    positives = pd.read_csv(stage_dir(cfg, "00_audit") / "validation_positives.csv")
    feature_cols = audit["feature_columns"]
    n_rows = int(audit["processed_rows"])
    chunk = int(cfg["runtime"].get("array_chunk_rows", 5_000_000))
    temp_dir = model_dir / "pv_recompute_tmp"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    fold_tables: list[pd.DataFrame] = []
    try:
        for fold in range(n_splits):
            fold_dir = model_dir / f"fold_{fold}"
            total = open_npy(temp_dir / "sum.npy", "w+", shape=(n_rows,))
            sumsq = open_npy(temp_dir / "sumsq.npy", "w+", shape=(n_rows,))
            total[:] = 0.0
            sumsq[:] = 0.0
            prediction_path = temp_dir / "prediction.npy"
            for seed in seeds:
                model = xgb.XGBClassifier()
                model.load_model(fold_dir / f"model_seed_{seed}.json")
                predicted = _predict_model(cfg, model, feature_cols, prediction_path, n_rows)
                for start in range(0, n_rows, chunk):
                    stop = min(start + chunk, n_rows)
                    values = np.asarray(predicted[start:stop], dtype=np.float32)
                    total[start:stop] += values
                    sumsq[start:stop] += values * values
                total.flush()
                sumsq.flush()
                del predicted, model
                gc.collect()

            work = open_npy(temp_dir / "score.npy", "w+", shape=(n_rows,))
            validation = positives[positives["fold"] == fold].copy()
            parts: list[pd.DataFrame] = []
            for kappa in cfg["score"]["kappas"]:
                for start in range(0, n_rows, chunk):
                    stop = min(start + chunk, n_rows)
                    mean = total[start:stop] / len(seeds)
                    variance = np.maximum(sumsq[start:stop] / len(seeds) - mean * mean, 0.0)
                    work[start:stop] = mean - float(kappa) * np.sqrt(variance)
                work.flush()
                parts.append(compute_pv_table(work, validation, cfg["validation"]["volume_rates"], float(kappa), fold))
            table = pd.concat(parts, ignore_index=True)
            table.to_csv(fold_dir / "pv_table.csv", index=False)
            fold_tables.append(table)
            del work, total, sumsq
            gc.collect()

        fold_pv = pd.concat(fold_tables, ignore_index=True)
        fold_pv.to_csv(model_dir / "pv_folds.csv", index=False)
        aggregate = aggregate_pv_tables(fold_pv)
        aggregate.to_csv(model_dir / "pv_aggregate.csv", index=False)
        selected = select_thresholds(aggregate)
        selected.to_csv(model_dir / "threshold_summary.csv", index=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # v* may have changed, so rebuild downstream targets and the report.
    shutil.rmtree(stage_dir(cfg, "02_targets"), ignore_errors=True)
    shutil.rmtree(stage_dir(cfg, "03_report"), ignore_errors=True)
    modeling = {
        "mu": model_dir / "mu.npy",
        "u": model_dir / "u.npy",
        "robust_scores": model_dir / "robust_scores.parquet",
        "thresholds": model_dir / "threshold_summary.csv",
    }
    run_targets(cfg, audit, modeling)
    report = run_reporting(cfg)
    print(f"Complete: {report['analysis_bundle']}")


if __name__ == "__main__":
    main()
