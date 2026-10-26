from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import yaml


LOG = logging.getLogger("target3d")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_project_dir"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg["_project_dir"]) / path
    return path.resolve()


def output_dir(cfg: dict[str, Any]) -> Path:
    path = resolve_path(cfg, cfg["paths"]["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_dir(cfg: dict[str, Any], name: str) -> Path:
    path = output_dir(cfg) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed(stage_path: Path, cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("runtime", {}).get("resume", True) and stage_path.exists())


def mark_complete(stage_path: Path, payload: dict[str, Any] | None = None) -> None:
    write_json(stage_path, payload or {"complete": True})


def feature_columns(schema_names: list[str]) -> list[str]:
    return [name for name in schema_names if "_cnt_k" in name or "_mean_k" in name]


def parquet_num_rows(path: str | Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def metadata_min_max(path: str | Path, column: str) -> tuple[float, float]:
    parquet = pq.ParquetFile(path)
    index = parquet.schema_arrow.names.index(column)
    mins: list[float] = []
    maxs: list[float] = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(index).statistics
        if stats is None or not stats.has_min_max:
            raise ValueError(f"Missing Parquet statistics for {column!r} in {path}")
        mins.append(float(stats.min))
        maxs.append(float(stats.max))
    return min(mins), max(maxs)


def iter_synced_batches(
    left_path: str | Path,
    right_path: str | Path,
    left_columns: list[str],
    right_columns: list[str],
    batch_size: int,
    max_rows: int | None = None,
) -> Iterator[tuple[Any, Any, int]]:
    left = pq.ParquetFile(left_path).iter_batches(
        batch_size=batch_size, columns=left_columns
    )
    right = pq.ParquetFile(right_path).iter_batches(
        batch_size=batch_size, columns=right_columns
    )
    offset = 0
    while True:
        try:
            left_batch = next(left)
        except StopIteration:
            try:
                next(right)
            except StopIteration:
                return
            raise ValueError("Right Parquet has more rows than left Parquet")
        try:
            right_batch = next(right)
        except StopIteration as exc:
            raise ValueError("Left Parquet has more rows than right Parquet") from exc
        if left_batch.num_rows != right_batch.num_rows:
            raise ValueError(
                f"Batch size mismatch at row {offset}: "
                f"{left_batch.num_rows} != {right_batch.num_rows}"
            )
        if max_rows is not None and offset >= max_rows:
            return
        if max_rows is not None and offset + left_batch.num_rows > max_rows:
            keep = max_rows - offset
            left_batch = left_batch.slice(0, keep)
            right_batch = right_batch.slice(0, keep)
        yield left_batch, right_batch, offset
        offset += left_batch.num_rows


def open_npy(path: str | Path, mode: str, shape: tuple[int, ...] | None = None):
    path = Path(path)
    if mode.startswith("w"):
        if shape is None:
            raise ValueError("shape is required when creating a memmap")
        return np.lib.format.open_memmap(path, mode=mode, dtype=np.float32, shape=shape)
    return np.load(path, mmap_mode=mode)


def available_device(requested: str) -> str:
    if requested != "auto":
        return requested
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible == "-1":
        return "cpu"
    try:
        import xgboost as xgb

        x = np.asarray([[0.0], [1.0]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int8)
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(x, y)
        return "cuda"
    except Exception:
        return "cpu"
