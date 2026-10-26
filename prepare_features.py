from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import uniform_filter


COL_X, COL_Y, COL_Z = "x", "y", "z"
LABEL_COL = "label"
STRATUM_COL = "地层"
EVENT_FEATURES = ["Ag_o", "Cu_o", "Mn_o"]

# Keep the native 1 m z grid. The kz=5 window carries the requested
# five-layer context without merging five physical layers into one cell.
WINDOWS = [
    (1, 3, 3),
    (3, 21, 21),
    (5, 21, 21),
    (7, 21, 21),
    (3, 21, 45),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1 m 3D convolution features")
    parser.add_argument("--input", default="data/属性_with_label.parquet")
    parser.add_argument("--output", default="data/feature_stack_3D_event.parquet")
    parser.add_argument("--metadata", default="data/feature_stack_3D_event.metadata.json")
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dy", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=1.0)
    parser.add_argument("--block-z", type=int, default=96)
    parser.add_argument("--batch-rows", type=int, default=2_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_columns(parquet: pq.ParquetFile) -> None:
    required = {COL_X, COL_Y, COL_Z, LABEL_COL, STRATUM_COL, *EVENT_FEATURES}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Input Parquet is missing columns: {sorted(missing)}")


def layer_ranges(nz: int, block_size: int, overlap: int):
    start = 0
    while start < nz:
        yield start, max(0, start - overlap), min(nz, start + block_size + overlap)
        start += block_size


def collect_bounds_and_main_stratum(
    source: Path, batch_rows: int
) -> tuple[float, float, float, float, float, float, int, int]:
    parquet = pq.ParquetFile(source)
    ensure_columns(parquet)
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = -np.inf, -np.inf, -np.inf
    counts: dict[int, int] = {}
    sampled = 0
    sample_limit = 2_000_000
    rng = np.random.default_rng(42)

    for batch in parquet.iter_batches(
        batch_size=batch_rows, columns=[COL_X, COL_Y, COL_Z, STRATUM_COL]
    ):
        frame = batch.to_pandas()
        xmin = min(xmin, float(frame[COL_X].min()))
        xmax = max(xmax, float(frame[COL_X].max()))
        ymin = min(ymin, float(frame[COL_Y].min()))
        ymax = max(ymax, float(frame[COL_Y].max()))
        zmin = min(zmin, float(frame[COL_Z].min()))
        zmax = max(zmax, float(frame[COL_Z].max()))
        if sampled >= sample_limit:
            continue
        values = pd.to_numeric(frame[STRATUM_COL], errors="coerce").fillna(0)
        values = values.to_numpy(np.int32)
        values = values[values != 0]
        take = min(len(values), sample_limit - sampled)
        if take < len(values):
            values = values[rng.choice(len(values), size=take, replace=False)]
        sampled += len(values)
        if len(values):
            unique, value_counts = np.unique(values, return_counts=True)
            for value, count in zip(unique.tolist(), value_counts.tolist()):
                counts[value] = counts.get(value, 0) + int(count)

    if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
        raise ValueError("Input Parquet contains no usable coordinate values")
    main_stratum = max(counts.items(), key=lambda item: item[1])[0] if counts else 0
    return xmin, xmax, ymin, ymax, zmin, zmax, main_stratum, sampled


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and metadata_path.exists() and not args.force:
        print(f"Feature file already exists, keeping it: {destination}")
        print("Use --force only after deleting obsolete output files.")
        return 0
    if destination.exists() and not args.force:
        raise FileExistsError(f"Existing output without metadata: {destination}")

    print("Scanning coordinate bounds and main stratum...")
    xmin, xmax, ymin, ymax, zmin, zmax, main_stratum, sampled = (
        collect_bounds_and_main_stratum(source, args.batch_rows)
    )
    nx = int(np.rint((xmax - xmin) / args.dx)) + 1
    ny = int(np.rint((ymax - ymin) / args.dy)) + 1
    nz = int(np.rint((zmax - zmin) / args.dz)) + 1
    print(f"Grid (z, y, x): ({nz}, {ny}, {nx}); z spacing={args.dz:g} m")
    print(f"Main stratum={main_stratum}; sample size={sampled:,}")

    max_kz = max(window[0] for window in WINDOWS)
    overlap = max_kz // 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        destination.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for block_number, (core_start, cache_start, cache_stop) in enumerate(
            layer_ranges(nz, args.block_z, overlap), start=1
        ):
            print(
                f"Z block {block_number}: core [{core_start}, "
                f"{min(core_start + args.block_z, nz)})"
            )
            cached: list[pd.DataFrame] = []
            parquet = pq.ParquetFile(source)
            columns = [COL_X, COL_Y, COL_Z, LABEL_COL, STRATUM_COL, *EVENT_FEATURES]
            for batch in parquet.iter_batches(batch_size=args.batch_rows, columns=columns):
                frame = batch.to_pandas()
                iz = np.rint(
                    (zmax - frame[COL_Z].to_numpy(np.float64)) / args.dz
                ).astype(np.int32)
                mask = (iz >= cache_start) & (iz < cache_stop)
                if mask.any():
                    selected = frame.loc[mask].copy()
                    selected["iz"] = iz[mask]
                    cached.append(selected)
            if not cached:
                continue

            block = pd.concat(cached, ignore_index=True)
            del cached
            ix = np.rint((block[COL_X].to_numpy(np.float64) - xmin) / args.dx).astype(
                np.int32
            )
            iy = np.rint((block[COL_Y].to_numpy(np.float64) - ymin) / args.dy).astype(
                np.int32
            )
            iz = block["iz"].to_numpy(np.int32)
            block["ix"] = ix
            block["iy"] = iy
            core = (iz >= core_start) & (iz < min(core_start + args.block_z, nz))
            core_ix, core_iy, core_iz = ix[core], iy[core], iz[core]
            output: dict[str, np.ndarray] = {
                "ix": core_ix,
                "iy": core_iy,
                "iz": core_iz,
                LABEL_COL: block.loc[core, LABEL_COL].to_numpy(np.int8),
            }
            block_depth = cache_stop - cache_start
            local_z = (iz - cache_start).astype(np.int32)
            valid = (
                (local_z >= 0)
                & (local_z < block_depth)
                & (iy >= 0)
                & (iy < ny)
                & (ix >= 0)
                & (ix < nx)
            )
            z_valid, y_valid, x_valid = local_z[valid], iy[valid], ix[valid]
            core_local_z = (core_iz - cache_start).astype(np.int32)
            stratum = pd.to_numeric(block[STRATUM_COL], errors="coerce").fillna(0)
            block["stratum_flag"] = (
                (stratum.to_numpy(np.int32) != 0)
                & (stratum.to_numpy(np.int32) != main_stratum)
            ).astype(np.float32)

            for feature in [*EVENT_FEATURES, "stratum_flag"]:
                print(f"  {feature}")
                values = pd.to_numeric(block[feature], errors="coerce").fillna(0.0)
                values = values.to_numpy(np.float32)[valid]
                nonzero = values != 0.0
                if not nonzero.any():
                    for window in WINDOWS:
                        suffix = f"k{window}"
                        output[f"{feature}_cnt_{suffix}"] = np.zeros(
                            len(core_ix), dtype=np.float32
                        )
                        output[f"{feature}_mean_{suffix}"] = np.zeros(
                            len(core_ix), dtype=np.float32
                        )
                    continue
                sum_grid = np.zeros((block_depth, ny, nx), dtype=np.float32)
                count_grid = np.zeros((block_depth, ny, nx), dtype=np.float32)
                np.add.at(
                    sum_grid,
                    (z_valid[nonzero], y_valid[nonzero], x_valid[nonzero]),
                    values[nonzero],
                )
                np.add.at(
                    count_grid,
                    (z_valid[nonzero], y_valid[nonzero], x_valid[nonzero]),
                    1.0,
                )
                for window in WINDOWS:
                    volume = float(np.prod(window))
                    count_window = uniform_filter(
                        count_grid, size=window, mode="constant", cval=0.0
                    ) * volume
                    sum_window = uniform_filter(
                        sum_grid, size=window, mode="constant", cval=0.0
                    ) * volume
                    mean_window = sum_window / np.maximum(count_window, 1.0)
                    suffix = f"k{window}"
                    output[f"{feature}_cnt_{suffix}"] = count_window[
                        core_local_z, core_iy, core_ix
                    ].astype(np.float32)
                    output[f"{feature}_mean_{suffix}"] = mean_window[
                        core_local_z, core_iy, core_ix
                    ].astype(np.float32)
                del sum_grid, count_grid, count_window, sum_window, mean_window

            table = pa.Table.from_pandas(pd.DataFrame(output), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += table.num_rows
            del block, output
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError("No feature rows were written")
    metadata = {
        "source": str(source),
        "output": str(destination),
        "rows": total_rows,
        "grid": {"dx": args.dx, "dy": args.dy, "dz": args.dz, "rebin_z": 1},
        "bounds": {"x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax, "z_min": zmin, "z_max": zmax},
        "grid_shape_zyx": [nz, ny, nx],
        "windows": WINDOWS,
        "main_stratum": main_stratum,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete: {destination} ({total_rows:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
