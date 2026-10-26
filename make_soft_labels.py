from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create global XY/Z soft labels")
    parser.add_argument("--input", default="data/feature_stack_3D_event.parquet")
    parser.add_argument("--output", default="data/feature_stack_3D_event_xylabel.parquet")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_positive_trees(source: Path, batch_rows: int) -> tuple[dict[int, cKDTree], dict[int, cKDTree], int]:
    positions: dict[int, list[np.ndarray]] = defaultdict(list)
    total = 0
    for batch in pq.ParquetFile(source).iter_batches(
        batch_size=batch_rows, columns=["ix", "iy", "iz", "label"]
    ):
        frame = batch.to_pandas()
        mask = frame["label"].to_numpy(np.int8) >= 0
        if not mask.any():
            continue
        positive = frame.loc[mask, ["ix", "iy", "iz"]]
        for layer, group in positive.groupby("iz", sort=False):
            positions[int(layer)].append(group[["ix", "iy"]].to_numpy(np.float32))
            total += len(group)
    same: dict[int, cKDTree] = {}
    for layer, groups in positions.items():
        same[layer] = cKDTree(np.vstack(groups))
    # A point at source layer z expands vertically onto layers z-1 and z+1.
    # Build those destination-layer trees explicitly: many destination layers
    # contain no known mineralization cells themselves.
    adjacent_positions: dict[int, list[np.ndarray]] = defaultdict(list)
    for source_layer, groups in positions.items():
        adjacent_positions[source_layer - 1].extend(groups)
        adjacent_positions[source_layer + 1].extend(groups)
    adjacent = {
        layer: cKDTree(np.vstack(groups))
        for layer, groups in adjacent_positions.items()
        if groups
    }
    return same, adjacent, total


def within_radius(tree: cKDTree | None, points: np.ndarray, radius: int) -> np.ndarray:
    if tree is None or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    distances, _ = tree.query(points.astype(np.float32), k=1, distance_upper_bound=radius)
    return np.isfinite(distances)


def main() -> int:
    args = parse_args()
    source, destination = Path(args.input).resolve(), Path(args.output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and not args.force:
        print(f"Soft-label file already exists, keeping it: {destination}")
        return 0
    if args.force:
        destination.unlink(missing_ok=True)
    print("Collecting known mineralization cells for global XY/Z expansion...")
    same_trees, adjacent_trees, positives = build_positive_trees(source, args.batch_rows)
    if positives == 0:
        raise RuntimeError("No label >= 0 cells found")
    print(f"Known mineralization cells: {positives}; occupied layers: {len(same_trees)}")

    counts: dict[int, int] = defaultdict(int)
    writer: pq.ParquetWriter | None = None
    try:
        for number, batch in enumerate(pq.ParquetFile(source).iter_batches(batch_size=args.batch_rows), start=1):
            frame = batch.to_pandas()
            labels = frame["label"].to_numpy(np.int8, copy=True)
            result = labels.copy()
            ix = frame["ix"].to_numpy(np.int32)
            iy = frame["iy"].to_numpy(np.int32)
            iz = frame["iz"].to_numpy(np.int32)
            for layer in np.unique(iz):
                rows = np.flatnonzero((iz == layer) & (result == -1))
                if not len(rows):
                    continue
                points = np.column_stack((ix[rows], iy[rows]))
                same_hits = within_radius(same_trees.get(int(layer)), points, args.radius)
                result[rows[same_hits]] = -2
                remaining = rows[~same_hits]
                if len(remaining):
                    neighbour_points = np.column_stack((ix[remaining], iy[remaining]))
                    z_hits = within_radius(
                        adjacent_trees.get(int(layer)), neighbour_points, args.radius
                    )
                    result[remaining[z_hits]] = -3
            frame["label_xy"] = result
            for value, count in zip(*np.unique(result, return_counts=True)):
                counts[int(value)] += int(count)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            if number % 25 == 0:
                print(f"Soft-label batches written: {number}")
    finally:
        if writer is not None:
            writer.close()
    destination.with_suffix(destination.suffix + ".labels.json").write_text(
        json.dumps({"radius_cells": args.radius, "counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Complete: {destination}")
    print("label_xy counts:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
