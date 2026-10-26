from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _exact_top_n_hits(
    scores: np.ndarray,
    row_ids: np.ndarray,
    volume_rate: float,
) -> tuple[np.ndarray, float, int]:
    """Select exactly ceil(N*v) rows, breaking equal-score ties by row ID."""
    n_scores = len(scores)
    n_cells = int(np.ceil(n_scores * volume_rate))
    n_cells = min(max(n_cells, 1), n_scores)
    threshold = float(np.quantile(scores, 1.0 - volume_rate, method="higher"))

    chunk = 5_000_000
    n_greater = 0
    for start in range(0, n_scores, chunk):
        stop = min(start + chunk, n_scores)
        n_greater += int(np.count_nonzero(scores[start:stop] > threshold))
    n_ties_needed = n_cells - n_greater
    if n_ties_needed < 0:
        raise RuntimeError("Top-N cutoff produced more rows above the cutoff than requested")

    positive_scores = scores[row_ids]
    hit_mask = positive_scores > threshold
    equal_positions = np.flatnonzero(positive_scores == threshold)
    if len(equal_positions) and n_ties_needed:
        # Rank positive rows within the globally ordered set of equal-score rows.
        order = np.argsort(row_ids[equal_positions])
        sorted_positions = equal_positions[order]
        sorted_ids = row_ids[sorted_positions]
        tie_ranks = np.zeros(len(sorted_ids), dtype=np.int64)
        pointer = 0
        seen = 0
        for start in range(0, n_scores, chunk):
            stop = min(start + chunk, n_scores)
            values = scores[start:stop]
            local = np.flatnonzero(values == threshold)
            while pointer < len(sorted_ids) and sorted_ids[pointer] < stop:
                local_id = int(sorted_ids[pointer] - start)
                tie_ranks[pointer] = seen + int(np.searchsorted(local, local_id, side="right"))
                pointer += 1
            seen += len(local)
        hit_mask[sorted_positions] = tie_ranks <= n_ties_needed
    return hit_mask, threshold, n_cells


def compute_pv_table(
    scores: np.ndarray,
    positives: pd.DataFrame,
    volume_rates: Iterable[float],
    kappa: float,
    fold: int,
) -> pd.DataFrame:
    rates = np.asarray(list(volume_rates), dtype=np.float64)
    row_ids = positives["row_id"].to_numpy(np.int64)
    positive_scores = scores[row_ids]
    strong = positives["is_strong"].to_numpy(bool)
    n_all = len(positive_scores)
    n_strong = int(strong.sum())
    rows: list[dict[str, float | int]] = []
    for rate in rates:
        hit_mask, threshold, n_cells = _exact_top_n_hits(scores, row_ids, float(rate))
        hits_all = int(hit_mask.sum())
        hits_strong = int((hit_mask & strong).sum())
        h_all = hits_all / n_all if n_all else np.nan
        h_strong = hits_strong / n_strong if n_strong else np.nan
        rows.append(
            {
                "fold": fold,
                "kappa": float(kappa),
                "volume_rate": float(rate),
                "score_threshold": float(threshold),
                "n_positive": n_all,
                "n_strong": n_strong,
                "hits_all": hits_all,
                "hits_strong": hits_strong,
                "H_all": h_all,
                "H_strong": h_strong,
                "Lift_all": h_all / rate if n_all else np.nan,
                "Lift_strong": h_strong / rate if n_strong else np.nan,
                "n_cells": n_cells,
                "selection_method": "exact_top_n_row_id_tiebreak",
            }
        )
    return pd.DataFrame(rows)


def aggregate_pv_tables(fold_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for (kappa, rate), group in fold_table.groupby(["kappa", "volume_rate"]):
        n_all = int(group["n_positive"].sum())
        n_strong = int(group["n_strong"].sum())
        hits_all = int(group["hits_all"].sum())
        hits_strong = int(group["hits_strong"].sum())
        h_all = hits_all / n_all if n_all else np.nan
        h_strong = hits_strong / n_strong if n_strong else np.nan
        rows.append(
            {
                "kappa": float(kappa),
                "volume_rate": float(rate),
                "mean_fold_threshold": float(group["score_threshold"].mean()),
                "n_positive": n_all,
                "n_strong": n_strong,
                "hits_all": hits_all,
                "hits_strong": hits_strong,
                "H_all": h_all,
                "H_strong": h_strong,
                "Lift_all": h_all / rate if n_all else np.nan,
                "Lift_strong": h_strong / rate if n_strong else np.nan,
                "n_cells": int(group["n_cells"].iloc[0]),
                "selection_method": "exact_top_n_row_id_tiebreak",
            }
        )
    return pd.DataFrame(rows).sort_values(["kappa", "volume_rate"])


def select_thresholds(pv_table: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, float | str]] = []
    for kappa, group in pv_table.groupby("kappa"):
        group = group.sort_values("volume_rate")
        h_max = float(group["H_all"].max())
        eligible = group[group["H_all"] >= 0.8 * h_max]
        primary = eligible.iloc[0]
        backup = group.loc[(group["H_all"] - group["volume_rate"]).idxmax()]
        selected.extend(
            [
                {
                    "kappa": float(kappa),
                    "rule": "80_percent_of_hmax",
                    "volume_rate": float(primary["volume_rate"]),
                    "H_all": float(primary["H_all"]),
                    "H_strong": float(primary["H_strong"]),
                    "Lift_all": float(primary["Lift_all"]),
                },
                {
                    "kappa": float(kappa),
                    "rule": "max_h_minus_v",
                    "volume_rate": float(backup["volume_rate"]),
                    "H_all": float(backup["H_all"]),
                    "H_strong": float(backup["H_strong"]),
                    "Lift_all": float(backup["Lift_all"]),
                },
            ]
        )
    return pd.DataFrame(selected)
