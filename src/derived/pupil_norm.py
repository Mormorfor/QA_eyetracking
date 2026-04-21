# src/derived/pupil_norm.py

import os
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

def zscore_pupil_by_participant(
    df: pd.DataFrame,
    pupil_col: str,
    participant_col: str,
    stats_csv_path: Path = None,
    out_col: str = None,
) -> pd.DataFrame:
    """
    Z-score pupil values using participant-level mean/std.

    pupil_z = (pupil - participant_mean) / participant_std
    """

    if stats_csv_path == None:
        stats_csv_path = PROJECT_ROOT / "data" / "participant_pupils.csv"


    if out_col is None:
        out_col = f"{pupil_col}_z"

    stats = pd.read_csv(stats_csv_path)
    stats = stats[[participant_col, "pupil_mean", "pupil_sd"]]

    df = df.merge(stats, on=participant_col, how="left")

    df[out_col] = (df[pupil_col] - df["pupil_mean"]) / df["pupil_sd"]
    df[out_col] = df[out_col].replace([float("inf"), float("-inf")], pd.NA)

    return df.drop(columns=["pupil_mean", "pupil_sd"])
