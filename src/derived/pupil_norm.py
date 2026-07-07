# src/derived/pupil_norm.py

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import constants as C
from src.data_paths import FIX_ANSWERS_PATH, PARTICIPANT_PUPILS_PATH


def scale_pupil_area_to_mm(
    pupil_area: pd.Series,
    artificial_pupil_width_mm: float = 3.5,
    avg_pupil_area: float = 1804.0,
) -> pd.Series:
    """
    Convert pupil area (arbitrary units) to pupil diameter in mm.

    Scaling:  diameter_mm = scaling_factor * sqrt(area)
    where:    scaling_factor = artificial_pupil_width_mm / sqrt(avg_pupil_area)
    """
    pupil_area = pupil_area.replace(".", np.nan).astype(float)
    scaling_factor = artificial_pupil_width_mm / np.sqrt(avg_pupil_area)
    return scaling_factor * np.sqrt(pupil_area)


def compute_participant_pupil_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-participant mean and SD of pupil size (in mm) from fixation-level
    data.

    Steps:
    1. Scale the raw fixation pupil size to mm.
    2. Compute mean and SD per participant.

    Returns a DataFrame with columns [participant_id, pupil_mean, pupil_sd].
    """
    df_local = df.copy()

    df_local["pupil_mm"] = scale_pupil_area_to_mm(df_local[C.CURRENT_FIX_PUPIL_SIZE])

    return (
        df_local.groupby(C.PARTICIPANT_ID)["pupil_mm"]
        .agg(pupil_mean="mean", pupil_sd="std")
        .reset_index()
    )


def get_participant_pupil_stats(
    stats=None,
    stats_csv_path: Path = PARTICIPANT_PUPILS_PATH,
    fixations_path: Path = FIX_ANSWERS_PATH,
    compute: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Resolve participant-level pupil statistics from one of three sources.

    Priority:
    1. If `stats` is already a DataFrame, it is used as-is.
    2. Else if `compute` is True, statistics are computed on the fly from the
       fixation-level CSV at `fixations_path` (no precomputed file required).
    3. Else the precomputed statistics CSV at `stats_csv_path` is loaded.

    Returns a DataFrame with columns [participant_id, pupil_mean, pupil_sd].
    """
    if isinstance(stats, pd.DataFrame):
        return stats

    if compute:
        if verbose:
            print(f"Computing participant pupil stats from: {fixations_path}")
        fixations = pd.read_csv(fixations_path)
        return compute_participant_pupil_stats(fixations)

    if verbose:
        print(f"Loading participant pupil stats from: {stats_csv_path}")
    return pd.read_csv(stats_csv_path)


def zscore_pupil_by_participant(
    df: pd.DataFrame,
    pupil_col: str,
    participant_col: str,
    stats=PARTICIPANT_PUPILS_PATH,
    out_col: str = None,
) -> pd.DataFrame:
    """
    Z-score pupil values using participant-level mean/std.

    pupil_z = (pupil - participant_mean) / participant_std

    `stats` may be either a resolved statistics DataFrame (with columns
    participant_col, "pupil_mean", "pupil_sd") or a path to such a CSV.
    """
    if out_col is None:
        out_col = f"{pupil_col}_z"

    if isinstance(stats, pd.DataFrame):
        stats_df = stats
    else:
        stats_df = pd.read_csv(stats)

    stats_df = stats_df[[participant_col, "pupil_mean", "pupil_sd"]]

    df = df.merge(stats_df, on=participant_col, how="left")

    df[out_col] = (df[pupil_col] - df["pupil_mean"]) / df["pupil_sd"]
    df[out_col] = df[out_col].replace([float("inf"), float("-inf")], pd.NA)

    return df.drop(columns=["pupil_mean", "pupil_sd"])
