"""Pupil size: convert to millimetres, and z-score against each person's own
typical size.

The tracker reports pupil size as an area in its own units. This converts it to
a diameter in mm, then expresses it as a z-score: how many standard deviations
above or below that participant's own average it is. Z-scoring keeps people
with naturally larger pupils from dominating.

Each person's average and SD (the "baseline") come from all of their fixations
on one screen of one dataset. The answer screen and the paragraph screen each
get their own baseline, because they differ in brightness and in task.
"""

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import columns as C
from src.features import area_metrics as am

# This module deliberately knows no dataset paths. It computes a baseline from
# whatever fixations it is given, so the caller must say which ones -- it can
# never quietly fall back to another dataset's fixations.


def scale_pupil_area_to_mm(
    pupil_area: pd.Series,
    artificial_pupil_width_mm: float = 3.5,
    avg_pupil_area: float = 1804.0,
) -> pd.Series:
    """Convert the tracker's pupil area (its own units) to a diameter in mm.

        diameter_mm = artificial_pupil_width_mm * sqrt(area / avg_pupil_area)

    That is: an artificial pupil 3.5 mm wide reads as an area of 1804, and
    diameter grows with the square root of area. "." (no reading) becomes NaN.
    """
    pupil_area = pupil_area.replace(".", np.nan).astype(float)
    scaling_factor = artificial_pupil_width_mm / np.sqrt(avg_pupil_area)
    return scaling_factor * np.sqrt(pupil_area)


def compute_participant_pupil_stats(
    df: pd.DataFrame,
    group_col: str = C.PARTICIPANT_ID,
) -> pd.DataFrame:
    """Each person's average and SD of pupil size (in mm), over their fixations.

    `df` is a fixation report (one row per fixation). `group_col` decides what
    "this person's typical pupil size" means; the project uses the person
    (`participant_id`) for every dataset, so in Study 2 a person's sessions are
    pooled.

    Returns columns [group_col, pupil_mean, pupil_sd].
    """
    df_local = df.copy()

    df_local["pupil_mm"] = scale_pupil_area_to_mm(df_local[C.CURRENT_FIX_PUPIL_SIZE])

    return (
        df_local.groupby(group_col)["pupil_mm"]
        .agg(pupil_mean="mean", pupil_sd="std")
        .reset_index()
    )


def get_participant_pupil_stats(
    stats=None,
    fixations_path: Path = None,
    fixations=None,
    group_col: str = C.PARTICIPANT_ID,
    verbose: bool = True,
) -> pd.DataFrame:
    """Each person's pupil baseline (average and SD), from the source you name.

    In order of preference:
    1. `stats`, if it is already a baseline table -- used as is;
    2. `fixations`, an already-loaded fixation report;
    3. the fixation report at `fixations_path`.

    There is no default: if none is given, this stops with an error, so a
    baseline can never quietly come from the wrong dataset or screen.

    The pipeline always computes the baseline from fixations. The saved
    `pupil_baseline__*.csv` files are a record of what a run used, not an input:
    a baseline read back from a file cannot say which fixations it came from.

    Returns columns [group_col, pupil_mean, pupil_sd].
    """
    if isinstance(stats, pd.DataFrame):
        return stats

    if isinstance(fixations, pd.DataFrame):
        if verbose:
            print("Computing participant pupil stats from preloaded fixations")
        return compute_participant_pupil_stats(fixations, group_col=group_col)

    if fixations_path is None:
        raise ValueError(
            "No pupil baseline source given. Pass one of: `stats` (a resolved "
            "stats frame), `fixations` (a loaded fixation report), or "
            "`fixations_path` (a fixation report to compute from). There is no "
            "default, so that a baseline can never come from the wrong dataset."
        )
    if verbose:
        print(f"Computing participant pupil stats from: {fixations_path}")
    return compute_participant_pupil_stats(
        pd.read_csv(fixations_path), group_col=group_col
    )


def zscore_pupil_by_participant(
    df: pd.DataFrame,
    pupil_col: str,
    participant_col: str,
    stats,
    out_col: str = None,
) -> pd.DataFrame:
    """Z-score one pupil column against each person's baseline.

        pupil_z = (pupil - person's average) / person's SD

    `stats` is the baseline table (columns participant_col, pupil_mean,
    pupil_sd), or the path to one. It is required.

    Stops with an error if anyone in `df` has no baseline, or a baseline SD
    that is missing or not positive -- otherwise their z-scores would quietly
    come out empty.
    """
    if out_col is None:
        out_col = f"{pupil_col}_z"

    if isinstance(stats, pd.DataFrame):
        stats_df = stats
    else:
        stats_df = pd.read_csv(stats)

    stats_df = stats_df[[participant_col, "pupil_mean", "pupil_sd"]]

    # Everyone needs a usable baseline. Without these checks, a missing one would
    # silently give that person empty z-scores. The worst case is a baseline
    # table from another dataset: no ids would match, and every z-score would be
    # empty without any error.
    missing = sorted(
        set(df[participant_col].dropna().unique())
        - set(stats_df[participant_col].dropna().unique())
    )
    if missing:
        raise ValueError(
            f"No pupil baseline for {len(missing)} of "
            f"{df[participant_col].nunique()} {participant_col} value(s): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
            "Either the stats table is from a different dataset, or these people "
            "have no fixations in the baseline report. Do not z-score without a "
            "baseline."
        )

    unusable = stats_df[
        stats_df[participant_col].isin(df[participant_col].dropna().unique())
        & (stats_df["pupil_sd"].isna() | (stats_df["pupil_sd"] <= 0))
    ][participant_col].tolist()
    if unusable:
        raise ValueError(
            f"Pupil baseline SD is missing or non-positive for {participant_col} "
            f"{unusable[:10]}{' ...' if len(unusable) > 10 else ''}. A z-score is "
            "undefined here; too few baseline fixations is the usual cause."
        )

    df = df.merge(stats_df, on=participant_col, how="left")

    df[out_col] = (df[pupil_col] - df["pupil_mean"]) / df["pupil_sd"]
    df[out_col] = df[out_col].replace([float("inf"), float("-inf")], pd.NA)

    return df.drop(columns=["pupil_mean", "pupil_sd"])

# ---------------------------------------------------------------------------
# Preparing a screen's pupil columns, and the registry entries
# ---------------------------------------------------------------------------


def prepare_screen_pupil(
    df: pd.DataFrame,
    pupil_stats: pd.DataFrame,
    pupil_cols=None,
) -> pd.DataFrame:
    """Turn a screen's word columns into numbers, then convert the pupil
    columns to mm and z-score them.

    `pupil_stats` must be this screen's own baseline, from this screen's
    fixations. Used for the paragraph screen.
    """
    from src.config.screens import PUPIL_RAW_COLS
    from src.features import area_metrics as _am

    cols = PUPIL_RAW_COLS if pupil_cols is None else pupil_cols
    out = _am.coerce_ia_columns(df, pupil=True)
    for col in cols:
        out[col] = scale_pupil_area_to_mm(out[col])
        out = zscore_pupil_by_participant(
            df=out,
            pupil_col=col,
            participant_col=C.PARTICIPANT_ID,
            stats=pupil_stats,
            out_col=f"{col}_z",
        )
    return out

def add_zscored_pupil_columns(
    df: pd.DataFrame,
    pupil_stats=None,
) -> pd.DataFrame:
    """Answer screen: convert the per-word pupil columns to mm and add `_z`
    versions.

    The mm values replace the raw ones in the same columns. `pupil_stats` is
    required and must be this dataset's own answer-screen baseline;
    `ingest/build.py::main` works it out and passes it in.
    """
    if pupil_stats is None:
        raise ValueError(
            "add_zscored_pupil_columns needs `pupil_stats` for THIS dataset. "
            "main() injects it; a direct caller must resolve it explicitly via "
            "get_participant_pupil_stats(fixations_path=<this dataset's fixations>)."
        )
    pupil_stats = get_participant_pupil_stats(stats=pupil_stats)

    out = df.copy()
    out = out.reset_index()

    pupil_cols = [
        C.IA_MAX_FIX_PUPIL_SIZE,
        C.IA_MIN_FIX_PUPIL_SIZE,
        C.IA_AVERAGE_FIX_PUPIL_SIZE,
    ]

    for col in pupil_cols:
        out[col] = scale_pupil_area_to_mm(out[col])

        out = zscore_pupil_by_participant(
            df=out,
            pupil_col=col,
            participant_col=C.PARTICIPANT_ID,
            stats=pupil_stats,
            out_col=f"{col}_z",
        )

    return out


def create_mean_pupil_size_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Average pupil size per answer-screen area, in mm and z-scored.

    The pupil columns are turned into numbers on a copy, so the caller's frame
    is not changed. See `area_metrics.mean_pupil_size`.
    """
    df_local = df.copy()

    mm_cols = [
        C.IA_MAX_FIX_PUPIL_SIZE,
        C.IA_MIN_FIX_PUPIL_SIZE,
        C.IA_AVERAGE_FIX_PUPIL_SIZE,
    ]
    for col in mm_cols + [f"{c}_z" for c in mm_cols]:
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

    return am.mean_pupil_size(
        df_local, C.AREA_LABEL_COLUMN, include_raw=True, include_z=True
    )


def create_first_encounter_pupil_size(df: pd.DataFrame) -> pd.DataFrame:
    """Pupil size on each answer-screen area's first word, by position, that
    was looked at -- in mm and z-scored.

    Not necessarily the first word looked at in time; see
    `area_metrics.first_encounter_pupil_size`.
    """
    df_local = df.copy()

    mm_col = C.IA_AVERAGE_FIX_PUPIL_SIZE
    z_col = f"{mm_col}_z"
    df_local[mm_col] = pd.to_numeric(df_local[mm_col], errors="coerce")
    df_local[z_col] = pd.to_numeric(df_local[z_col], errors="coerce")

    return am.first_encounter_pupil_size(
        df_local, C.AREA_LABEL_COLUMN, include_raw=True, include_z=True
    )
