# src/features/pupil.py

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import columns as C
from src.features import area_metrics as am

# NOTE: this module deliberately imports no dataset path. Its job is to compute a
# baseline from whatever fixations it is handed; knowing where L1 keeps its
# fixation report is what let the wrong baseline leak in (docs/todo.md T3.20).


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


def compute_participant_pupil_stats(
    df: pd.DataFrame,
    group_col: str = C.PARTICIPANT_ID,
) -> pd.DataFrame:
    """
    Compute mean and SD of pupil size (in mm) per `group_col`, from fixation-level
    data.

    Steps:
    1. Scale the raw fixation pupil size to mm.
    2. Compute mean and SD per group.

    `group_col` is the BASELINE UNIT and it is a scientific choice, not a detail:
    it decides what "this participant's typical pupil size" means. The project's
    unit is the person (`participant_id`) for every dataset -- including KnowQA,
    where a person sits for several recording sessions and their baseline pools
    across all of them (Diana, 2026-09-23; see docs/todo.md T3.20).

    Returns a DataFrame with columns [group_col, pupil_mean, pupil_sd].
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
    stats_csv_path: Path = None,
    fixations_path: Path = None,
    fixations=None,
    compute: bool = True,
    group_col: str = C.PARTICIPANT_ID,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Resolve participant-level pupil statistics from one of three sources.

    Priority:
    1. If `stats` is already a DataFrame, it is used as-is.
    2. Else if `compute` is True, statistics are computed on the fly from a
       fixation-level DataFrame: `fixations` when it is provided (to avoid
       re-reading an already-loaded report), otherwise the CSV at `fixations_path`.
    3. Else the precomputed statistics CSV at `stats_csv_path` is loaded.

    THERE IS NO DEFAULT SOURCE, deliberately. This function used to default to
    L1's answer-screen fixation report, so any caller that forgot to say where
    its baseline came from silently normalised against L1's participants -- for
    a different dataset that is not merely the wrong screen, it is a different
    set of people (docs/todo.md T3.20). Callers name the baseline they mean, or
    this raises.

    Returns a DataFrame with columns [group_col, pupil_mean, pupil_sd].
    """
    if isinstance(stats, pd.DataFrame):
        return stats

    if compute:
        if isinstance(fixations, pd.DataFrame):
            if verbose:
                print("Computing participant pupil stats from preloaded fixations")
            return compute_participant_pupil_stats(fixations, group_col=group_col)
        if fixations_path is None:
            raise ValueError(
                "No pupil baseline source given. Pass one of: `stats` (a resolved "
                "stats frame), `fixations` (a loaded fixation report), "
                "`fixations_path` (a fixation report to compute from), or "
                "`compute=False` with `stats_csv_path`. There is no default -- "
                "see docs/todo.md T3.20 for why."
            )
        if verbose:
            print(f"Computing participant pupil stats from: {fixations_path}")
        return compute_participant_pupil_stats(
            pd.read_csv(fixations_path), group_col=group_col
        )

    if stats_csv_path is None:
        raise ValueError(
            "compute=False but no `stats_csv_path` given, and there is no default "
            "stats file -- name the dataset's own pupil-stats CSV."
        )
    if verbose:
        print(f"Loading participant pupil stats from: {stats_csv_path}")
    return pd.read_csv(stats_csv_path)


def zscore_pupil_by_participant(
    df: pd.DataFrame,
    pupil_col: str,
    participant_col: str,
    stats,
    out_col: str = None,
) -> pd.DataFrame:
    """
    Z-score pupil values using participant-level mean/std.

    pupil_z = (pupil - participant_mean) / participant_std

    `stats` may be either a resolved statistics DataFrame (with columns
    participant_col, "pupil_mean", "pupil_sd") or a path to such a CSV. It is
    REQUIRED -- it used to default to L1's stats file, which meant a caller
    working on another dataset could normalise against L1's people without
    saying so (docs/todo.md T3.20).
    """
    if out_col is None:
        out_col = f"{pupil_col}_z"

    if isinstance(stats, pd.DataFrame):
        stats_df = stats
    else:
        stats_df = pd.read_csv(stats)

    stats_df = stats_df[[participant_col, "pupil_mean", "pupil_sd"]]

    # Every person must have a usable baseline. Without these checks the left
    # merge below turns a missing one into NaN mean/SD and therefore an all-NaN
    # `_z` column -- no error, no count, just a feature that quietly isn't there.
    # The sharp case is a stats table from the WRONG DATASET: no id matches, so
    # every z goes NaN rather than merely wrong. Fail loudly instead.
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
            "baseline -- see docs/todo.md T3.20."
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
# Moved here from data_prep/data_csv_generation.py in stage C (2026-10-06).
# These are the registry's `create_*` / `add_*` entries -- the functions the
# prep pipeline calls to put columns on the IA frame. They mutate the frame
# they are handed and depend on registry order; that is kept as-is by decision
# (Diana, 2026-10-06), with the default runner executing them in that order.
# ---------------------------------------------------------------------------



def prepare_screen_pupil(
    df: pd.DataFrame,
    pupil_stats: pd.DataFrame,
    pupil_cols=None,
) -> pd.DataFrame:
    """Coerce the measure columns once, then scale and z-score the pupil columns.

    **One implementation for both screens** (stage D step 2b). This is the
    paragraph path's version, adopted rather than the answer path's, because it
    coerces every metric column **once, up front** instead of letting each metric
    function coerce on the way past -- which is the arrangement `todo.md` T3.11
    wants the answer side to reach, and the reason the two screens can share one
    set of metric builders at all.

    `pupil_stats` is that screen's own per-participant baseline (T3.20: each
    screen baselines against its own fixations, never the other's).
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
    """
    1) Convert IA pupil columns to mm (stored back into original columns)
    2) Z-score them using participant stats
    3) Store z-scored values into new <column>_z columns

    `pupil_stats` is REQUIRED and must be this dataset's own baseline -- main()
    resolves it once from `pupil_fixations_path` and injects it here. There is no
    fallback: the resolver used to default to L1's answer-screen fixation report,
    so calling this without stats on any other dataset normalised against L1's
    people (docs/todo.md T3.20).
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
    """Per-area means of the pupil-size columns, raw (mm) and z-scored.

    Coerces locally on a copy rather than through `coerce_ia_columns(pupil=True)`,
    because the `_z` columns were produced earlier by `add_zscored_pupil_columns`
    and are not raw report text -- both families still need `to_numeric`, but
    neither should be written back onto the caller's frame.
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
    """Pupil size at the first fixated word of each area.

    Reads IA_FIRST_FIXATION_DURATION as a number. The column arrives from the
    report as text carrying "." sentinels; it is resolved once by
    `generate_new_row_features` before any group function runs, so this no
    longer depends on `create_mean_first_fix_duration` having gone first and
    `group_function_names=[...]` on a subset works (`todo.md` T3.11, fixed
    2026-09-27). If the frame does reach here uncoerced, `area_metrics`
    raises and names the missing step rather than comparing str with int.
    """
    df_local = df.copy()

    mm_col = C.IA_AVERAGE_FIX_PUPIL_SIZE
    z_col = f"{mm_col}_z"
    df_local[mm_col] = pd.to_numeric(df_local[mm_col], errors="coerce")
    df_local[z_col] = pd.to_numeric(df_local[z_col], errors="coerce")

    return am.first_encounter_pupil_size(
        df_local, C.AREA_LABEL_COLUMN, include_raw=True, include_z=True
    )
