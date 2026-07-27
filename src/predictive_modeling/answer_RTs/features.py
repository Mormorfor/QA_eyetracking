# features.py
#
# Paragraph-span features for the answer reading-time (answer_RTs) regression.
#
# Goal: predict reading times on the answers from features describing how the
# participant read the *paragraph*. This module extracts, per paragraph span
# (critical / distractor / outside), the same per-area eye-tracking metrics that
# `answer_correctness` extracts per answer area (answer_A..D / question) --
# mean dwell time, fixation count, first-fixation duration, skip rate, dwell
# proportion, pupil-size (z) metrics and label-visit counts -- producing one row
# per trial with columns of the form ``<metric>__<span>`` (e.g.
# ``mean_dwell_time__critical``).
#
# The metrics mirror the definitions in `src/data_prep/data_csv_generation.py`,
# but grouped by the paragraph span column (`auxiliary_span_type`) instead of the
# answer `area_label`. This is the feature side of the module; models and further
# feature variants live alongside it (see the package structure notes in the
# module docstring of `model_data.py`).

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src import constants as Con
from src.constants import TRIAL_ID_COLS
from src.data_paths import IA_PARAGRAPH_PATH, PARAGRAPH_SPAN_FEATURES_PATH
from src.derived.pupil_norm import (
    get_participant_pupil_stats,
    scale_pupil_area_to_mm,
    zscore_pupil_by_participant,
)
from src.predictive_modeling.common.feature_builders import build_area_metric_pivot

# ---------------------------------------------------------------------------
# Paragraph span definitions
# ---------------------------------------------------------------------------

# Column in the paragraph IA report that tags each interest area with its span.
PARAGRAPH_SPAN_COL = "auxiliary_span_type"
PARAGRAPH_SPANS = ("outside", "distractor", "critical")

# The per-span metrics produced, matching Con.AREA_METRIC_COLUMNS_MODELING so the
# resulting columns line up 1:1 with the per-answer area features.
PARAGRAPH_METRIC_COLUMNS = list(Con.AREA_METRIC_COLUMNS_MODELING)

# Trial-level paragraph-screen attribute carried through as a feature: the
# question-preview condition (True for hunters, False for gatherers), exposed as
# a 0/1 column so a single model can distinguish the two reading regimes.
QUESTION_PREVIEW_COL = Con.QUESTION_PREVIEW_COLUMN

_PUPIL_RAW_COLS = (
    Con.IA_MAX_FIX_PUPIL_SIZE,
    Con.IA_MIN_FIX_PUPIL_SIZE,
    Con.IA_AVERAGE_FIX_PUPIL_SIZE,
)

_SPAN_GROUP_COLS = list(TRIAL_ID_COLS) + [PARAGRAPH_SPAN_COL]


# ---------------------------------------------------------------------------
# Per-span metric builders (one row per participant-trial-span)
#
# Each mirrors a create_* function in data_csv_generation, but grouped by the
# paragraph span column rather than the answer area_label.
# ---------------------------------------------------------------------------

def _mean_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(_SPAN_GROUP_COLS, as_index=False).agg(
        **{Con.MEAN_DWELL_TIME: (Con.IA_DWELL_TIME, "mean")}
    )


def _mean_fixations_count(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(_SPAN_GROUP_COLS, as_index=False).agg(
        **{Con.MEAN_FIXATIONS_COUNT: (Con.IA_FIXATIONS_COUNT, "mean")}
    )


def _mean_first_fix_duration(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(_SPAN_GROUP_COLS, as_index=False).agg(
        **{Con.MEAN_FIRST_FIXATION_DURATION: (Con.IA_FIRST_FIXATION_DURATION, "mean")}
    )


def _skip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """A skip is an IA with zero dwell time; skip_rate is the per-span mean."""
    d = df[_SPAN_GROUP_COLS + [Con.IA_DWELL_TIME]].copy()
    d[Con.AREA_SKIPPED] = (d[Con.IA_DWELL_TIME] == 0).astype(int)
    return d.groupby(_SPAN_GROUP_COLS, as_index=False).agg(
        **{Con.SKIP_RATE: (Con.AREA_SKIPPED, "mean")}
    )


def _dwell_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """Per-span dwell time as a fraction of the whole trial's dwell time."""
    agg = (
        df.groupby(_SPAN_GROUP_COLS, as_index=False)
        .agg({Con.IA_DWELL_TIME: "sum"})
        .rename(columns={Con.IA_DWELL_TIME: Con.TOTAL_IA_DWELL_TIME})
    )
    agg[Con.TOTAL_TRIAL_DWELL_TIME] = agg.groupby(list(TRIAL_ID_COLS))[
        Con.TOTAL_IA_DWELL_TIME
    ].transform("sum")
    agg[Con.AREA_DWELL_PROPORTION] = (
        agg[Con.TOTAL_IA_DWELL_TIME] / agg[Con.TOTAL_TRIAL_DWELL_TIME]
    ).fillna(0)
    return agg[_SPAN_GROUP_COLS + [Con.AREA_DWELL_PROPORTION]]


def _mean_pupil_z(df: pd.DataFrame) -> pd.DataFrame:
    """Per-span means of the z-scored pupil-size columns (max / min / avg)."""
    return df.groupby(_SPAN_GROUP_COLS, as_index=False).agg(
        **{
            Con.MEAN_MAX_FIX_PUPIL_SIZE_Z: (f"{Con.IA_MAX_FIX_PUPIL_SIZE}_z", "mean"),
            Con.MEAN_MIN_FIX_PUPIL_SIZE_Z: (f"{Con.IA_MIN_FIX_PUPIL_SIZE}_z", "mean"),
            Con.MEAN_AVG_FIX_PUPIL_SIZE_Z: (f"{Con.IA_AVERAGE_FIX_PUPIL_SIZE}_z", "mean"),
        }
    )


def _first_encounter_pupil_z(df: pd.DataFrame) -> pd.DataFrame:
    """Avg-pupil (z) of the first fixated IA (in reading order) of each span."""
    z_col = f"{Con.IA_AVERAGE_FIX_PUPIL_SIZE}_z"
    d = df[df[Con.IA_FIRST_FIXATION_DURATION] > 0]
    d = d.sort_values(list(TRIAL_ID_COLS) + [Con.INTEREST_AREA_ID])
    first_fix = d.groupby(_SPAN_GROUP_COLS, as_index=False).head(1)
    return first_fix[_SPAN_GROUP_COLS + [z_col]].rename(
        columns={z_col: Con.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z}
    )


def _num_span_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Count visits to each span from the trial's fixation sequence.

    The fixation sequence (a list of IA_IDs) is mapped to spans, consecutive
    repeats are collapsed into a single visit, and the visits per span are
    counted -- mirroring `create_simplified_visit_counts` for the answers.
    """
    rows = []
    for (pid, tid), g in df.groupby(list(TRIAL_ID_COLS), sort=False):
        ia_to_span = dict(zip(g[Con.INTEREST_AREA_ID], g[PARAGRAPH_SPAN_COL]))

        seq_raw = g[Con.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        try:
            seq = ast.literal_eval(seq_raw) if isinstance(seq_raw, str) else seq_raw
        except (ValueError, SyntaxError):
            seq = []
        if not isinstance(seq, (list, tuple)):
            seq = []

        spans = [ia_to_span.get(ia) for ia in seq]
        collapsed = [s for i, s in enumerate(spans) if s is not None and (i == 0 or s != spans[i - 1])]
        counts = Counter(collapsed)

        for span in PARAGRAPH_SPANS:
            rows.append(
                {
                    Con.PARTICIPANT_ID: pid,
                    Con.TRIAL_ID: tid,
                    PARAGRAPH_SPAN_COL: span,
                    Con.NUM_LABEL_VISITS: int(counts.get(span, 0)),
                }
            )
    return pd.DataFrame(rows, columns=_SPAN_GROUP_COLS + [Con.NUM_LABEL_VISITS])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _prepare_paragraph_ia(
    paragraph_ia: pd.DataFrame,
    pupil_stats=None,
) -> pd.DataFrame:
    """Coerce metric source columns to numeric and add z-scored pupil columns.

    Pupil z-scoring uses the same per-participant statistics as the answer
    pipeline (resolved via `get_participant_pupil_stats`), so paragraph and
    answer pupil-z values share a participant baseline.
    """
    df = paragraph_ia.copy()

    numeric_cols = [
        Con.IA_DWELL_TIME,
        Con.IA_FIXATIONS_COUNT,
        Con.IA_FIRST_FIXATION_DURATION,
        *_PUPIL_RAW_COLS,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert pupil areas to mm and z-score per participant (mirrors
    # add_zscored_pupil_columns in the answer pipeline).
    pupil_stats = get_participant_pupil_stats(stats=pupil_stats)
    for col in _PUPIL_RAW_COLS:
        df[col] = scale_pupil_area_to_mm(df[col])
        df = zscore_pupil_by_participant(
            df=df,
            pupil_col=col,
            participant_col=Con.PARTICIPANT_ID,
            stats=pupil_stats,
            out_col=f"{col}_z",
        )

    return df


def build_paragraph_span_metrics(
    paragraph_ia: pd.DataFrame,
    pupil_stats=None,
) -> pd.DataFrame:
    """Build a per-(participant, trial, span) table of all paragraph metrics.

    Columns: TRIAL_ID_COLS + [PARAGRAPH_SPAN_COL] + PARAGRAPH_METRIC_COLUMNS.
    """
    df = _prepare_paragraph_ia(paragraph_ia, pupil_stats=pupil_stats)

    parts = [
        _mean_dwell_time(df),
        _mean_fixations_count(df),
        _mean_first_fix_duration(df),
        _skip_rate(df),
        _dwell_proportion(df),
        _mean_pupil_z(df),
        _first_encounter_pupil_z(df),
        _num_span_visits(df),
    ]

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=_SPAN_GROUP_COLS, how="outer")

    return out[_SPAN_GROUP_COLS + PARAGRAPH_METRIC_COLUMNS]


def build_trial_level_paragraph_features(
    paragraph_ia: Optional[pd.DataFrame] = None,
    paragraph_ia_path: Path = IA_PARAGRAPH_PATH,
    metric_cols: Sequence[str] = PARAGRAPH_METRIC_COLUMNS,
    pupil_stats=None,
) -> pd.DataFrame:
    """One row per trial with ``<metric>__<span>`` paragraph features.

    `paragraph_ia` may be passed in-memory; otherwise it is read from
    `paragraph_ia_path`. The output pivots spans into feature columns exactly
    like the per-answer area pivot, e.g. ``mean_dwell_time__critical``.
    """
    if paragraph_ia is None:
        paragraph_ia = pd.read_csv(paragraph_ia_path)

    span_metrics = build_paragraph_span_metrics(paragraph_ia, pupil_stats=pupil_stats)

    pivot = build_area_metric_pivot(
        df=span_metrics,
        area_col=PARAGRAPH_SPAN_COL,
        metric_cols=list(metric_cols),
    )

    preview = _trial_question_preview(paragraph_ia)
    pivot = pivot.merge(preview, on=list(TRIAL_ID_COLS), how="left")
    return pivot


def _trial_question_preview(paragraph_ia: pd.DataFrame) -> pd.DataFrame:
    """One row per trial with the question-preview flag as a 0/1 int column."""
    preview = (
        paragraph_ia[list(TRIAL_ID_COLS) + [QUESTION_PREVIEW_COL]]
        .groupby(list(TRIAL_ID_COLS), as_index=False)
        .first()
    )
    preview[QUESTION_PREVIEW_COL] = (
        preview[QUESTION_PREVIEW_COL].astype("boolean").astype("Int64")
    )
    return preview


# ---------------------------------------------------------------------------
# Cached feature CSV
# ---------------------------------------------------------------------------

def save_paragraph_features(
    paragraph_ia: Optional[pd.DataFrame] = None,
    output_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    paragraph_ia_path: Path = IA_PARAGRAPH_PATH,
    pupil_stats=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the trial-level paragraph features and save them to CSV."""
    features = build_trial_level_paragraph_features(
        paragraph_ia=paragraph_ia,
        paragraph_ia_path=paragraph_ia_path,
        pupil_stats=pupil_stats,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    if verbose:
        print(
            f"Saved {len(features)} trials x {len(features.columns)} cols "
            f"to {output_path}"
        )
    return features


def load_paragraph_features(
    path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
) -> pd.DataFrame:
    """Load the cached paragraph features produced by `save_paragraph_features`."""
    return pd.read_csv(Path(path))


if __name__ == "__main__":
    save_paragraph_features()
