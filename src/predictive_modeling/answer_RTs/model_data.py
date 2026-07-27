# model_data.py
#
# Assemble the trial-level modeling frame for the answer reading-time regression.
#
# Target: reading time on a single answer (A / B / C / D).
# Predictors: paragraph-only features -- deliberately NO answer-area features.
#   1. Per-span area metrics (mean_dwell_time__critical, ...) from features.py.
#   2. Per-span reading times / total fixation durations, pure and normalized
#      (RT_pure_critical, RT_normalized_critical, TFD_pure_critical, ...).
#   3. The question-preview flag (hunter = 1 / gatherer = 0).
#
# One PreparedTrialDataset is built per answer via `make_answer_rt_dataset`; the
# four answers share the same feature matrix and differ only in the target.
#
# Feature-group getters mirror `answer_correctness/../feature_specs.py` so further
# feature variants (e.g. dropping pupil metrics, or span subsets) can be layered
# on by composing column lists.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src import constants as Con
from src.constants import TRIAL_ID_COLS
from src.data_paths import PARAGRAPH_SPAN_FEATURES_PATH, READY_ALL_FEATURES_PATH
from src.predictive_modeling.common.prepared_dataset import PreparedTrialDataset
from src.predictive_modeling.answer_RTs.features import (
    PARAGRAPH_METRIC_COLUMNS,
    PARAGRAPH_SPANS,
    QUESTION_PREVIEW_COL,
    load_paragraph_features,
)

# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------

ANSWER_LABELS = ("A", "B", "C", "D")

# Per-span reading-time / total-fixation-duration features (paragraph regions),
# in both pure and normalized forms. These live in the L1 model-ready file but
# are computed from the paragraph screen, not the answer areas.
SPAN_RT_TFD_METRICS = ("RT_pure", "RT_normalized", "TFD_pure", "TFD_normalized")

# Default reading-time metric used for the answer target.
DEFAULT_TARGET_RT_METRIC = "RT_normalized"


def answer_rt_target_col(
    answer: str,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
) -> str:
    """Target column for one answer, e.g. ('A', 'RT_normalized') -> RT_normalized_answer_A."""
    return f"{rt_metric}_answer_{answer}"


def get_span_area_feature_cols(df: pd.DataFrame) -> List[str]:
    """Per-span area-metric columns (``<metric>__<span>``) present in df."""
    return [
        f"{metric}__{span}"
        for metric in PARAGRAPH_METRIC_COLUMNS
        for span in PARAGRAPH_SPANS
        if f"{metric}__{span}" in df.columns
    ]


def get_span_rt_tfd_feature_cols(df: pd.DataFrame) -> List[str]:
    """Per-span RT/TFD columns (pure + normalized) present in df."""
    return [
        f"{metric}_{span}"
        for metric in SPAN_RT_TFD_METRICS
        for span in PARAGRAPH_SPANS
        if f"{metric}_{span}" in df.columns
    ]


def get_preview_feature_cols(df: pd.DataFrame) -> List[str]:
    """The question-preview (hunter/gatherer) flag, if present."""
    return [QUESTION_PREVIEW_COL] if QUESTION_PREVIEW_COL in df.columns else []


def get_paragraph_feature_cols(df: pd.DataFrame) -> List[str]:
    """Full paragraph-only predictor set: span area metrics + span RT/TFD + preview.

    Contains no answer-area features by construction.
    """
    return (
        get_span_area_feature_cols(df)
        + get_span_rt_tfd_feature_cols(df)
        + get_preview_feature_cols(df)
    )


# ---------------------------------------------------------------------------
# Modeling dataframe assembly
# ---------------------------------------------------------------------------

def _load_span_rt_tfd_and_targets(
    ready_features_path: Path,
    rt_metrics: Sequence[str],
) -> pd.DataFrame:
    """Load span RT/TFD features and answer-RT targets from the L1 file.

    Only paragraph-span RT/TFD columns and the answer-RT target columns are
    selected -- no answer-area predictor columns are pulled in.
    """
    header = pd.read_csv(ready_features_path, nrows=0).columns
    span_cols = [
        f"{metric}_{span}"
        for metric in SPAN_RT_TFD_METRICS
        for span in PARAGRAPH_SPANS
        if f"{metric}_{span}" in header
    ]
    target_cols = [
        f"{metric}_answer_{answer}"
        for metric in rt_metrics
        for answer in ANSWER_LABELS
        if f"{metric}_answer_{answer}" in header
    ]
    usecols = list(TRIAL_ID_COLS) + span_cols + target_cols
    return pd.read_csv(ready_features_path, usecols=usecols)


def build_answer_rt_model_df(
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    ready_features_path: Path = READY_ALL_FEATURES_PATH,
    target_rt_metrics: Sequence[str] = ("RT_normalized", "RT_pure"),
) -> pd.DataFrame:
    """One row per trial: paragraph predictors + answer-RT targets.

    Inner-joins the paragraph-span features (area metrics + preview flag) with
    the paragraph-span RT/TFD features and answer-RT targets, keyed on
    (participant_id, TRIAL_INDEX). Trials without both sides (e.g. practice or
    repeated-reading paragraph trials that have no answer screen) are dropped.
    """
    if paragraph_features is None:
        paragraph_features = load_paragraph_features(paragraph_features_path)

    rt_tfd_and_targets = _load_span_rt_tfd_and_targets(
        ready_features_path=ready_features_path,
        rt_metrics=target_rt_metrics,
    )

    model_df = paragraph_features.merge(
        rt_tfd_and_targets,
        on=list(TRIAL_ID_COLS),
        how="inner",
    )
    return model_df


def make_answer_rt_dataset(
    model_df: pd.DataFrame,
    answer: str,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
    feature_cols: Optional[Sequence[str]] = None,
) -> PreparedTrialDataset:
    """Build a PreparedTrialDataset for one answer's reading-time regression.

    Rows with a missing target are dropped. Feature columns default to the full
    paragraph-only predictor set.
    """
    target_col = answer_rt_target_col(answer, rt_metric=rt_metric)
    if target_col not in model_df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    cols = (
        list(feature_cols)
        if feature_cols is not None
        else get_paragraph_feature_cols(model_df)
    )

    df = model_df.dropna(subset=[target_col]).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return PreparedTrialDataset(
        df=df,
        feature_cols=cols,
        target_col=target_col,
        id_cols=list(TRIAL_ID_COLS),
    )
