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
# `build_external_answer_rt_model_df` is the same frame with the lab's external
# EyeBench trial-level features as predictors instead of ours -- same targets,
# same key, so both go through `run_answer_rt_regression` unchanged.
#
# Feature-group getters mirror `answer_correctness/../feature_specs.py` so further
# feature variants (e.g. dropping pupil metrics, or span subsets) can be layered
# on by composing column lists.

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
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

def _target_cols_present(header: Sequence[str], rt_metrics: Sequence[str]) -> List[str]:
    """The answer-RT target columns for `rt_metrics` that the L1 file actually has."""
    return [
        f"{metric}_answer_{answer}"
        for metric in rt_metrics
        for answer in ANSWER_LABELS
        if f"{metric}_answer_{answer}" in header
    ]


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
    usecols = (
        list(TRIAL_ID_COLS) + span_cols + _target_cols_present(header, rt_metrics)
    )
    return pd.read_csv(ready_features_path, usecols=usecols)


def _load_answer_rt_targets(
    ready_features_path: Path,
    rt_metrics: Sequence[str],
) -> pd.DataFrame:
    """Trial ids + the answer-RT target columns only, no predictors."""
    header = pd.read_csv(ready_features_path, nrows=0).columns
    usecols = list(TRIAL_ID_COLS) + _target_cols_present(header, rt_metrics)
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


# ---------------------------------------------------------------------------
# External (EyeBench) trial-level feature variant
# ---------------------------------------------------------------------------

# Target columns are named `<metric>_answer_<A..D>`; everything else that comes
# out of the external extraction is a candidate predictor.
_ANSWER_TARGET_SUFFIX = re.compile(r"_answer_[A-D]$")


def get_external_feature_cols(
    df: pd.DataFrame,
    drop_constant: bool = True,
) -> List[str]:
    """Predictor columns of an external-feature model frame.

    Everything numeric except the trial ids and the `*_answer_<A..D>` targets,
    so the ~500 EyeBench trial-level features plus the preview flag. Columns
    that are all-NaN or take a single value carry no signal (the extraction
    leaves ~145 of those, e.g. `min` of count columns); `drop_constant` removes
    them so the coefficient table is not padded with zeros.
    """
    cols = [
        c
        for c in df.columns
        if c not in TRIAL_ID_COLS
        and not _ANSWER_TARGET_SUFFIX.search(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if drop_constant:
        cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def build_external_answer_rt_model_df(
    trial_features: Optional[pd.DataFrame] = None,
    ready_features_path: Path = READY_ALL_FEATURES_PATH,
    target_rt_metrics: Sequence[str] = ("RT_normalized", "RT_pure"),
    **feature_kwargs,
) -> pd.DataFrame:
    """One row per trial: EyeBench trial-level features + answer-RT targets.

    The counterpart of `build_answer_rt_model_df` for the lab's external
    extraction: same targets, same (participant_id, TRIAL_INDEX) key, but the
    predictors are the trial-level paragraph features instead of our span
    metrics. Still paragraph-only -- the extraction runs on the paragraph
    reports, so no answer-area information gets in.

    Args:
        trial_features: an already-loaded feature table; loaded from the cached
            CSV when omitted.
        feature_kwargs: forwarded to `get_paragraph_trial_features` (e.g.
            `quick_test`), only used when `trial_features` is None.
    """
    if trial_features is None:
        from src.derived.external.EyeBench.runner import get_paragraph_trial_features

        trial_features = get_paragraph_trial_features(**feature_kwargs)

    if set(TRIAL_ID_COLS) - set(trial_features.columns):
        # A freshly built table is indexed by (participant_id, TRIAL_INDEX);
        # the cached CSV carries them as plain columns.
        trial_features = trial_features.reset_index()

    targets = _load_answer_rt_targets(
        ready_features_path=ready_features_path,
        rt_metrics=target_rt_metrics,
    )
    return trial_features.merge(targets, on=list(TRIAL_ID_COLS), how="inner")


def make_answer_rt_dataset(
    model_df: pd.DataFrame,
    answer: str,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
    feature_cols: Optional[Sequence[str]] = None,
    log_target: bool = False,
) -> PreparedTrialDataset:
    """Build a PreparedTrialDataset for one answer's reading-time regression.

    Rows with a missing target are dropped. Feature columns default to the full
    paragraph-only predictor set.

    `log_target` regresses on `log_<target>` instead of the raw reading time.
    Raw answer RTs are strongly right-skewed (skew 3-14), which a squared-error
    fit spends most of its capacity on; in logs they are near-symmetric. Trials
    with a zero RT -- the answer was never fixated, 0.3-2.8% depending on the
    answer -- are dropped rather than shifted: they are a skip, not a fast read,
    and `log1p` would place them 7 log-units below every real observation.
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

    if log_target:
        df = df[df[target_col] > 0].reset_index(drop=True)
        log_col = pd.Series(np.log(df[target_col].to_numpy()), name=f"log_{target_col}")
        df = pd.concat([df, log_col], axis=1)
        target_col = str(log_col.name)

    return PreparedTrialDataset(
        df=df,
        feature_cols=cols,
        target_col=target_col,
        id_cols=list(TRIAL_ID_COLS),
    )
