# model_data.py
#
# Assemble the trial-level modeling frame for the answer reading-time regression.
#
# Target: reading time on a single answer (A / B / C / D).
# Predictors: paragraph-only features -- deliberately NO answer-area eye-tracking
# features.
#   1. Per-span area metrics (mean_dwell_time__critical, ...) from features.py.
#   2. Per-span reading times / total fixation durations, pure and normalized
#      (RT_pure_critical, RT_normalized_critical, TFD_pure_critical, ...).
#   3. The question-preview flag (hunter = 1 / gatherer = 0).
#   4. Answer/question text sizes from answer_text_features.py -- stimulus
#      properties (how much text is on the screen), not measures of how the
#      answer area was read, so the paragraph-only rule still holds.
#
# One PreparedTrialDataset is built per answer via `make_answer_rt_dataset`. The
# four answers share the eye-tracking feature matrix; the text-size features are
# target-aligned, so `answer_len_words` means "length of the answer being
# predicted" and does differ between the four datasets.
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
from src.data_paths import (
    ANSWER_TEXT_FEATURES_PATH,
    PARAGRAPH_SPAN_FEATURES_PATH,
    READY_ALL_FEATURES_PATH,
)
from src.predictive_modeling.common.prepared_dataset import PreparedTrialDataset
from src.predictive_modeling.answer_RTs.answer_text_features import (
    ANSWER_TEXT_FEATURE_COLS,
    answer_len_col,
    load_answer_text_features,
)
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

# Text-size predictors, named relative to whichever answer is being predicted
# (materialized per dataset by `add_answer_text_features`).
TARGET_ALIGNED_TEXT_FEATURE_COLS = [
    "answer_len_words",       # words in the target answer
    "answer_len_chars",       # characters in the target answer
    "answer_len_words_rel",   # target answer length / mean length of the four
    "answers_total_len_words",  # total words across the four options
    "answers_std_len_words",  # spread of option lengths on the screen
    "question_len_words",
]


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


def get_answer_text_feature_cols(df: pd.DataFrame) -> List[str]:
    """Target-aligned answer/question text-size columns present in df.

    These are the columns `add_answer_text_features` materializes for one
    answer, not the raw per-label ones -- see that function for the naming.
    """
    return [c for c in TARGET_ALIGNED_TEXT_FEATURE_COLS if c in df.columns]


def get_paragraph_feature_cols(
    df: pd.DataFrame,
    include_answer_text: bool = True,
) -> List[str]:
    """Full paragraph-only predictor set: span area metrics + span RT/TFD + preview.

    With `include_answer_text`, the target-aligned text-size columns are appended
    (they are only present after `add_answer_text_features` has run, i.e. inside
    `make_answer_rt_dataset`). Contains no answer-area eye-tracking features by
    construction.
    """
    cols = (
        get_span_area_feature_cols(df)
        + get_span_rt_tfd_feature_cols(df)
        + get_preview_feature_cols(df)
    )
    if include_answer_text:
        cols += get_answer_text_feature_cols(df)
    return cols


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


def _merge_answer_text_features(
    model_df: pd.DataFrame,
    answer_text_features: Optional[pd.DataFrame],
    answer_text_features_path: Path,
) -> pd.DataFrame:
    """Left-join the per-label answer/question text sizes onto a model frame.

    Left, not inner: a trial missing from the text table keeps its eye-tracking
    features and gets NaN lengths, rather than silently dropping out of the
    sample.
    """
    if answer_text_features is None:
        answer_text_features = load_answer_text_features(answer_text_features_path)

    cols = list(TRIAL_ID_COLS) + [
        c for c in ANSWER_TEXT_FEATURE_COLS if c in answer_text_features.columns
    ]
    return model_df.merge(
        answer_text_features[cols],
        on=list(TRIAL_ID_COLS),
        how="left",
    )


def build_answer_rt_model_df(
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    ready_features_path: Path = READY_ALL_FEATURES_PATH,
    target_rt_metrics: Sequence[str] = ("RT_normalized", "RT_pure"),
    answer_text_features: Optional[pd.DataFrame] = None,
    answer_text_features_path: Path = ANSWER_TEXT_FEATURES_PATH,
    include_answer_text: bool = True,
) -> pd.DataFrame:
    """One row per trial: paragraph predictors + answer-RT targets.

    Inner-joins the paragraph-span features (area metrics + preview flag) with
    the paragraph-span RT/TFD features and answer-RT targets, keyed on
    (participant_id, TRIAL_INDEX). Trials without both sides (e.g. practice or
    repeated-reading paragraph trials that have no answer screen) are dropped.

    `include_answer_text` additionally left-joins the per-label answer/question
    text sizes; `make_answer_rt_dataset` turns those into target-aligned
    columns.
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

    if include_answer_text:
        model_df = _merge_answer_text_features(
            model_df,
            answer_text_features=answer_text_features,
            answer_text_features_path=answer_text_features_path,
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

    The raw per-label text sizes (`answer_len_words_A`, ...) are excluded here:
    `make_answer_rt_dataset` adds their target-aligned counterparts instead, so
    the external set gets the same text features as the paragraph-span set
    rather than all four answers' lengths regardless of the target.
    """
    cols = [
        c
        for c in df.columns
        if c not in TRIAL_ID_COLS
        and c not in ANSWER_TEXT_FEATURE_COLS
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
    answer_text_features: Optional[pd.DataFrame] = None,
    answer_text_features_path: Path = ANSWER_TEXT_FEATURES_PATH,
    include_answer_text: bool = True,
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
    model_df = trial_features.merge(targets, on=list(TRIAL_ID_COLS), how="inner")

    if include_answer_text:
        model_df = _merge_answer_text_features(
            model_df,
            answer_text_features=answer_text_features,
            answer_text_features_path=answer_text_features_path,
        )
    return model_df


def add_answer_text_features(df: pd.DataFrame, answer: str) -> pd.DataFrame:
    """Materialize the text-size predictors for one answer.

    The cached table holds one column per label (`answer_len_words_A`, ...);
    a model predicting answer A's reading time should see A's length, not all
    four. This renames the target answer's columns to the target-aligned names
    in `TARGET_ALIGNED_TEXT_FEATURE_COLS` and derives the relative length.
    Returns `df` unchanged if the raw per-label columns are absent.
    """
    words_col = answer_len_col(answer, "words")
    chars_col = answer_len_col(answer, "chars")
    if words_col not in df.columns:
        return df

    out = df.copy()
    out["answer_len_words"] = pd.to_numeric(out[words_col], errors="coerce")
    if chars_col in out.columns:
        out["answer_len_chars"] = pd.to_numeric(out[chars_col], errors="coerce")

    if "answers_mean_len_words" in out.columns:
        mean_len = pd.to_numeric(out["answers_mean_len_words"], errors="coerce")
        out["answer_len_words_rel"] = out["answer_len_words"] / mean_len.replace(0, np.nan)

    return out


def make_answer_rt_dataset(
    model_df: pd.DataFrame,
    answer: str,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
    feature_cols: Optional[Sequence[str]] = None,
    log_target: bool = False,
    include_answer_text: bool = True,
) -> PreparedTrialDataset:
    """Build a PreparedTrialDataset for one answer's reading-time regression.

    Rows with a missing target are dropped. Feature columns default to the full
    paragraph-only predictor set.

    `include_answer_text` materializes this answer's text-size columns (see
    `add_answer_text_features`) and appends them to `feature_cols`, so an
    explicit `feature_cols` -- e.g. the external EyeBench set -- gets them too.
    Note that the default target `RT_normalized_*` is already RT divided by the
    answer's word count, so length has largely been divided out of it; the text
    features bite hardest against `RT_pure_*`.

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

    frame = (
        add_answer_text_features(model_df, answer)
        if include_answer_text
        else model_df
    )

    cols = (
        list(feature_cols)
        if feature_cols is not None
        else get_paragraph_feature_cols(frame, include_answer_text=False)
    )
    if include_answer_text:
        cols += [c for c in get_answer_text_feature_cols(frame) if c not in cols]
    else:
        # The toggle has to be exhaustive at this end, not just skip the
        # per-answer columns: an explicit `feature_cols` built by
        # `get_paragraph_feature_cols` already carries the three trial-level
        # sizes (answers_total/std_len_words, question_len_words), so leaving
        # them in would make the "no text features" run still a text run.
        text_cols = set(ANSWER_TEXT_FEATURE_COLS) | set(
            TARGET_ALIGNED_TEXT_FEATURE_COLS
        )
        cols = [c for c in cols if c not in text_cols]

    df = frame.dropna(subset=[target_col]).reset_index(drop=True)
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
