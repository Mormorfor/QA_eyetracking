"""IA tables -> the model-ready trial-level table.

**This is where word grain becomes trial grain.** Everything above it in
`features/` measures things per word or per area; this module pivots those to one
column per (metric x area), derives the correct/wrong contrasts, collapses to one
row per trial, and merges the families computed elsewhere (reading times,
sequences, strategies, preference, the paragraph cache).

Assembled here in stage D step 2 (2026-10-07). Before that it was split in two,
for no reason either half could state:

* `predictive_modeling/common/feature_builders.py` held the pivot and the
  contrast columns -- generic feature construction, sitting in a *modelling*
  package. `features/paragraph/spans.py` had to reach **upward** into
  `predictive_modeling/` to get the pivot, which is a layering inversion that
  made this move worth doing on its own.
* `answer_correctness/model_data.py` held the six `build_trial_level_*` builders
  and the grain collapse -- so "which features exist" lived inside the module
  that decides which of them a model uses. Those are different questions
  (stage D proposal section 3.3).

Step 7 then emptied `model_data.py` entirely: the cache pair moved here as
`save_model_ready` / `load_model_ready` (bottom of this file), the three
`make_*_dataset` helpers were deleted as unused, and the file is gone.

**Layering:** this module imports from `features/` and `config/` and nothing
else. `features/` must not import `ingest/` or `predictive_modeling/`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import (
    PARAGRAPH_SPAN_FEATURES_PATH,
    READY_ALL_FEATURES_PATH,
    dataset as _dataset,
)
from src.features.preference import compute_trial_matching
from src.features.reading_times import (
    ANSWER_REGIONS as RT_ANSWER_REGIONS,
    PARAGRAPH_REGIONS as RT_PARAGRAPH_REGIONS,
)
from src.features.sequences import (
    compute_trial_mean_dwell_per_word,
    has_back_and_forth_xyx,
    has_back_and_forth_xyxy,
    longest_alternating_answer_run,
    parse_seq,
    sequence_len_literal_eval,
)
from src.features.strategies import build_trial_level_pattern_features


# ---------------------------------------------------------------------------
# Column vocabularies these builders produce. They moved with the builders in
# stage D step 2: they describe what the generator emits, not which of it a
# model consumes -- that distinction is the whole point of the split.
# ---------------------------------------------------------------------------

# Standalone answer-region metric prefixes (column = f"{metric}_{region}").
ANSWER_RT_TFD_METRICS = (
    "RT_pure",
    "RT_normalized",
    "TFD_pure",
    "TFD_normalized",
    "TimeSinceOffset_pure",
    "TimeSinceOffset_normalized",
)

# Paragraph-screen spans and the dwell-proportion columns they produce
# (feature_groups.PARAGRAPH_BASED). Built per span by features/paragraph/spans.py
# and cached in PARAGRAPH_SPAN_FEATURES_PATH; the trial frame gets them as-is.
PARAGRAPH_SPANS = ("critical", "distractor", "outside")

PARAGRAPH_PROPORTION_COLS = tuple(
    f"{Con.AREA_DWELL_PROPORTION}__{span}" for span in PARAGRAPH_SPANS
)

# The per-span RT / TFD / TimeSinceOffset columns. These used to reach the trial
# frame through RT_and_TFD.csv, because the ANSWER pipeline built them; since
# T6.1 (2026-09-23) the paragraph pipeline owns them and they arrive through the
# paragraph join instead. Same column names, same values, different provenance --
# and now an explicit join rather than a silent by-product of preparing the
# answer screen.
PARAGRAPH_RT_TFD_COLS = tuple(
    f"{metric}_{span}"
    for metric in ANSWER_RT_TFD_METRICS
    for span in RT_PARAGRAPH_REGIONS
)

PARAGRAPH_MODEL_COLS = PARAGRAPH_PROPORTION_COLS + PARAGRAPH_RT_TFD_COLS


def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:

    df_cols = set(df.columns)
    present_cols = [c for c in feature_cols if c in df_cols]
    return df[present_cols].copy()


def build_area_metric_pivot(
    df: pd.DataFrame,
    area_col: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse already-aggregated area-level metrics into one row per group,
    pivoting areas into feature columns.

    Input: word-level df with columns:
        group_cols + [area_col] + metric_cols
    Output: one row per group_cols, columns:
        <metric>__<area_label>
    """
    cols_needed = list(TRIAL_ID_COLS) + [area_col] + list(metric_cols)
    metrics_df = (
        df[cols_needed]
        .dropna(subset=[area_col])
        .groupby(list(TRIAL_ID_COLS) + [area_col], as_index=False)
        .agg({m: "first" for m in metric_cols})
    )

    metrics_pivot = metrics_df.pivot_table(
        index=list(TRIAL_ID_COLS),
        columns=area_col,
        values=metric_cols,
        aggfunc="first",
    )

    metrics_pivot.columns = [
        f"{metric}__{area_label}"
        for metric, area_label in metrics_pivot.columns
    ]
    metrics_pivot = metrics_pivot.reset_index()
    return metrics_pivot


def add_answer_correct_wrong_contrast_columns(
    df_pivot: pd.DataFrame,
    metric_cols: Sequence[str],
    sep: str = "__",
    correct_label: str = "answer_A",
    wrong_labels: Sequence[str] = ("answer_B", "answer_C", "answer_D"),
    out_correct_suffix: str = Con.CORRECT_SUFFIX,
    out_wrong_mean_suffix: str = Con.WRONG_MEAN_SUFFIX,
    out_contrast_suffix: str = Con.CONTRAST_SUFFIX,
    out_distance_furthest_suffix: str = Con.DISTANCE_FURTHEST_SUFFIX,
    out_distance_closest_suffix: str = Con.DISTANCE_CLOSEST_SUFFIX,
) -> pd.DataFrame:
    """
    Given a pivoted trial-level dataframe that already contains columns like:
        <metric>__answer_A, <metric>__answer_B, <metric>__answer_C, <metric>__answer_D

    add:
        <metric>__correct
        <metric>__wrong_mean
        <metric>__contrast
        <metric>__distance_furthest
        <metric>__distance_closest

    where:
        correct             = value of the correct answer
        wrong_mean          = mean value across wrong answers
        contrast            = correct - wrong_mean
        distance_furthest   = max absolute distance between correct and any wrong answer
        distance_closest    = min absolute distance between correct and any wrong answer

    Does not drop any columns.
    Assumes columns exist.
    """
    base = df_pivot.copy()
    new_cols = {}

    for metric in metric_cols:
        a = f"{metric}{sep}{correct_label}"
        bs = [f"{metric}{sep}{lbl}" for lbl in wrong_labels]

        out_correct = f"{metric}{sep}{out_correct_suffix}"
        out_wrong_mean = f"{metric}{sep}{out_wrong_mean_suffix}"
        out_contrast = f"{metric}{sep}{out_contrast_suffix}"
        out_distance_furthest = f"{metric}{sep}{out_distance_furthest_suffix}"
        out_distance_closest = f"{metric}{sep}{out_distance_closest_suffix}"

        correct_vals = pd.to_numeric(base[a], errors="coerce")
        wrong_vals = base[bs].apply(pd.to_numeric, errors="coerce")

        wrong_mean = wrong_vals.mean(axis=1)
        contrast = correct_vals - wrong_mean
        abs_diffs = wrong_vals.sub(correct_vals, axis=0).abs()

        new_cols[out_correct] = correct_vals
        new_cols[out_wrong_mean] = wrong_mean
        new_cols[out_contrast] = contrast
        new_cols[out_distance_furthest] = abs_diffs.max(axis=1)
        new_cols[out_distance_closest] = abs_diffs.min(axis=1)

    derived_df = pd.DataFrame(new_cols, index=base.index)
    out = pd.concat([base, derived_df], axis=1)

    return out


def build_trial_level_constant_numeric_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse numeric columns that are constant within a trial-participant pair
    into one row per trial.

    Returns:
        one row per trial with columns:
            TRIAL_ID_COLS + feature_cols
    """
    feature_cols = list(feature_cols)
    cols_needed = list(TRIAL_ID_COLS) + feature_cols

    out = (
        df[cols_needed]
        .groupby(list(TRIAL_ID_COLS), as_index=False)
        .agg({c: "first" for c in feature_cols})
    )

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_trial_level_categorical_feature(
    df: pd.DataFrame,
    feature_col: str,
    prefix: Optional[str] = None,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """
    Build one-hot encoded trial-level features from a categorical column
    that should be constant within each trial.

    Returns:
        one row per trial with columns:
            group_cols + dummy columns
    """
    prefix = prefix or feature_col

    cols_needed = list(TRIAL_ID_COLS) + [feature_col]
    d = df[cols_needed].copy()

    trial_cat = (
        d.groupby(list(TRIAL_ID_COLS), as_index=False)[feature_col]
        .first()
    )

    dummies = pd.get_dummies(
        trial_cat[feature_col],
        prefix=prefix,
        drop_first=drop_first,
        dummy_na=dummy_na,
    ).astype(int)

    out = pd.concat([trial_cat[list(TRIAL_ID_COLS)], dummies], axis=1)
    return out


def _deduplicate_keep_cols(
    keep_cols: Optional[Sequence[str]],
) -> List[str]:
    keep_cols = list(keep_cols) if keep_cols is not None else []
    return [c for c in keep_cols if c not in TRIAL_ID_COLS]


def _build_trial_core(
    df: pd.DataFrame,
    target_col: str = Con.IS_CORRECT_COLUMN,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a one-row-per-trial core dataframe with IDs, target, and optional keep_cols.
    """
    keep_cols = _deduplicate_keep_cols(keep_cols)
    cols = list(TRIAL_ID_COLS) + keep_cols + [target_col]

    out = (
        df[cols]
        .drop_duplicates()
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )

    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype(int)
    return out


def build_trial_level_area_features(
    df: pd.DataFrame,
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS_MODELING,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    add_correct_wrong_contrasts: bool = True,
) -> pd.DataFrame:
    """
    Build a one-row-per-trial dataframe containing:
      - target_col
      - keep_cols
      - area pivot columns: <metric>__<area>
      - optionally contrast columns:
            <metric>__correct
            <metric>__wrong_mean
            <metric>__contrast
            <metric>__distance_furthest
            <metric>__distance_closest
    """
    trial_core = _build_trial_core(
        df=df,
        target_col=target_col,
        keep_cols=keep_cols,
    )

    metrics_pivot = build_area_metric_pivot(
        df=df,
        area_col=area_col,
        metric_cols=metric_cols,
    )


    if add_correct_wrong_contrasts:
        metrics_pivot = add_answer_correct_wrong_contrast_columns(
            df_pivot=metrics_pivot,
            metric_cols=metric_cols,
            sep="__",
            correct_label="answer_A",
            wrong_labels=("answer_B", "answer_C", "answer_D"),
        )

    out = trial_core.merge(metrics_pivot, on=list(TRIAL_ID_COLS), how="left")
    return out


def build_trial_level_derived_features(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    dwell_col: str = Con.MEAN_DWELL_TIME,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    keep_cols = _deduplicate_keep_cols(keep_cols)

    required = list(TRIAL_ID_COLS) + keep_cols + [target_col, seq_col, dwell_col]
    d = df[required].copy()

    d[target_col] = pd.to_numeric(d[target_col], errors="coerce").astype(int)

    d["_seq"] = d[seq_col].apply(ast.literal_eval)
    d["_seq_len"] = d["_seq"].apply(lambda s: len(s) if isinstance(s, (list, tuple)) else 0)
    d["_has_xyx"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyx(s)) if s is not None else False)
    d["_has_xyxy"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyxy(s)) if s is not None else False)
    d["_longest_alt_answer_run"] = d["_seq"].apply(
        lambda s: longest_alternating_answer_run(s) if s is not None else 0
    )
    d["_trial_mean_dwell"] = compute_trial_mean_dwell_per_word(d, dwell_col=dwell_col)

    agg_dict = {
        target_col: "first",
        "_seq_len": "first",
        "_has_xyx": "first",
        "_has_xyxy": "first",
        "_longest_alt_answer_run": "first",
        "_trial_mean_dwell": "first",
    }
    for c in keep_cols:
        agg_dict[c] = "first"

    out = (
        d.groupby(list(TRIAL_ID_COLS), as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "_seq_len": "seq_len",
            "_has_xyx": "has_xyx",
            "_has_xyxy": "has_xyxy",
            "_longest_alt_answer_run": "longest_alt_answer_run",
            "_trial_mean_dwell": "trial_mean_dwell",
        })
    )

    out["has_xyx"] = out["has_xyx"].astype(int)
    out["has_xyxy"] = out["has_xyxy"].astype(int)
    out["longest_alt_answer_run"] = out["longest_alt_answer_run"].astype(int)

    return out


def build_trial_level_rt_tfd_features(
    df: pd.DataFrame,
    target_col: str = Con.IS_CORRECT_COLUMN,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Trial-level features derived from the RT_and_TFD merge:

    - One feature per region per metric — column name f"{metric}_{region}"
      for region in ANSWER_REGIONS + PARAGRAPH_REGIONS and metric in
      ANSWER_RT_TFD_METRICS (RT/TFD/TimeSinceOffset, pure & normalized).
      For every region, answer and paragraph alike, RT_* is run-based (time
      accrues only while the region is being looked at) and TimeSinceOffset_* is
      the first-to-last-fixation span, which counts excursions away and back.
      Paragraph regions only gained their TimeSinceOffset_* counterpart when the
      run-based paragraph RT was added; older feature CSVs lack those columns,
      and there RT_* holds the span values.
    - For each metric, correct/wrong contrast columns derived from the answer
      regions (answer_A is correct) — column names f"{metric}_correct",
      f"{metric}_wrong_mean", f"{metric}_contrast", f"{metric}_distance_furthest"
      and f"{metric}_distance_closest". The "question" and paragraph regions are
      left untouched.
    """
    keep_cols = _deduplicate_keep_cols(keep_cols)

    # ANSWER regions only. The paragraph regions used to be taken from here too,
    # because the answer pipeline built them into RT_and_TFD.csv; since T6.1 the
    # paragraph table owns them and they arrive through the paragraph join. Two
    # sources for one column name is not a tie to break -- pandas silently
    # suffixes them `_x`/`_y` and the feature simply stops existing under the name
    # every feature set refers to.
    #
    # NOTE: an `all_participants.csv` built before T6.1 still carries the
    # paragraph RT columns, merged in by the old Stage 1. They are ignored here
    # rather than used, and disappear on the next Stage 1 rebuild.
    metric_cols = [
        f"{m}_{r}"
        for m in ANSWER_RT_TFD_METRICS
        for r in tuple(RT_ANSWER_REGIONS)
        if f"{m}_{r}" in df.columns
    ]

    cols = list(TRIAL_ID_COLS) + keep_cols + [target_col] + metric_cols
    cols = list(dict.fromkeys(cols))

    trial = (
        df[cols]
        .drop_duplicates(subset=list(TRIAL_ID_COLS))
        .dropna(subset=[target_col])
        .reset_index(drop=True)
        .copy()
    )
    trial[target_col] = pd.to_numeric(trial[target_col], errors="coerce").astype(int)

    for c in metric_cols:
        trial[c] = pd.to_numeric(trial[c], errors="coerce")

    # Correct/wrong contrast columns over the answer regions (answer_A is correct).
    # Column names are f"{metric}_{region}", so the separator here is a single "_".
    # Only metrics whose four answer-region columns are all present can be
    # contrasted (the builder assumes the columns exist).
    contrast_answer_regions = ("answer_A", "answer_B", "answer_C", "answer_D")
    contrast_metrics = [
        m
        for m in ANSWER_RT_TFD_METRICS
        if all(f"{m}_{r}" in trial.columns for r in contrast_answer_regions)
    ]
    if contrast_metrics:
        trial = add_answer_correct_wrong_contrast_columns(
            df_pivot=trial,
            metric_cols=contrast_metrics,
            sep="_",
            correct_label="answer_A",
            wrong_labels=("answer_B", "answer_C", "answer_D"),
        )

    return trial


def build_trial_level_paragraph_features(
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    feature_cols: Sequence[str] = PARAGRAPH_MODEL_COLS,
) -> pd.DataFrame:
    """
    One row per trial with the paragraph-screen features the model uses:

        area_dwell_proportion__critical / __distractor / __outside
        RT_* / TFD_* / TimeSinceOffset_* per span

    The dwell proportions are the same quantity as the per-answer
    `area_dwell_proportion__*` columns, grouped by paragraph span rather than
    answer area. Read from the cache written by
    `features.paragraph.spans.save_paragraph_features`; run that first if the file
    is missing.

    A column named in `feature_cols` but absent from the cache is skipped rather
    than raising, because an older cache predates the RT/TFD half. That is worth
    knowing: a stale cache silently yields a trial frame with no paragraph RT
    columns, so rebuild it after changing the paragraph pipeline.
    """
    if paragraph_features is None:
        path = Path(paragraph_features_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Paragraph-span features not found at {path}. Build them with "
                "src.features.paragraph.spans.save_paragraph_features()."
            )
        paragraph_features = pd.read_csv(path)

    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in paragraph_features.columns]
    if missing:
        raise KeyError(
            f"Paragraph-span features are missing {missing}; rebuild them with "
            "save_paragraph_features()."
        )

    out = (
        paragraph_features[list(TRIAL_ID_COLS) + feature_cols]
        .drop_duplicates(subset=list(TRIAL_ID_COLS))
        .reset_index(drop=True)
    )
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# The last-visited one-hot column names, moved here from
# `answer_correctness/feature_groups.py` in stage D step 4 (2026-10-07). They are
# the output vocabulary of `build_trial_level_last_visited_features` just below,
# so they belong with their producer: "what the generator emits" rather than
# "what a model chooses" (proposal section 3.3).
#
# This is also what removes the upward import that section 3.3 exists to fix.
# `common/feature_specs.py` needed exactly one name from `feature_groups`
# -- LAST_ALL -- and reached up into a modelling package to get it. Both it and
# `feature_groups` now read these from here, which is downstream for both.
# ---------------------------------------------------------------------------

LAST_CONFIRM_LONG: List[str] = [
    "last_before_confirm_answer_A",
    "last_before_confirm_answer_B",
    "last_before_confirm_answer_C",
    # "last_before_confirm_answer_D",
    "last_before_confirm_question",
]

LAST_CONFIRM_COMPACT: List[str] = [
    "last_before_confirm_correct",
    # "last_before_confirm_wrong",
    "last_before_confirm_question",
]

LAST_SELECT_LONG: List[str] = [
    "last_before_select_answer_A",
    "last_before_select_answer_B",
    "last_before_select_answer_C",
    # "last_before_select_answer_D",
    "last_before_select_question",
]

LAST_SELECT_COMPACT: List[str] = [
    "last_before_select_correct",
    # "last_before_select_wrong",
    "last_before_select_question",
]

# Backwards-compatible aliases: the bare names default to the original
# (long / one-hot) form.
LAST_CONFIRM: List[str] = LAST_CONFIRM_LONG

LAST_SELECT: List[str] = LAST_SELECT_LONG

LAST_ALL: List[str] = LAST_CONFIRM_LONG + LAST_SELECT_LONG

LAST_ALL_COMPACT: List[str] = LAST_CONFIRM_COMPACT + LAST_SELECT_COMPACT


def build_trial_level_last_visited_features(
    df: pd.DataFrame,
    feature_col: str = Con.LAST_VISITED_LABEL,
    prefix: str = "last_visited",
) -> pd.DataFrame:
    """
    One-hot encode the last visited label at trial level, and add semantic
    indicator columns derived from those dummies (answer_A is the correct
    answer, answer_B/C/D are wrong):

      * f"{prefix}_correct"  -- last label is the correct answer (answer_A)
      * f"{prefix}_wrong"    -- last label is a wrong answer (answer_B/C/D)
      * f"{prefix}_question" -- last label is the question
    """
    out = build_trial_level_categorical_feature(
        df=df,
        feature_col=feature_col,
        prefix=prefix,
        drop_first=False,
        dummy_na=True,
    )

    dummy_cols = [c for c in out.columns if c not in TRIAL_ID_COLS]
    if dummy_cols:
        out[dummy_cols] = out[dummy_cols].fillna(0).astype(int)

    def _dummy(label: str) -> pd.Series:
        col = f"{prefix}_{label}"
        if col in out.columns:
            return out[col]
        return pd.Series(0, index=out.index, dtype=int)

    out[f"{prefix}_correct"] = _dummy("answer_A")
    out[f"{prefix}_wrong"] = (
        _dummy("answer_B") | _dummy("answer_C") | _dummy("answer_D")
    ).astype(int)
    out[f"{prefix}_question"] = _dummy("question")

    return out


def build_trial_level_model_df(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = [Con.TEXT_ID_WITH_Q_COLUMN],
    target_col: str = Con.IS_CORRECT_COLUMN,
    dataset: "str | object" = "l1",
    numeric_feature_cols: Sequence[str] = (Con.NUM_OF_SELECTS,),
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS_MODELING,
    area_col: str = Con.AREA_LABEL_COLUMN,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    dwell_col: str = Con.IA_DWELL_TIME,
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    optional_keep_cols: Sequence[str] = (Con.REGIME_COLUMN, Con.SESSION_ID),
    pattern_scope_df: Optional[pd.DataFrame] = None,
    pattern_scope_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build the final one-row-per-trial modeling dataframe.

    This is the main function you can use in the pipeline.

    The paragraph-span features are the one block not computed from `df`: they
    come from the cached paragraph-screen table, left-joined on
    (participant_id, TRIAL_INDEX). A `df` from a different experiment therefore
    needs its own cache passed via `paragraph_features`/`paragraph_features_path`,
    or those columns come back NaN.

    `keep_cols` is required to exist; `optional_keep_cols` is carried when
    present and skipped otherwise. The split is deliberate: `regime` and
    `session_id` are Study 2 identity/condition columns that L1 does not have,
    so demanding them would make one pipeline refuse one of its two datasets --
    but dropping them silently is what left the trial-level table unable to be
    grouped by regime at all, which is why they are named here rather than
    left out.

    `pattern_scope_df` / `pattern_scope_by` set the scope of the participant-level
    pattern-breaking features -- which trials estimate the dominant strategy, and
    how they are partitioned. Defaults (None/None) estimate over `df` itself,
    pooled per participant. See `features.strategies` and `todo.md` T3.21.
    """
    # Whether this dataset HAS a paragraph screen is a property of the dataset,
    # not a choice the caller makes per run -- so it is read off the record
    # rather than passed as a flag (stage D step 5). `dataset` accepts a key or a
    # Dataset; `config/screens.py` and `config/datasets.py` hold the vocabulary.
    ds = dataset if hasattr(dataset, "has_paragraph") else _dataset(dataset)

    present_optional = [
        c for c in optional_keep_cols
        if c in df.columns and c not in (keep_cols or [])
    ]

    trial_core = _build_trial_core(
        df=df,
        target_col=target_col,
        keep_cols=list(keep_cols or []) + present_optional,
    )

    out = trial_core.copy()

    area_df = build_trial_level_area_features(
        df=df,
        area_col=area_col,
        metric_cols=metric_cols,
        keep_cols=None,
        target_col=target_col,
        add_correct_wrong_contrasts=True,
    )
    drop_cols = [target_col]
    area_df = area_df.drop(columns=[c for c in drop_cols if c in area_df.columns])
    assert_full_coverage(out, area_df, TRIAL_ID_COLS, "area features")
    out = out.merge(area_df, on=list(TRIAL_ID_COLS), how="left")

    derived_df = build_trial_level_derived_features(
        df=df,
        seq_col=seq_col,
        dwell_col=dwell_col,
        keep_cols=None,
        target_col=target_col,
    )
    drop_cols = [target_col]
    derived_df = derived_df.drop(
        columns=[c for c in drop_cols if c in derived_df.columns]
    )
    assert_full_coverage(out, derived_df, TRIAL_ID_COLS, "derived features")
    out = out.merge(derived_df, on=list(TRIAL_ID_COLS), how="left")

    pattern_df = build_trial_level_pattern_features(
        df,
        kind="location",
        window_len=4,
        scope_df=pattern_scope_df,
        scope_by=pattern_scope_by,
    )
    assert_full_coverage(out, pattern_df, TRIAL_ID_COLS, "pattern-breaking features")
    out = out.merge(pattern_df, on=list(TRIAL_ID_COLS), how="left")

    if ds.has_paragraph:
        paragraph_df = build_trial_level_paragraph_features(
            paragraph_features=paragraph_features,
            paragraph_features_path=paragraph_features_path,
        )
        # A name carried by both frames would be silently suffixed `_x`/`_y`, and
        # the feature every feature set refers to would simply stop existing --
        # no error, just a column gone missing. Caught here rather than
        # discovered when a model reports a KeyError three steps later.
        #
        # This block now runs only when the dataset record says the dataset HAS a
        # paragraph screen, so an empty join is no longer a plausible
        # configuration mistake -- it is a broken cache or a key mismatch, and it
        # raises. Before stage D step 5 it was a printed warning, because the
        # caller could reach here with the flag left on by accident.
        overlap = len(
            set(map(tuple, out[list(TRIAL_ID_COLS)].values))
            & set(map(tuple, paragraph_df[list(TRIAL_ID_COLS)].values))
        )
        if overlap == 0:
            raise ValueError(
                f"{ds.key} declares has_paragraph=True, but the paragraph cache at "
                f"{paragraph_features_path} shares no trial with this frame, so all "
                f"{len(paragraph_df.columns) - 2} paragraph columns would be NaN. "
                f"Either the cache is stale or the trial keys disagree."
            )

        clash = (set(out.columns) & set(paragraph_df.columns)) - set(TRIAL_ID_COLS)
        if clash:
            raise ValueError(
                f"Paragraph features collide with columns already on the trial "
                f"frame: {sorted(clash)}. Each of these has two owners; since "
                "T6.1 the paragraph table is the only one. The usual cause is an "
                "`all_participants.csv` built before T6.1, whose Stage 1 baked "
                "the paragraph RT columns into the answer table."
            )
        # No assert_full_coverage here, unlike every other block (T3.17): this
        # is the one block not built from `df`, so partial coverage is a real
        # state rather than a pipeline fault -- a dataset with no paragraph
        # screen legitimately matches nothing, which the warning above reports.
        # The row count must still not move, which duplicate keys in the cache
        # would do silently.
        before = len(out)
        out = out.merge(paragraph_df, on=list(TRIAL_ID_COLS), how="left")
        assert len(out) == before, (
            f"paragraph join changed the row count: {before} -> {len(out)}"
        )

    last_before_confirm_df = build_trial_level_last_visited_features(
        df=df,
        feature_col=Con.LAST_LBL_BEFORE_CONFIRM,
        prefix="last_before_confirm",
    )
    assert_full_coverage(
        out, last_before_confirm_df, TRIAL_ID_COLS, "last-before-confirm features"
    )
    out = out.merge(last_before_confirm_df, on=list(TRIAL_ID_COLS), how="left")

    last_before_select_df = build_trial_level_last_visited_features(
        df=df,
        feature_col=Con.LAST_LBL_BEFORE_SELECT,
        prefix="last_before_select",
    )
    assert_full_coverage(
        out, last_before_select_df, TRIAL_ID_COLS, "last-before-select features"
    )
    out = out.merge(last_before_select_df, on=list(TRIAL_ID_COLS), how="left")

    rt_tfd_df = build_trial_level_rt_tfd_features(
        df=df,
        target_col=target_col,
    )
    rt_tfd_df = rt_tfd_df.drop(
        columns=[c for c in [target_col] if c in rt_tfd_df.columns]
    )
    assert_full_coverage(out, rt_tfd_df, TRIAL_ID_COLS, "RT/TFD features")
    out = out.merge(rt_tfd_df, on=list(TRIAL_ID_COLS), how="left")

    if numeric_feature_cols:
        numeric_df = build_trial_level_constant_numeric_features(
            df=df,
            feature_cols=numeric_feature_cols,
        )
        assert_full_coverage(out, numeric_df, TRIAL_ID_COLS, "constant numeric features")
        out = out.merge(numeric_df, on=list(TRIAL_ID_COLS), how="left")

    # Trial-level total answering RT, taken from the raw CONFIRM_FINAL_ANSWER_RT
    # column and exposed under the friendlier name TOTAL_ANSWERING_RT.
    if Con.CONFIRM_FINAL_ANSWER_RT in df.columns:
        feature_cols = [Con.CONFIRM_FINAL_ANSWER_RT]

        if Con.TOTAL_ANSWERING_RT_NORMALIZED in df.columns:
            feature_cols.append(Con.TOTAL_ANSWERING_RT_NORMALIZED)

        total_rt_df = build_trial_level_constant_numeric_features(
            df=df,
            feature_cols=feature_cols,
        ).rename(columns={
            Con.CONFIRM_FINAL_ANSWER_RT: Con.TOTAL_ANSWERING_RT,
        })

        assert_full_coverage(out, total_rt_df, TRIAL_ID_COLS, "total answering RT")
        out = out.merge(total_rt_df, on=list(TRIAL_ID_COLS), how="left")

    return out

# ===========================================================================
# The cache: writing the model-ready table, and reading it back
#
# Moved here from `answer_correctness/model_data.py` in stage D step 7 (Diana,
# 2026-10-07: "putting them with the builders makes sense"). `save_model_ready`
# is `build_trial_level_model_df` plus `to_csv`, so it belongs beside the builder
# rather than in a modelling package -- and `analyses/text_qa_relationship` calls it to
# rebuild a missing cache, which from `analyses/` is a downward import only if
# the pair lives here.
#
# Renamed from `save_all_features` / `load_all_features`. "All features" named
# neither the grain nor the contents, and collided with a different function of
# the same name in `RT_correlations`. These match the artifact and the registry
# property that locates it (`Dataset.model_ready`).
# ===========================================================================

def save_model_ready(
    df: pd.DataFrame,
    output_path: Path = READY_ALL_FEATURES_PATH,
    keep_cols: Optional[Sequence[str]] = [Con.TEXT_ID_WITH_Q_COLUMN],
    target_col: str = Con.IS_CORRECT_COLUMN,
    verbose: bool = True,
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    dataset: "str | object" = "l1",
    pattern_scope_df: Optional[pd.DataFrame] = None,
    pattern_scope_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build the full trial-level feature DataFrame (every include_* flag turned on)
    and save it to `output_path` as CSV. The CSV can later be read back with
    `load_model_ready`.

    The cache this writes is the whole dataset, so its participant-level
    pattern features are estimated over every trial the participant has --
    the widest scope there is. That is the right default for a cache meant to
    be sliced descriptively, and the wrong one to hand to cross-validation as
    `trial_df`: see `cross_validation.evaluate_one_fold_on_regimes` and
    `todo.md` T3.21. `pattern_scope_by` narrows it, e.g. `["regime"]` for a
    per-regime dominance score on KnowQA.

    `paragraph_features` / `paragraph_features_path` point at the paragraph-span
    cache to join in -- see `build_trial_level_model_df`.

    **`dataset` says which dataset this frame is**, and the paragraph block runs
    only if that dataset's record has `has_paragraph=True` (stage D step 5). It
    replaced an `include_paragraph_features` flag that every caller had to
    remember to turn off: the paragraph cache path is global, so leaving it on
    for a dataset with no paragraph screen joined it against L1's table, and
    every column came back NaN -- harmlessly, but only because the participant
    ids happen not to collide. Being a property of the dataset, it is now read
    rather than passed.
    """
    trial_df = build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        dataset=dataset,
        paragraph_features=paragraph_features,
        paragraph_features_path=paragraph_features_path,
        pattern_scope_df=pattern_scope_df,
        pattern_scope_by=pattern_scope_by,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output_path, index=False)
    if verbose:
        print(
            f"Saved {len(trial_df)} trials x {len(trial_df.columns)} cols "
            f"to {output_path}"
        )
    return trial_df


def load_model_ready(path: Path = READY_ALL_FEATURES_PATH) -> pd.DataFrame:
    """
    Load the cached full feature DataFrame produced by `save_model_ready`.

    `participant_id` is pinned to str: KnowQA person ids are all digits
    (`4000`), so type inference would make them int64 here while every consumer
    that joins on them (the regime and confidence merges, the button-click
    join) casts its own side to str -- and the merge would then fail on the
    dtype mismatch. L1's ids are already strings, so this is a no-op there.
    """
    return pd.read_csv(Path(path), dtype={Con.PARTICIPANT_ID: str})
