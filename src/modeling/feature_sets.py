"""Named feature-column sets: what a model is *told to use*.

Merged in stage D step 7 from `answer_correctness/feature_groups.py` and
`common/feature_specs.py`, which were the two halves of one job (proposal §3.3,
§4). This is the **chosen** side of the split: hand-written decisions about which
columns a model sees, each with its reason. The **generated** side -- what the
pipeline produces, and under what name -- lives in `features/`; the last-visited
vocabulary below is imported from there rather than restated.

The two merged modules expressed "a feature set" in two different ways, and both
survive here because they are not interchangeable:

* **Static lists** (`SELECT_1_COLS`, `GENERAL_FEATURES`, `RT_COLS`, ...) name
  columns unconditionally. A model asked for one gets exactly those columns, and
  a missing column is an error at fit time -- which is the point.
* **Frame-aware predicates** (`get_*_feature_cols(df)`) return the columns that
  are *present* in a given frame. Used where a caller does not know which blocks
  a dataset carries -- KnowQA has no paragraph columns, for instance.

> **They deliberately disagree on the RT variants, and this is not a bug to fix.**
> `RT_TFD_VARIANTS` is `["normalized"]`, so the static `RT_COLS` / `TFD_COLS` /
> `TIME_SINCE_OFFSET_COLS` name only the normalized columns. The predicate
> `get_rt_tfd_feature_cols` walks `RT_TFD_ANSWER_METRICS`, which lists **both**
> `pure` and `normalized`, and returns whichever are in the frame. So "the RT
> features" means a different set depending on which door you came through.
> Unifying them would change which columns existing models see. If it is ever
> unified, that is a modelling decision with a rerun attached, not a tidy-up.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.config import columns as Con

# The last-visited vocabulary belongs to the module that produces those columns
# (stage D step 4). Re-exported here because seven call sites name it through
# this module, and because it is genuinely part of the chosen vocabulary too.
from src.features.build import (
    LAST_ALL,
    LAST_ALL_COMPACT,
    LAST_CONFIRM,
    LAST_CONFIRM_COMPACT,
    LAST_CONFIRM_LONG,
    LAST_SELECT,
    LAST_SELECT_COMPACT,
    LAST_SELECT_LONG,
)

# ---------------------------------------------------------------------------
# Base metric columns
# ---------------------------------------------------------------------------

METRIC_COLUMNS: List[str] = [
    Con.MEAN_DWELL_TIME,
    Con.MEAN_FIXATIONS_COUNT,
    Con.MEAN_FIRST_FIXATION_DURATION,
    Con.SKIP_RATE,
    Con.AREA_DWELL_PROPORTION,
    Con.MEAN_AVG_FIX_PUPIL_SIZE_Z,
    Con.MEAN_MAX_FIX_PUPIL_SIZE_Z,
    Con.MEAN_MIN_FIX_PUPIL_SIZE_Z,
    Con.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z,
    Con.NUM_LABEL_VISITS,
]


# ---------------------------------------------------------------------------
# Derived trial-level features
# ---------------------------------------------------------------------------

DERIVED_COLS: List[str] = [
    "seq_len",
    "has_xyx",
    "has_xyxy",
    "longest_alt_answer_run",
    "trial_mean_dwell",
]


# ---------------------------------------------------------------------------
# Per-trial pattern-breaking features
#   breaks_pattern_{with,no}_q     -- trial deviates from participant's dominant
#                                     starting strategy (question kept / dropped)
#   dominance_score_{with,no}_q    -- participant's dominance score, per trial
# ---------------------------------------------------------------------------

PATTERN_COLS: List[str] = [
    Con.BREAKS_PATTERN_WITH_Q,
    Con.BREAKS_PATTERN_NO_Q,
    Con.DOMINANCE_SCORE_WITH_Q,
    Con.DOMINANCE_SCORE_NO_Q,
]

# Opt-in interaction terms (breaks_pattern * dominance_score). Kept separate
# from PATTERN_COLS / the aggregate sets so they are only added on request, e.g.
#   feature_cols = FS.GENERAL_FEATURES + FS.PATTERN_INTERACTION_COLS
PATTERN_INTERACTION_COLS: List[str] = [
    Con.BREAKS_X_DOMINANCE_WITH_Q,
    Con.BREAKS_X_DOMINANCE_NO_Q,
]

# Convenience: base pattern features together with their interaction terms.
PATTERN_COLS_WITH_INTERACTIONS: List[str] = PATTERN_COLS + PATTERN_INTERACTION_COLS

# Graded "breaks pattern": Levenshtein distance between the trial's starting
# strategy and the participant's dominant one. Collinear with the binary
# breaks_pattern_* cols (distance == 0 iff the trial does not break the
# pattern), so kept as a separate opt-in group -- typically swapped in *instead*
# of breaks_pattern rather than added alongside it.
PATTERN_DISTANCE_COLS: List[str] = [
    Con.STRATEGY_DISTANCE_WITH_Q,
    Con.STRATEGY_DISTANCE_NO_Q,
]


# ---------------------------------------------------------------------------
# Area-derived metric columns
#   <metric>__correct, <metric>__wrong_mean, <metric>__contrast,
#   <metric>__distance_furthest, <metric>__distance_closest
# ---------------------------------------------------------------------------

AREA_COLS: List[str] = (
    [f"{m}__{Con.CORRECT_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}__{Con.WRONG_MEAN_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}__{Con.CONTRAST_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}__{Con.DISTANCE_FURTHEST_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}__{Con.DISTANCE_CLOSEST_SUFFIX}" for m in METRIC_COLUMNS]
)


# ---------------------------------------------------------------------------
# Per-label metric columns (raw pivot: one column per metric x area label)
#   PER_QUESTION_COLS:  <metric>__question
#   PER_ANSWER_COLS:    <metric>__answer_A, ..._B, ..._C, ..._D
#   PER_LABEL_COLS:     question + answers (everything in Con.LABEL_CHOICES)
# ---------------------------------------------------------------------------

PER_QUESTION_COLS: List[str] = [f"{m}__question" for m in METRIC_COLUMNS]

# PER_ANSWER_COLS: List[str] = [
#     f"{m}__{Con.ANSWER_PREFIX}{letter}"
#     for m in METRIC_COLUMNS
#     for letter in Con.ANSWER_LABELS
# ]

# PER_LABEL_COLS: List[str] = PER_QUESTION_COLS + PER_ANSWER_COLS


# ---------------------------------------------------------------------------
# Paragraph-based features
#   Dwell proportions measured on the *paragraph* screen, one per span: the
#   share of the trial's paragraph dwell time spent on the critical span, on
#   the distractor span, and on the rest of the text. Same quantity as the
#   per-answer area_dwell_proportion__* columns, computed over
#   `auxiliary_span_type` instead of `area_label`; merged into the trial frame
#   by build_trial_level_model_df from the cached paragraph-span features.
#
#   Opt-in, like PATTERN_INTERACTION_COLS: kept out of ALL_FEATURES /
#   GENERAL_FEATURES so existing runs stay comparable, e.g.
#       feature_cols = FS.GENERAL_FEATURES + FS.PARAGRAPH_BASED
#   Note the three shares are compositional (they sum to 1), so alongside an
#   intercept they carry only two degrees of freedom.
# ---------------------------------------------------------------------------

PARAGRAPH_SPANS: List[str] = ["critical", "distractor", "outside"]

PARAGRAPH_BASED: List[str] = [
    f"{Con.AREA_DWELL_PROPORTION}__{span}" for span in PARAGRAPH_SPANS
]


# ---------------------------------------------------------------------------
# Last-visited / last-before-action one-hot groups
#
# Each "last before action" feature comes in two flavors:
#   * LONG    -- the original one-hot encoding, one column per label.
#   * COMPACT -- the collapsed correct / wrong / question indicators.
#
# The lists themselves are imported at the top of this module from
# `features/build.py`, which is what produces those columns.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RT / TFD / TimeSinceOffset feature groups
# (column names produced by build_trial_level_rt_tfd_features)
# ---------------------------------------------------------------------------

RT_TFD_PARAGRAPH_REGIONS: List[str] = ["outside", "distractor", "critical"]
RT_TFD_VARIANTS: List[str] = ["normalized"]  # ["pure", "normalized"]

# Correct/wrong contrast suffixes derived from the answer regions.
RT_TFD_CONTRAST_SUFFIXES: List[str] = [
    Con.CORRECT_SUFFIX,
    Con.WRONG_MEAN_SUFFIX,
    Con.CONTRAST_SUFFIX,
    Con.DISTANCE_FURTHEST_SUFFIX,
    Con.DISTANCE_CLOSEST_SUFFIX,
]

# Regions kept as standalone columns in the aggregate feature sets: the question
# and paragraph regions. The four answer regions (answer_A-D) are represented
# through the correct/wrong contrast suffixes instead, mirroring AREA_COLS (which
# excludes per-answer columns and keeps only the contrast + question variants).
RT_TFD_NON_ANSWER_REGIONS: List[str] = ["question"] + RT_TFD_PARAGRAPH_REGIONS

RT_COLS: List[str] = [
    f"RT_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_NON_ANSWER_REGIONS
] + [f"RT_{v}_{s}" for v in RT_TFD_VARIANTS for s in RT_TFD_CONTRAST_SUFFIXES]

TFD_COLS: List[str] = [
    f"TFD_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_NON_ANSWER_REGIONS
] + [f"TFD_{v}_{s}" for v in RT_TFD_VARIANTS for s in RT_TFD_CONTRAST_SUFFIXES]

# TimeSinceOffset has no paragraph counterpart, so among the non-answer regions
# only "question" applies; it still gets the answer-region contrast columns
# (computed from answer_A-D).
TIME_SINCE_OFFSET_COLS: List[str] = [
    f"TimeSinceOffset_{v}_question" for v in RT_TFD_VARIANTS
] + [
    f"TimeSinceOffset_{v}_{s}"
    for v in RT_TFD_VARIANTS
    for s in RT_TFD_CONTRAST_SUFFIXES
]


RT_TFD_OFFSET_COLS: List[str] = RT_COLS + TFD_COLS + TIME_SINCE_OFFSET_COLS


# ---------------------------------------------------------------------------
# Per-answer RT / TFD / TimeSinceOffset groups (raw answer_A-D columns).
# Available as standalone named groups for targeted experiments, but
# deliberately NOT part of ALL_FEATURES / GENERAL_FEATURES (those use the
# correct/wrong contrast representation of the answers instead).
# ---------------------------------------------------------------------------

RT_TFD_PER_ANSWER_REGIONS: List[str] = ["answer_A", "answer_B", "answer_C", "answer_D"]

RT_PER_ANSWER_COLS: List[str] = [
    f"RT_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_PER_ANSWER_REGIONS
]

TFD_PER_ANSWER_COLS: List[str] = [
    f"TFD_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_PER_ANSWER_REGIONS
]

TIME_SINCE_OFFSET_PER_ANSWER_COLS: List[str] = [
    f"TimeSinceOffset_{v}_{r}"
    for v in RT_TFD_VARIANTS
    for r in RT_TFD_PER_ANSWER_REGIONS
]

RT_TFD_OFFSET_PER_ANSWER_COLS: List[str] = (
    RT_PER_ANSWER_COLS + TFD_PER_ANSWER_COLS + TIME_SINCE_OFFSET_PER_ANSWER_COLS
)


# ---------------------------------------------------------------------------
# Aggregate "all features" sets
# ---------------------------------------------------------------------------

ALL_FEATURES_NO_LAST: List[str] = (
    AREA_COLS
    + PER_QUESTION_COLS
    + DERIVED_COLS
    + PATTERN_COLS
    + [Con.NUM_OF_SELECTS]
    + RT_COLS
    + TFD_COLS
    + TIME_SINCE_OFFSET_COLS
)

ALL_FEATURES: List[str] = (
    ALL_FEATURES_NO_LAST + LAST_CONFIRM_COMPACT + LAST_SELECT_COMPACT
)


# ---------------------------------------------------------------------------
# General features
#   = ALL_FEATURES_NO_LAST minus RT/TFD/TSO base columns and their
#     interaction terms. Used as the "general" base for additive groupings.
# ---------------------------------------------------------------------------

GENERAL_FEATURES: List[str] = (
    AREA_COLS + PER_QUESTION_COLS + DERIVED_COLS + PATTERN_COLS + [Con.NUM_OF_SELECTS]
)


# ---------------------------------------------------------------------------
# Manually curated feature subsets
# ---------------------------------------------------------------------------

SELECT_1_COLS: List[str] = [
    "area_dwell_proportion__correct",
    "area_dwell_proportion__question",
    "skip_rate__correct",
    "has_xyx",
    "ANSWER_PRESS_NUMBER",
    "num_label_visits__correct",
    "num_label_visits__contrast",
    "mean_fixations_count__question",
    "mean_fixations_count__wrong_mean",
    "mean_max_fix_pupil_size_z__correct",
]


# ===========================================================================
# Frame-aware selectors
#
# Was `common/feature_specs.py`. These answer "which of these columns does this
# frame actually have?", which the static lists above deliberately do not.
#
# Four constants this half used to define for itself were character-for-character
# equal to lists above -- `DERIVED_BASE_FEATURES` == `DERIVED_COLS`,
# `PATTERN_FEATURE_COLS` == `PATTERN_COLS`, and its own copies of
# `RT_TFD_CONTRAST_SUFFIXES` and `RT_TFD_PARAGRAPH_REGIONS`. None was referenced
# outside its own module, so the merge drops the duplicates and the selectors
# read the canonical lists. The two names below have no counterpart above,
# because they decompose the RT columns differently -- see the module docstring.
# ===========================================================================

# Standalone answer-region columns produced by build_trial_level_rt_tfd_features.
# NOTE both variants, unlike RT_TFD_VARIANTS above. This is the disagreement the
# module docstring describes; it is load-bearing, not an oversight.
RT_TFD_ANSWER_METRICS = (
    "RT_pure",
    "RT_normalized",
    "TFD_pure",
    "TFD_normalized",
    "TimeSinceOffset_pure",
    "TimeSinceOffset_normalized",
)
RT_TFD_ANSWER_REGIONS = ("question", "answer_A", "answer_B", "answer_C", "answer_D")


def get_area_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return area-based feature columns that are present in df.

    Includes:
      - <metric>__<area>
      - <metric>__correct
      - <metric>__wrong_mean
      - <metric>__contrast
      - <metric>__distance_furthest
      - <metric>__distance_closest
    """
    cols: List[str] = []

    area_labels = list(Con.LABEL_CHOICES)
    derived_suffixes = [
        Con.CORRECT_SUFFIX,
        Con.WRONG_MEAN_SUFFIX,
        Con.CONTRAST_SUFFIX,
        Con.DISTANCE_FURTHEST_SUFFIX,
        Con.DISTANCE_CLOSEST_SUFFIX,
    ]

    for metric in Con.AREA_METRIC_COLUMNS_MODELING:
        for area in area_labels:
            col = f"{metric}__{area}"
            if col in df.columns:
                cols.append(col)

        for suffix in derived_suffixes:
            col = f"{metric}__{suffix}"
            if col in df.columns:
                cols.append(col)

    return cols


def get_derived_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return derived trial-level feature columns that are present in df.
    """
    cols: List[str] = []

    cols.extend([c for c in DERIVED_COLS if c in df.columns])
    cols.extend(sorted(c for c in df.columns if c.startswith("pref_matching__")))

    return cols


def get_pattern_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return per-trial pattern-breaking feature columns that are present in df.
    """
    return [c for c in PATTERN_COLS if c in df.columns]


def get_last_visited_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the last-before-action one-hot feature columns present in df.

    Takes the canonical list from `LAST_ALL` rather than prefix-matching, for
    two reasons (`todo.md` T2.4).

    First, the prefix it used to match -- `last_visited_` -- names nothing.
    These are built with `last_before_confirm` / `last_before_select`
    (`build_trial_level_last_visited_features`), so the function returned `[]`
    on every frame and this block was silently absent from
    `get_full_feature_cols`.

    Second, and the reason the fix is not simply a corrected prefix: the 16
    built columns are **not** a usable feature block. Each family is six
    mutually exclusive one-hots (answer_A-D, question, nan) that sum to 1 --
    perfectly collinear with an intercept -- plus `correct` / `wrong`, which
    are linear functions of the same indicators. `LAST_ALL` is the encoding the
    rest of the project already uses: the long one-hot form with `answer_D`
    held out as the reference level, and `nan` / `correct` / `wrong` excluded.
    Matching on a prefix would have put the collinear block back, and under L2
    that produces unstable coefficients rather than an error.
    """
    return [c for c in LAST_ALL if c in df.columns]


def get_rt_tfd_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Feature columns produced by build_trial_level_rt_tfd_features:
      - per-region features:  f"{metric}_{region}" for metric in
        RT_TFD_ANSWER_METRICS and region in
        RT_TFD_ANSWER_REGIONS + RT_TFD_PARAGRAPH_REGIONS. Paragraph regions
        (outside, distractor, critical) have no TimeSinceOffset counterpart,
        so those columns are simply absent.
      - correct/wrong contrast features derived from the answer regions:
        f"{metric}_{suffix}" for suffix in RT_TFD_CONTRAST_SUFFIXES.
      - the trial-level total answering RT (Con.TOTAL_ANSWERING_RT).
    """
    cols: List[str] = []
    regions = list(RT_TFD_ANSWER_REGIONS) + list(RT_TFD_PARAGRAPH_REGIONS)
    for metric in RT_TFD_ANSWER_METRICS:
        for region in regions:
            col = f"{metric}_{region}"
            if col in df.columns:
                cols.append(col)
        for suffix in RT_TFD_CONTRAST_SUFFIXES:
            col = f"{metric}_{suffix}"
            if col in df.columns:
                cols.append(col)
    if Con.TOTAL_ANSWERING_RT in df.columns:
        cols.append(Con.TOTAL_ANSWERING_RT)
    return cols


def get_full_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the full model feature set:
      - area features
      - derived features
      - pattern-breaking features
      - last-before-confirm / last-before-select one-hots (LAST_ALL)
      - RT / TFD / TimeSinceOffset features (per-region: answer + paragraph)
    """
    cols: List[str] = []
    cols.extend(get_area_feature_cols(df))
    cols.extend(get_derived_feature_cols(df))
    cols.extend(get_pattern_feature_cols(df))
    cols.extend(get_last_visited_feature_cols(df))
    cols.extend(get_rt_tfd_feature_cols(df))
    return cols
