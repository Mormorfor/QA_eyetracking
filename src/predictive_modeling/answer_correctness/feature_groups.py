"""
Named feature-column groups for answer-correctness modeling.

Single source of truth for every named feature-list used across notebooks
and column-set generators (run_model_bundles, generate_column_options, etc.).
"""

from __future__ import annotations

from typing import List

from src import constants as Con

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
    "trial_mean_dwell",
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
# Last-visited / last-before-action one-hot groups
# ---------------------------------------------------------------------------

LAST_ANSWER: List[str] = [
    "last_visited_answer_A",
    "last_visited_answer_B",
    "last_visited_answer_C",
    "last_visited_answer_D",
]

LAST_CONFIRM: List[str] = [
    "last_before_confirm_answer_A",
    "last_before_confirm_answer_B",
    "last_before_confirm_answer_C",
    "last_before_confirm_answer_D",
    "last_before_confirm_question",
]

LAST_SELECT: List[str] = [
    "last_before_select_answer_A",
    "last_before_select_answer_B",
    "last_before_select_answer_C",
    "last_before_select_answer_D",
    "last_before_select_question",
]

LAST_ALL: List[str] = LAST_ANSWER + LAST_CONFIRM + LAST_SELECT


# ---------------------------------------------------------------------------
# Manually curated feature subsets
# ---------------------------------------------------------------------------

SELECT_1_METRIC_COLUMNS: List[str] = [
    Con.SKIP_RATE,
    Con.AREA_DWELL_PROPORTION,
    Con.NUM_LABEL_VISITS,
]

SELECT_1_COLS: List[str] = (
    [f"{m}__{Con.CORRECT_SUFFIX}" for m in SELECT_1_METRIC_COLUMNS]
    + [f"{m}__{Con.WRONG_MEAN_SUFFIX}" for m in SELECT_1_METRIC_COLUMNS]
    + ["seq_len", "has_xyx"]
)


# ---------------------------------------------------------------------------
# RT / TFD / TimeSinceOffset feature groups
# (column names produced by build_trial_level_rt_tfd_features in model_data.py)
# ---------------------------------------------------------------------------

RT_TFD_ANSWER_REGIONS: List[str] = [
    "question",
    "answer_A",
    "answer_B",
    "answer_C",
    "answer_D",
]
RT_TFD_PARAGRAPH_REGIONS: List[str] = ["outside", "distractor", "critical"]
RT_TFD_VARIANTS: List[str] = ["pure", "normalized"]
RT_TFD_INTERACTION_SEP: str = "__x__"

RT_COLS: List[str] = [
    f"RT_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_ANSWER_REGIONS
]

TFD_COLS: List[str] = [
    f"TFD_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_ANSWER_REGIONS
]

TIME_SINCE_OFFSET_COLS: List[str] = [
    f"TimeSinceOffset_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_ANSWER_REGIONS
]

RT_INTERACTION_COLS: List[str] = [
    f"RT_{v}_{p}{RT_TFD_INTERACTION_SEP}RT_{v}_{a}"
    for v in RT_TFD_VARIANTS
    for p in RT_TFD_PARAGRAPH_REGIONS
    for a in RT_TFD_ANSWER_REGIONS
]

TFD_INTERACTION_COLS: List[str] = [
    f"TFD_{v}_{p}{RT_TFD_INTERACTION_SEP}TFD_{v}_{a}"
    for v in RT_TFD_VARIANTS
    for p in RT_TFD_PARAGRAPH_REGIONS
    for a in RT_TFD_ANSWER_REGIONS
]

RT_TFD_OFFSET_COLS: List[str] = RT_COLS + TFD_COLS + TIME_SINCE_OFFSET_COLS
RT_TFD_INTERACTION_COLS: List[str] = RT_INTERACTION_COLS + TFD_INTERACTION_COLS


# ---------------------------------------------------------------------------
# Aggregate "all features" sets
# ---------------------------------------------------------------------------

ALL_FEATURES_NO_LAST: List[str] = (
    AREA_COLS
    + PER_QUESTION_COLS
    + DERIVED_COLS
    + [Con.NUM_OF_SELECTS]
    + RT_COLS
    + TFD_COLS
    + TIME_SINCE_OFFSET_COLS
    + RT_INTERACTION_COLS
    + TFD_INTERACTION_COLS
)

ALL_FEATURES: List[str] = (
    ALL_FEATURES_NO_LAST + LAST_ANSWER + LAST_CONFIRM + LAST_SELECT
)


# ---------------------------------------------------------------------------
# General features
#   = ALL_FEATURES_NO_LAST minus RT/TFD/TSO base columns and their
#     interaction terms. Used as the "general" base for additive groupings.
# ---------------------------------------------------------------------------

GENERAL_FEATURES: List[str] = (
    AREA_COLS + PER_QUESTION_COLS + DERIVED_COLS + [Con.NUM_OF_SELECTS]
)
