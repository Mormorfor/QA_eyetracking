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
#   feature_cols = FG.GENERAL_FEATURES + FG.PATTERN_INTERACTION_COLS
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
# Last-visited / last-before-action one-hot groups
# ---------------------------------------------------------------------------

# Each "last before action" feature comes in two flavors:
#   * LONG    -- the original one-hot encoding, one column per label.
#   * COMPACT -- the collapsed correct / wrong / question indicators.

LAST_CONFIRM_LONG: List[str] = [
    "last_before_confirm_answer_A",
    "last_before_confirm_answer_B",
    "last_before_confirm_answer_C",
    #"last_before_confirm_answer_D",
    "last_before_confirm_question",
]

LAST_CONFIRM_COMPACT: List[str] = [
    "last_before_confirm_correct",
    #"last_before_confirm_wrong",
    "last_before_confirm_question",
]

LAST_SELECT_LONG: List[str] = [
    "last_before_select_answer_A",
    "last_before_select_answer_B",
    "last_before_select_answer_C",
    #"last_before_select_answer_D",
    "last_before_select_question",
]

LAST_SELECT_COMPACT: List[str] = [
    "last_before_select_correct",
    #"last_before_select_wrong",
    "last_before_select_question",
]

# Backwards-compatible aliases: the bare names default to the original
# (long / one-hot) form.
LAST_CONFIRM: List[str] = LAST_CONFIRM_LONG
LAST_SELECT: List[str] = LAST_SELECT_LONG

LAST_ALL: List[str] = LAST_CONFIRM_LONG + LAST_SELECT_LONG
LAST_ALL_COMPACT: List[str] = LAST_CONFIRM_COMPACT + LAST_SELECT_COMPACT


# ---------------------------------------------------------------------------
# RT / TFD / TimeSinceOffset feature groups
# (column names produced by build_trial_level_rt_tfd_features in model_data.py)
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

RT_COLS: List[str] = (
    [f"RT_{v}_{r}" for v in RT_TFD_VARIANTS for r in RT_TFD_NON_ANSWER_REGIONS]
    + [f"RT_{v}_{s}" for v in RT_TFD_VARIANTS for s in RT_TFD_CONTRAST_SUFFIXES]
)

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
    ALL_FEATURES_NO_LAST + LAST_CONFIRM + LAST_SELECT
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

SELECT_1_COLS: List[str] = (
    ['area_dwell_proportion__correct', 'area_dwell_proportion__question', 
    'skip_rate__correct', 
    'has_xyx', 'ANSWER_PRESS_NUMBER',
    'num_label_visits__correct', 'num_label_visits__contrast',
    'mean_fixations_count__question', 'mean_fixations_count__wrong_mean', 
    'mean_max_fix_pupil_size_z__correct'
    ]
)
