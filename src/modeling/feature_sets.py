"""Named lists of the columns a model uses.

These are hand-written choices. What the pipeline makes, and under what names,
is decided in `features/`; this file only picks from it.

A feature set is written one of two ways, and they are not interchangeable:

* **Fixed lists** (`SELECT_1_COLS`, `GENERAL_FEATURES`, `RT_COLS`, ...) name
  columns outright. A model given one gets exactly those columns, and a missing
  column stops the fit with an error.
* **Selectors** (`get_*_feature_cols(df)`) return whichever of their columns a
  given table has. They are for tables whose blocks vary -- Study 2 has no
  paragraph columns, for instance.

**The two disagree on the reading-time columns.** The fixed lists use only the
per-word versions (`_normalized`), while `get_rt_tfd_feature_cols` returns both
the per-word and the total (`_pure`) versions, whichever are present. So "the reading-time features" is a different
set depending on which you use. Making them agree would change which columns
existing models see, so it is a modelling decision, not a tidy-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import columns as Con
from src.config.outputs import FEATURE_SETS_DIR
from src.config.screens import screen

SEP = Con.DERIVED_SEP
ANSWER_SCREEN = screen("answers")
PARAGRAPH_SCREEN = screen("paragraph")

# The last-look column lists are defined in `features/build.py`, which makes
# those columns. They are re-exported here because code that picks feature sets
# reads them from this module.
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

# The five sequence statistics. Imported from `features/sequences.py`, which
# makes them, so a feature set can only name ones the pipeline actually builds.
from src.features.sequences import DERIVED_COLS  # noqa: E402


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

# Graded "breaks pattern": the edit distance between the trial's starting
# strategy and the participant's dominant one -- how many single entries must
# change to turn one into the other. It is 0 exactly when the trial does not
# break the pattern, so it overlaps with breaks_pattern_*; kept as a separate
# group, to use *instead of* breaks_pattern rather than beside it.
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
    [f"{m}{SEP}{Con.CORRECT_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}{SEP}{Con.WRONG_MEAN_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}{SEP}{Con.CONTRAST_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}{SEP}{Con.DISTANCE_FURTHEST_SUFFIX}" for m in METRIC_COLUMNS]
    + [f"{m}{SEP}{Con.DISTANCE_CLOSEST_SUFFIX}" for m in METRIC_COLUMNS]
)


# ---------------------------------------------------------------------------
# The question area's own columns: <metric>__question
#   The four answers have no columns of their own in the feature sets; the
#   comparisons above stand in for them.
# ---------------------------------------------------------------------------

PER_QUESTION_COLS: List[str] = [f"{m}{SEP}{Con.QUESTION_AREA}" for m in METRIC_COLUMNS]


# ---------------------------------------------------------------------------
# Paragraph-screen features (Study 1 only)
#   The share of the paragraph-screen looking time spent on each part:
#   critical, distractor and outside. The same measure as the answer areas'
#   area_dwell_proportion__*, joined in from features/paragraph_spans.csv.
#
#   Not in ALL_FEATURES / GENERAL_FEATURES; add them on purpose, e.g.
#       feature_cols = FS.GENERAL_FEATURES + FS.PARAGRAPH_BASED
#   The three shares add up to 1, so with an intercept only two of them carry
#   information.
# ---------------------------------------------------------------------------

PARAGRAPH_BASED: List[str] = [
    f"{Con.AREA_DWELL_PROPORTION}{SEP}{span}" for span in PARAGRAPH_SCREEN.regions
]


# ---------------------------------------------------------------------------
# The area looked at just before each button press, as 0/1 columns
#
# Two versions of each:
#   * LONG    -- one column per area, with answer_D left out as the reference
#   * COMPACT -- "the correct answer" and "the question" only
#
# The lists are imported at the top of this module from `features/build.py`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RT / TFD / TimeSinceOffset feature groups
# (column names produced by build_trial_level_rt_tfd_features)
# ---------------------------------------------------------------------------

RT_TFD_PARAGRAPH_REGIONS: List[str] = list(PARAGRAPH_SCREEN.regions)

# The fixed lists below use the per-word versions only (`_normalized`); the
# `_pure` prefixes would add the totals.

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
RT_TFD_NON_ANSWER_REGIONS: List[str] = [Con.QUESTION_AREA] + RT_TFD_PARAGRAPH_REGIONS

RT_COLS: List[str] = [
    f"{Con.RT_NORMALIZED}_{r}" for r in RT_TFD_NON_ANSWER_REGIONS
] + [f"{Con.RT_NORMALIZED}_{s}" for s in RT_TFD_CONTRAST_SUFFIXES]

TFD_COLS: List[str] = [
    f"{Con.TFD_NORMALIZED}_{r}" for r in RT_TFD_NON_ANSWER_REGIONS
] + [f"{Con.TFD_NORMALIZED}_{s}" for s in RT_TFD_CONTRAST_SUFFIXES]

# The question and the answer comparisons. Unlike RT_COLS and TFD_COLS, this
# leaves out the paragraph parts, although the model table has
# TimeSinceOffset_* columns for them too.
TIME_SINCE_OFFSET_COLS: List[str] = [
    f"{Con.TIME_SINCE_OFFSET_NORMALIZED}_{Con.QUESTION_AREA}"
] + [f"{Con.TIME_SINCE_OFFSET_NORMALIZED}_{s}" for s in RT_TFD_CONTRAST_SUFFIXES]


RT_TFD_OFFSET_COLS: List[str] = RT_COLS + TFD_COLS + TIME_SINCE_OFFSET_COLS


# ---------------------------------------------------------------------------
# Per-answer RT / TFD / TimeSinceOffset groups (raw answer_A-D columns).
# Available as standalone named groups for targeted experiments, but
# deliberately NOT part of ALL_FEATURES / GENERAL_FEATURES (those use the
# correct/wrong contrast representation of the answers instead).
# ---------------------------------------------------------------------------

RT_TFD_PER_ANSWER_REGIONS: List[str] = list(Con.ANSWER_AREAS)

RT_PER_ANSWER_COLS: List[str] = [
    f"{Con.RT_NORMALIZED}_{r}" for r in RT_TFD_PER_ANSWER_REGIONS
]

TFD_PER_ANSWER_COLS: List[str] = [
    f"{Con.TFD_NORMALIZED}_{r}" for r in RT_TFD_PER_ANSWER_REGIONS
]

TIME_SINCE_OFFSET_PER_ANSWER_COLS: List[str] = [
    f"{Con.TIME_SINCE_OFFSET_NORMALIZED}_{r}" for r in RT_TFD_PER_ANSWER_REGIONS
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
#   ALL_FEATURES_NO_LAST without the reading-time columns. The usual base that
#   other groups are added to.
# ---------------------------------------------------------------------------

GENERAL_FEATURES: List[str] = (
    AREA_COLS + PER_QUESTION_COLS + DERIVED_COLS + PATTERN_COLS + [Con.NUM_OF_SELECTS]
)


# ---------------------------------------------------------------------------
# Hand-picked sets
#   SELECT_1_COLS: the ten features of the paper's headline model.
#   feature_sets/select_1.json holds the same list.
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
# Selectors: which of these columns does this table have?
# ===========================================================================

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
            col = f"{metric}{SEP}{area}"
            if col in df.columns:
                cols.append(col)

        for suffix in derived_suffixes:
            col = f"{metric}{SEP}{suffix}"
            if col in df.columns:
                cols.append(col)

    return cols


def get_derived_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the five sequence statistics (`DERIVED_COLS`) present in df.

    Preference matching is deliberately not a model feature. It is used as an
    analysis instead: `analyses/correctness_associations/plots.py` calls
    `features/preference.py::compute_trial_matching` directly, for each measure
    and direction it needs. It is a family with settings, not one column.
    """
    return [c for c in DERIVED_COLS if c in df.columns]


def get_pattern_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return per-trial pattern-breaking feature columns that are present in df.
    """
    return [c for c in PATTERN_COLS if c in df.columns]


def get_last_visited_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the last-look 0/1 columns of `LAST_ALL` present in df.

    Not every last-look column the pipeline builds. Each of the two families
    (before confirming, before selecting) has six 0/1 columns -- answer_A to
    answer_D, question and nan -- that add up to 1, so with an intercept they
    are perfectly collinear; its `correct` and `wrong` columns are sums of the
    same six. `LAST_ALL` is the usable encoding: one column per area, with
    answer_D left out as the reference and nan / correct / wrong left out.
    Picking the columns by their name prefix instead would bring the collinear
    set back, and with the model's L2 penalty that gives unstable coefficients
    rather than an error.
    """
    return [c for c in LAST_ALL if c in df.columns]


def get_rt_tfd_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the reading-time columns present in df:

      - per area: f"{metric}_{area}" for each of the six measures in
        `Con.RT_TFD_METRICS` -- both the total (`_pure`) and the per-word
        (`_normalized`) versions, unlike the fixed lists above; that is the
        disagreement described at the top of this file, and it is deliberate --
        and each area of the answer screen and of the paragraph (the paragraph
        parts exist only for Study 1);
      - the comparisons of the correct answer with the wrong ones:
        f"{metric}_{suffix}" for each suffix in RT_TFD_CONTRAST_SUFFIXES;
      - the total answering time (`total_answering_RT`).
    """
    cols: List[str] = []
    regions = list(ANSWER_SCREEN.regions) + list(RT_TFD_PARAGRAPH_REGIONS)
    for metric in Con.RT_TFD_METRICS:
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


# ===========================================================================
# The sets saved in configs/feature_sets/
# ===========================================================================

def load_feature_sets(folder: Path = FEATURE_SETS_DIR) -> Dict[str, List[str]]:
    """The feature sets saved as JSON in `configs/feature_sets/`, as `{name: columns}`.

    Each file holds `{"identifier": ..., "columns": [...]}`. A set's name is the
    identifier inside its file, not the file name: `total_answering_RT.json` holds
    the set `total_answering_RT_normalized`. Files are read in name order.

    Stops if two files use the same identifier, since one name would then stand
    for two different sets.
    """
    sets: Dict[str, List[str]] = {}
    for path in sorted(Path(folder).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = payload["identifier"]
        if name in sets:
            raise ValueError(f"Two files in {folder} use the identifier {name!r}.")
        sets[name] = list(payload["columns"])
    return sets
