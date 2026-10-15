"""Build the final table: one row per trial, every feature a model can use.

This is where everything measured per word or per area becomes one row per
trial. `build_trial_level_model_df` puts together, for each trial:

    area measurements    one column per measurement and area
                         (mean_dwell_time__answer_B), plus the comparisons of
                         the correct answer with the wrong ones
                         (mean_dwell_time__contrast)
    sequence statistics  seq_len, has_xyx, ... (made during prep)
    opening scan         breaks_pattern_*, dominance_score_*, ...
    paragraph features   paragraph-screen measurements (Study 1 only)
    last look            which area was looked at before the selection and
                         confirmation presses, as 0/1 columns
    reading times        RT_*, TFD_*, TimeSinceOffset_* per answer area, plus
                         comparisons
    answering            ANSWER_PRESS_NUMBER, total_answering_RT(_normalized)

`save_model_ready` builds it and saves it as `features/model_ready.csv`;
`load_model_ready` reads it back.

This module imports only from `features/` and `config/`.
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
from src.config.screens import screen
from src.features.sequences import (
    DERIVED_COLS,
    compute_trial_mean_dwell_per_word,
    has_back_and_forth_xyx,
    has_back_and_forth_xyxy,
    longest_alternating_answer_run,
    parse_seq,
    sequence_len_literal_eval,
)
from src.features.strategies import build_trial_level_pattern_features


# ---------------------------------------------------------------------------
# Columns made in other files that this file takes into the model table by
# name. Which of them a model uses is decided in modeling/feature_sets.py.
# ---------------------------------------------------------------------------

ANSWER_SCREEN = screen("answers")
PARAGRAPH_SCREEN = screen("paragraph")

# From the paragraph table (features/paragraph/spans.py, saved at
# PARAGRAPH_SPAN_FEATURES_PATH), taken as they are: each part's share of the
# looking time (area_dwell_proportion__critical, ...) and its reading times
# (RT_normalized_critical, ...).
PARAGRAPH_PROPORTION_COLS = tuple(
    f"{Con.AREA_DWELL_PROPORTION}{Con.DERIVED_SEP}{span}"
    for span in PARAGRAPH_SCREEN.regions
)
PARAGRAPH_RT_TFD_COLS = tuple(
    f"{metric}_{span}"
    for metric in Con.RT_TFD_METRICS
    for span in PARAGRAPH_SCREEN.regions
)

PARAGRAPH_MODEL_COLS = PARAGRAPH_PROPORTION_COLS + PARAGRAPH_RT_TFD_COLS


def build_area_metric_pivot(
    df: pd.DataFrame,
    area_col: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Turn the per-area measurements into one row per trial, one column per
    measurement and area: `<metric>__<area>`, e.g. `skip_rate__answer_B`.

    `df` is the word-level frame, where each area's values are repeated on
    every word of that area. Words with no area are ignored.
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
    correct_label: str = Con.CORRECT_ANSWER_AREA,
    wrong_labels: Sequence[str] = tuple(Con.WRONG_ANSWER_AREAS),
    out_correct_suffix: str = Con.CORRECT_SUFFIX,
    out_wrong_mean_suffix: str = Con.WRONG_MEAN_SUFFIX,
    out_contrast_suffix: str = Con.CONTRAST_SUFFIX,
    out_distance_furthest_suffix: str = Con.DISTANCE_FURTHEST_SUFFIX,
    out_distance_closest_suffix: str = Con.DISTANCE_CLOSEST_SUFFIX,
) -> pd.DataFrame:
    """Add the five comparisons of the correct answer with the wrong ones.

    For each metric, given `<metric>__answer_A` .. `<metric>__answer_D`, adds:

        <metric>__correct            the value on the correct answer (answer_A)
        <metric>__wrong_mean         the average over the three wrong answers
        <metric>__contrast           correct minus wrong_mean
        <metric>__distance_furthest  the largest gap between the correct answer
                                     and any one wrong answer
        <metric>__distance_closest   the smallest such gap

    The two distances are absolute values. Wrong answers with no value are
    skipped, so a comparison is missing only if the correct answer has no value
    or none of the wrong answers does.
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
    """One row per trial for columns that have one value per trial (taken from
    the trial's first row), as numbers."""
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
    """Turn a column with one category per trial into 0/1 columns, one per
    category (`<prefix>_<category>`). One row per trial."""
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
    """`keep_cols` as a list, without the two trial-id columns."""
    keep_cols = list(keep_cols) if keep_cols is not None else []
    return [c for c in keep_cols if c not in TRIAL_ID_COLS]


def _build_trial_core(
    df: pd.DataFrame,
    target_col: str = Con.IS_CORRECT_COLUMN,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """One row per trial with the ids, the target (`is_correct`) and `keep_cols`.

    Trials with no target value are dropped. (There should be none: a missing
    answer stops the run when `is_correct` is made.)
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
    """One row per trial: the target, `keep_cols`, every `<metric>__<area>`
    column, and (if `add_correct_wrong_contrasts`) the five comparisons of the
    correct answer with the wrong ones."""
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
            correct_label=Con.CORRECT_ANSWER_AREA,
            wrong_labels=Con.WRONG_ANSWER_AREAS,
        )

    out = trial_core.merge(metrics_pivot, on=list(TRIAL_ID_COLS), how="left")
    return out


def build_trial_level_rt_tfd_features(
    df: pd.DataFrame,
    target_col: str = Con.IS_CORRECT_COLUMN,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """The answer screen's reading times, one row per trial.

    One column per measure and area, e.g. `RT_normalized_answer_B` (see
    `features/reading_times.py`), plus the five comparisons of the correct
    answer with the wrong ones, e.g. `RT_normalized_contrast`. Note the single
    "_" in these names, where the area measurements use "__".

    Answer-screen areas only; the paragraph parts' reading times come with the
    paragraph features.

    Stops with an error if any of these columns is missing from `df`.
    """
    keep_cols = _deduplicate_keep_cols(keep_cols)

    # Answer-screen areas only. The paragraph parts' columns come in through the
    # paragraph join; taking them here too would give two columns of the same
    # name, which pandas would quietly rename to `_x` / `_y`.
    metric_cols = [
        f"{m}_{r}" for m in Con.RT_TFD_METRICS for r in ANSWER_SCREEN.regions
    ]
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"reading-time columns missing: {missing}. They are made by "
            "features/reading_times.py when the dataset is built with reading "
            "times (ingest/build.py::main, add_rts=True)."
        )

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

    # The comparisons of the correct answer with the wrong ones, for every
    # measure. These names use a single "_".
    trial = add_answer_correct_wrong_contrast_columns(
        df_pivot=trial,
        metric_cols=list(Con.RT_TFD_METRICS),
        sep="_",
        correct_label=Con.CORRECT_ANSWER_AREA,
        wrong_labels=Con.WRONG_ANSWER_AREAS,
    )

    return trial


def build_trial_level_paragraph_features(
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    feature_cols: Sequence[str] = PARAGRAPH_MODEL_COLS,
) -> pd.DataFrame:
    """One row per trial with the paragraph-screen features:

        area_dwell_proportion__critical / __distractor / __outside
        RT_* / TFD_* / TimeSinceOffset_* for each paragraph part

    The share of looking time is the same measure as on the answer screen, but
    over the three paragraph parts. Read from the file written by
    `features.paragraph.spans.save_paragraph_features` (or passed in as
    `paragraph_features`).

    Stops with an error if the file is missing or lacks any of
    `feature_cols`; rebuild it with `save_paragraph_features()`.
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
# Names of the "last look" 0/1 columns made by
# `build_trial_level_last_visited_features` below.
#
# Each list leaves one category out (answer_D, or "wrong" in the compact
# lists). With every category included, the columns would add up to 1 on
# almost every trial.
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

# Short names for the per-answer lists.
LAST_CONFIRM: List[str] = LAST_CONFIRM_LONG

LAST_SELECT: List[str] = LAST_SELECT_LONG

LAST_ALL: List[str] = LAST_CONFIRM_LONG + LAST_SELECT_LONG

LAST_ALL_COMPACT: List[str] = LAST_CONFIRM_COMPACT + LAST_SELECT_COMPACT


def build_trial_level_last_visited_features(
    df: pd.DataFrame,
    feature_col: str = Con.LAST_VISITED_LABEL,
    prefix: str = "last_visited",
) -> pd.DataFrame:
    """A "last look" area as 0/1 columns, one row per trial.

    One column per area (`<prefix>_answer_A`, ..., `<prefix>_question`, and
    `<prefix>_nan` when there was no last look), plus three summary columns:

      <prefix>_correct   the last look was at the correct answer (answer_A)
      <prefix>_wrong     it was at a wrong answer (B, C or D)
      <prefix>_question  it was at the question

    The pipeline calls this for the look before confirming and before selecting.
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

    out[f"{prefix}_correct"] = _dummy(Con.CORRECT_ANSWER_AREA)
    wrong = pd.Series(0, index=out.index, dtype=int)
    for label in Con.WRONG_ANSWER_AREAS:
        wrong = wrong | _dummy(label)
    out[f"{prefix}_wrong"] = wrong.astype(int)
    out[f"{prefix}_{Con.QUESTION_AREA}"] = _dummy(Con.QUESTION_AREA)

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
    """Build the final table: one row per trial, every feature (see the module
    docstring for the list).

    `df` is the wide word-level frame (`load_model_input` makes it). `dataset`
    says which dataset it is; the paragraph features are added only if that
    dataset has a paragraph screen. They are the one part not computed from
    `df` -- they come from the paragraph features file.

    `keep_cols` are carried through and must exist. `optional_keep_cols`
    (`regime`, `session_id`) are carried only if present: Study 2 has them,
    Study 1 does not.

    `pattern_scope_df` / `pattern_scope_by` choose which trials the opening-scan
    features' dominant strategy is worked out from; see
    `strategies.build_trial_level_pattern_features`. By default, `df` itself.

    Every block except the paragraph one is checked to have exactly one row for
    every trial before it is joined, and the run stops if not.
    """
    # Whether the dataset has a paragraph screen is read from its record.
    # `dataset` can be a key ("l1") or the record itself.
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

    # The five sequence statistics were made during prep and are already on
    # `df`, so they are carried over, not recomputed.
    derived_df = (
        df.groupby(list(TRIAL_ID_COLS), as_index=False)[list(DERIVED_COLS)].first()
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
        # Two checks before joining. If the paragraph file shares no trial with
        # this table, the file is out of date or the trial ids do not match --
        # stop. And if a column name is in both, pandas would quietly rename the
        # two copies `_x` / `_y`, and the column every feature list refers to
        # would vanish -- stop.
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
                f"frame: {sorted(clash)}. These should come only from the "
                "paragraph features file. The usual cause is a word table that "
                "still holds paragraph reading-time columns; rebuild the dataset."
            )
        # Unlike the other blocks, this one is not required to cover every
        # trial, because it comes from a separate file rather than from `df`.
        # It still must not add rows, which repeated trials in the file would.
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

    # The time from the answers appearing to the confirming press, from the raw
    # CONFIRM_FINAL_ANSWER_RT column, under the clearer name total_answering_RT.
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
# Reading the input, and saving and loading the final table
# ===========================================================================

def load_model_input(dataset, facet: str = None, verbose: bool = True) -> pd.DataFrame:
    """Read a dataset's three tables and join them back into one wide
    word-level frame -- the input `build_trial_level_model_df` takes.

    Every word row gets its area's and its trial's values copied onto it.
    `facet` reads one group's tables instead, e.g. "hunters" or "gatherers".

    Stops with an error naming the build command if the tables are missing.

    Careful: the two joins are not checked. A word whose area or trial is
    missing from the other tables would get empty values for them, with no
    error.
    """
    def _path(grain):
        return (
            dataset.grain_split(grain, facet) if facet else getattr(dataset, grain)
        )

    missing = [
        name
        for name in ("interest_areas", "areas_answers", "trials")
        if not _path(name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{dataset.key}: no grain tables ({', '.join(missing)}). Run "
            f"`python scripts/build_dataset.py --dataset {dataset.key}` to build "
            f"them, or pass a word-level frame directly."
        )

    read = dict(dtype={Con.PARTICIPANT_ID: str}, low_memory=False)
    ia = pd.read_csv(_path("interest_areas"), **read)
    areas = pd.read_csv(_path("areas_answers"), **read)
    trials = pd.read_csv(_path("trials"), **read)
    if verbose:
        print(
            f"  grains: interest_areas {ia.shape}, areas_answers {areas.shape}, "
            f"trials {trials.shape}"
        )

    area_key = list(TRIAL_ID_COLS) + [Con.AREA_LABEL_COLUMN]
    out = ia.merge(areas, on=area_key, how="left")
    out = out.merge(trials, on=list(TRIAL_ID_COLS), how="left")
    return out


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
    """Build the final table (`build_trial_level_model_df`) and save it to
    `output_path`. Read it back with `load_model_ready`.

    Careful: this table covers the whole dataset, so its opening-scan features
    (dominance score etc.) are worked out from every trial a participant has.
    That is right for describing the data, but wrong to hand to
    cross-validation as a ready-made `trial_df` -- the scores would include the
    held-out trials. `pattern_scope_by=["regime"]` gives Study 2 one dominance
    score per regime.

    `dataset` says which dataset this is; paragraph features are added only if
    it has a paragraph screen.
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
    """Read the final table saved by `save_model_ready`.

    `participant_id` is read as text. Study 2's ids are all digits (`4000`), so
    pandas would otherwise read them as numbers, and joins with other tables,
    where they are text, would fail.
    """
    return pd.read_csv(Path(path), dtype={Con.PARTICIPANT_ID: str})
