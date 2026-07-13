# model_data.py

from pathlib import Path
from typing import Sequence, Optional, Tuple, List
import ast
import pandas as pd
import numpy as np

from src import constants as Con
from src.data_paths import READY_ALL_FEATURES_PATH

from src.predictive_modeling.common.feature_builders import (
    build_area_metric_pivot,
    add_answer_correct_wrong_contrast_columns,
    build_trial_level_categorical_feature,
    build_trial_level_constant_numeric_features,
)

from src.predictive_modeling.common.prepared_dataset import PreparedTrialDataset
from src.predictive_modeling.common.feature_specs import (
    get_area_feature_cols,
    get_derived_feature_cols,
    get_full_feature_cols,
)

from src.derived.correctness_measures import (
    has_back_and_forth_xyx,
    has_back_and_forth_xyxy,
    longest_alternating_answer_run,
    compute_trial_mean_dwell_per_word,
)

from src.derived.preference_matching import compute_trial_matching

from src.derived.pattern_breaking import build_trial_level_pattern_features

from src.derived.reading_times import (
    ANSWER_REGIONS as RT_ANSWER_REGIONS,
    PARAGRAPH_REGIONS as RT_PARAGRAPH_REGIONS,
)

from src.constants import TRIAL_ID_COLS

# Standalone answer-region metric prefixes (column = f"{metric}_{region}").
ANSWER_RT_TFD_METRICS = (
    "RT_pure",
    "RT_normalized",
    "TFD_pure",
    "TFD_normalized",
    "TimeSinceOffset_pure",
    "TimeSinceOffset_normalized",
)

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Area features
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------

def build_trial_level_derived_features(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
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


# ---------------------------------------------------------------------
# RT / TFD / TimeSinceOffset features
# ---------------------------------------------------------------------

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
      Paragraph regions (outside, distractor, critical) have no TimeSinceOffset
      counterpart, so those columns are simply absent.
    - For each metric, correct/wrong contrast columns derived from the answer
      regions (answer_A is correct) — column names f"{metric}_correct",
      f"{metric}_wrong_mean", f"{metric}_contrast", f"{metric}_distance_furthest"
      and f"{metric}_distance_closest". The "question" and paragraph regions are
      left untouched.
    """
    keep_cols = _deduplicate_keep_cols(keep_cols)

    metric_cols = [
        f"{m}_{r}"
        for m in ANSWER_RT_TFD_METRICS
        for r in tuple(RT_ANSWER_REGIONS) + tuple(RT_PARAGRAPH_REGIONS)
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


# ---------------------------------------------------------------------
# Last visited categorical features
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Final merged modeling dataframe
# ---------------------------------------------------------------------

def build_trial_level_model_df(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = [Con.TEXT_ID_WITH_Q_COLUMN],
    target_col: str = Con.IS_CORRECT_COLUMN,
    include_area_features: bool = True,
    include_derived_features: bool = True,
    include_last_lbl_before_confirm_features: bool = True,
    include_last_lbl_before_select_features: bool = True,
    include_rt_tfd_features: bool = True,
    include_total_answering_rt: bool = True,
    include_pattern_features: bool = True,
    numeric_feature_cols: Sequence[str] = (Con.NUM_OF_SELECTS,),
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS_MODELING,
    area_col: str = Con.AREA_LABEL_COLUMN,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    dwell_col: str = Con.IA_DWELL_TIME,
) -> pd.DataFrame:
    """
    Build the final one-row-per-trial modeling dataframe.

    This is the main function you can use in the pipeline.
    """
    trial_core = _build_trial_core(
        df=df,
        target_col=target_col,
        keep_cols=keep_cols,
    )

    out = trial_core.copy()

    if include_area_features:
        area_df = build_trial_level_area_features(
            df=df,
            area_col=area_col,
            metric_cols=metric_cols,
            keep_cols=None,
            target_col=target_col,
            add_correct_wrong_contrasts=True,
        )
        drop_cols = [target_col]
        out = out.merge(
            area_df.drop(columns=[c for c in drop_cols if c in area_df.columns]),
            on=list(TRIAL_ID_COLS),
            how="left",
        )

    if include_derived_features:
        derived_df = build_trial_level_derived_features(
            df=df,
            seq_col=seq_col,
            dwell_col=dwell_col,
            keep_cols=None,
            target_col=target_col,
        )
        drop_cols = [target_col]
        out = out.merge(
            derived_df.drop(columns=[c for c in drop_cols if c in derived_df.columns]),
            on=list(TRIAL_ID_COLS),
            how="left",
        )

    if include_pattern_features:
        pattern_df = build_trial_level_pattern_features(df, kind="location", window_len=4)
        out = out.merge(pattern_df, on=list(TRIAL_ID_COLS), how="left")

    if include_last_lbl_before_confirm_features:
        last_before_confirm_df = build_trial_level_last_visited_features(
            df=df,
            feature_col=Con.LAST_LBL_BEFORE_CONFIRM,
            prefix="last_before_confirm",
        )
        out = out.merge(last_before_confirm_df, on=list(TRIAL_ID_COLS), how="left")

    if include_last_lbl_before_select_features:
        last_before_select_df = build_trial_level_last_visited_features(
            df=df,
            feature_col=Con.LAST_LBL_BEFORE_SELECT,
            prefix="last_before_select",
        )
        out = out.merge(last_before_select_df, on=list(TRIAL_ID_COLS), how="left")

    if include_rt_tfd_features:
        rt_tfd_df = build_trial_level_rt_tfd_features(
            df=df,
            target_col=target_col,
        )
        out = out.merge(
            rt_tfd_df.drop(columns=[c for c in [target_col] if c in rt_tfd_df.columns]),
            on=list(TRIAL_ID_COLS),
            how="left",
        )

    if numeric_feature_cols:
        numeric_df = build_trial_level_constant_numeric_features(
            df=df,
            feature_cols=numeric_feature_cols,
        )
        out = out.merge(numeric_df, on=list(TRIAL_ID_COLS), how="left")

    # Trial-level total answering RT, taken from the raw CONFIRM_FINAL_ANSWER_RT
    # column and exposed under the friendlier name TOTAL_ANSWERING_RT.
    if include_total_answering_rt and Con.CONFIRM_FINAL_ANSWER_RT in df.columns:
        feature_cols = [Con.CONFIRM_FINAL_ANSWER_RT]

        if Con.TOTAL_ANSWERING_RT_NORMALIZED in df.columns:
            feature_cols.append(Con.TOTAL_ANSWERING_RT_NORMALIZED)

        total_rt_df = build_trial_level_constant_numeric_features(
            df=df,
            feature_cols=feature_cols,
        ).rename(columns={
            Con.CONFIRM_FINAL_ANSWER_RT: Con.TOTAL_ANSWERING_RT,
        })

        out = out.merge(total_rt_df, on=list(TRIAL_ID_COLS), how="left")

    return out


# ---------------------------------------------------------------------
# Cached full feature CSV
# ---------------------------------------------------------------------

def save_all_features(
    df: pd.DataFrame,
    output_path: Path = READY_ALL_FEATURES_PATH,
    keep_cols: Optional[Sequence[str]] = [Con.TEXT_ID_WITH_Q_COLUMN],
    target_col: str = Con.IS_CORRECT_COLUMN,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the full trial-level feature DataFrame (every include_* flag turned on)
    and save it to `output_path` as CSV. The CSV can later be read back with
    `load_all_features`.
    """
    trial_df = build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=True,
        include_last_lbl_before_confirm_features=True,
        include_last_lbl_before_select_features=True,
        include_rt_tfd_features=True,
        include_total_answering_rt=True,
        include_pattern_features=True,
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


def load_all_features(path: Path = READY_ALL_FEATURES_PATH) -> pd.DataFrame:
    """
    Load the cached full feature DataFrame produced by `save_all_features`.
    """
    return pd.read_csv(Path(path))


# ---------------------------------------------------------------------
# Prepared dataset builders
# ---------------------------------------------------------------------

def make_area_only_dataset(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> PreparedTrialDataset:
    """
    Build a prepared dataset for area-only logistic regression.
    """
    trial_df = build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=False,
        include_rt_tfd_features=False,
    )

    feature_cols = get_area_feature_cols(trial_df)

    return PreparedTrialDataset(
        df=trial_df,
        feature_cols=feature_cols,
        target_col=target_col,
        id_cols=list(TRIAL_ID_COLS),
    )


def make_derived_dataset(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> PreparedTrialDataset:
    """
    Build a prepared dataset for derived-feature logistic regression.
    """
    trial_df = build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=False,
        include_derived_features=True,
        include_rt_tfd_features=False,
    )

    feature_cols = get_derived_feature_cols(trial_df)

    return PreparedTrialDataset(
        df=trial_df,
        feature_cols=feature_cols,
        target_col=target_col,
        id_cols=list(TRIAL_ID_COLS),
    )


def make_full_dataset(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> PreparedTrialDataset:
    """
    Build a prepared dataset for the full logistic regression model.
    """
    trial_df = build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=True,
        include_rt_tfd_features=True,
    )

    feature_cols = get_full_feature_cols(trial_df)

    return PreparedTrialDataset(
        df=trial_df,
        feature_cols=feature_cols,
        target_col=target_col,
        id_cols=list(TRIAL_ID_COLS),
    )