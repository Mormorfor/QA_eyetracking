# answer_correctness_data.py

from typing import Sequence, Optional
import pandas as pd
import numpy as np

from src import constants as Con
from src.predictive_modeling.common.data_utils import build_area_metric_pivot


from src.derived.correctness_measures import (
    sequence_len_literal_eval,
    parse_seq,
    has_back_and_forth_xyx,
    has_back_and_forth_xyxy,
    compute_trial_mean_dwell_per_word,
)

from src.derived.preference_matching import compute_trial_matching



def build_trial_level_with_area_metrics_for_correctness(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
) -> pd.DataFrame:
    core_cols = list(group_cols) + [
        last_loc_col,
        Con.SELECTED_ANSWER_POSITION_COLUMN,
        Con.CORRECT_ANSWER_POSITION_COLUMN,
    ]
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        raise KeyError(f"Missing required columns for trial core: {missing_core}")

    trial_core = (
        df[core_cols]
        .drop_duplicates()
        .dropna(
            subset=[
                last_loc_col,
                Con.SELECTED_ANSWER_POSITION_COLUMN,
                Con.CORRECT_ANSWER_POSITION_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )

    trial_core[Con.IS_CORRECT_COLUMN] = (
        trial_core[Con.SELECTED_ANSWER_POSITION_COLUMN]
        == trial_core[Con.CORRECT_ANSWER_POSITION_COLUMN]
    ).astype(int)

    metrics_pivot = build_area_metric_pivot(
        df=df,
        group_cols=group_cols,
        area_col=area_col,
        metric_cols=metric_cols,
    )

    trial_df = trial_core.merge(metrics_pivot, on=list(group_cols), how="left")
    return trial_df


def _safe_pref_feature_name(metric_col: str) -> str:
    return f"pref_matching__{metric_col}"


def build_trial_level_with_derived_features_for_correctness(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    dwell_col: str = Con.IA_DWELL_TIME,
    pref_metric_cols: Optional[Sequence[str]] = None,
    pref_direction: str = "high",
    pref_extreme_mode: str = "polarity",
) -> pd.DataFrame:
    """
    One row per trial (group_cols), containing:
      - is_correct (0/1)
      - seq_len
      - has_xyx (0/1)
      - has_xyxy (0/1)
      - trial_mean_dwell
      - pref_matching__<metric> for each metric in pref_metric_cols (0/1)

    """
    required = [*group_cols, Con.IS_CORRECT_COLUMN, seq_col, dwell_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for derived features: {missing}")

    d = df[list(dict.fromkeys(required))].copy()
    d[Con.IS_CORRECT_COLUMN] = d[Con.IS_CORRECT_COLUMN].astype(int)

    # --- seq len
    d["_seq_len"] = d[seq_col].apply(sequence_len_literal_eval)

    # --- XYX / XYXY
    d["_seq"] = d[seq_col].apply(parse_seq)
    d["_has_xyx"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyx(s)) if s is not None else False)
    d["_has_xyxy"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyxy(s)) if s is not None else False)

    # --- mean dwell per word (trial-level)
    d["_trial_mean_dwell"] = compute_trial_mean_dwell_per_word(d, dwell_col=dwell_col)

    trial_feat = (
        d.groupby(list(group_cols), as_index=False)
        .agg(
            is_correct=(Con.IS_CORRECT_COLUMN, "first"),
            seq_len=("_seq_len", "first"),
            has_xyx=("_has_xyx", "first"),
            has_xyxy=("_has_xyxy", "first"),
            trial_mean_dwell=("_trial_mean_dwell", "first"),
        )
    )

    trial_feat["has_xyx"] = trial_feat["has_xyx"].astype(int)
    trial_feat["has_xyxy"] = trial_feat["has_xyxy"].astype(int)

    # --- preference matching features (multiple)
    pref_metric_cols = list(pref_metric_cols) if pref_metric_cols else []
    for metric_col in pref_metric_cols:
        if metric_col not in df.columns:
            raise KeyError(f"Preference metric column '{metric_col}' not found in df.")

        pref_df = compute_trial_matching(
            df=df,
            metric_col=metric_col,
            direction=pref_direction,
            extreme_mode=pref_extreme_mode,
            out_col="pref_group",
        )

        feat_col = _safe_pref_feature_name(metric_col)

        pref_df = pref_df[list(group_cols) + ["pref_group"]].copy()
        pref_df[feat_col] = (pref_df["pref_group"] == "matching").astype(int)
        pref_df = pref_df.drop(columns=["pref_group"])

        trial_feat = trial_feat.merge(pref_df, on=list(group_cols), how="left")

    pref_cols = [c for c in trial_feat.columns if c.startswith("pref_matching__")]
    if pref_cols:
        trial_feat[pref_cols] = trial_feat[pref_cols].fillna(0).astype(int)

    return trial_feat



def build_trial_level_full_features_for_correctness(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_metric_cols: Optional[Sequence[str]] = None,
    pref_direction: str = "high",
    pref_extreme_mode: str = "polarity",
    include_last_location: bool = False,
) -> pd.DataFrame:
    """
    Builds ONE row per trial×participant containing:
      - is_correct
      - area-metric pivot features
      - derived behavioral features
      - multiple preference-matching features
      - optional last_loc_numeric
    """

    # --- area-metric features (already trial-level)
    area_df = build_trial_level_with_area_metrics_for_correctness(
        df=df,
        group_cols=group_cols,
    )

    # --- derived features (also trial-level)
    derived_df = build_trial_level_with_derived_features_for_correctness(
        df=df,
        group_cols=group_cols,
        pref_metric_cols=pref_metric_cols,
        pref_direction=pref_direction,
        pref_extreme_mode=pref_extreme_mode,
    )

    # --- merge
    merged = area_df.merge(
        derived_df,
        on=list(group_cols),
        how="inner",
        suffixes=("", "_derived"),
    )

    # --- optional last location
    if include_last_location and Con.LAST_VISITED_LOCATION in merged.columns:
        from src.predictive_modeling.common.features import map_last_location_to_position

        merged["last_loc_numeric"] = (
            map_last_location_to_position(merged[Con.LAST_VISITED_LOCATION])
            .fillna(-1)
            .astype(int)
        )

    return merged
