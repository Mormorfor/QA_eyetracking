# answer_correctness_data.py

from typing import Sequence, Optional, Tuple
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



def build_trial_level_with_area_metrics(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS_MODELING,
) -> pd.DataFrame:

    core_cols = list(group_cols) + [Con.IS_CORRECT_COLUMN]
    d = df[list(dict.fromkeys(core_cols))].copy()
    d[Con.IS_CORRECT_COLUMN] = d[Con.IS_CORRECT_COLUMN].astype(int)

    trial_core = (
        df[core_cols]
        .drop_duplicates()
        .dropna(subset=[Con.IS_CORRECT_COLUMN])
        .reset_index(drop=True)
    )
    metrics_pivot = build_area_metric_pivot(
        df=df,
        group_cols=group_cols,
        area_col=area_col,
        metric_cols=metric_cols,
    )
    trial_df = trial_core.merge(metrics_pivot, on=list(group_cols), how="left")
    return trial_df



Direction = str  # "high" | "low"
PrefSpec = Tuple[str, Direction]  # (metric_col, direction)

def _safe_pref_feature_name(metric_col: str, direction: str) -> str:
    return f"pref_matching__{metric_col}__{direction}"


def build_trial_level_with_derived_features_for_correctness(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    dwell_col: str = Con.MEAN_DWELL_TIME,
    pref_specs: Optional[Sequence[PrefSpec]] = None,
    pref_extreme_mode: str = "polarity",
) -> pd.DataFrame:
    """
    One row per trial (group_cols), containing:
      - is_correct
      - seq_len
      - has_xyx
      - has_xyxy
      - trial_mean_dwell
      - pref_matching__<metric>__<direction> (0/1) for each pref spec

    pref_specs example:
      [
        (Con.MEAN_DWELL_TIME, "high"),
        (Con.SKIP_RATE, "low"),
      ]
    """
    required = [*group_cols, Con.IS_CORRECT_COLUMN, seq_col, dwell_col]

    d = df[list(dict.fromkeys(required))].copy()
    d[Con.IS_CORRECT_COLUMN] = d[Con.IS_CORRECT_COLUMN].astype(int)

    d["_seq_len"] = d[seq_col].apply(sequence_len_literal_eval)

    d["_seq"] = d[seq_col].apply(parse_seq)
    d["_has_xyx"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyx(s)) if s is not None else False)
    d["_has_xyxy"] = d["_seq"].apply(lambda s: bool(has_back_and_forth_xyxy(s)) if s is not None else False)

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

    pref_specs = list(pref_specs) if pref_specs else []
    for metric_col, direction in pref_specs:

        pref_df = compute_trial_matching(
            df=df,
            metric_col=metric_col,
            direction=direction,
            extreme_mode=pref_extreme_mode,
            out_col="pref_group",
        )

        feat_col = _safe_pref_feature_name(metric_col, direction)

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
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
) -> pd.DataFrame:
    """
    Build trial-level table with all features for answer correctness prediction (both area level and derived).
    """

    area_df = build_trial_level_with_area_metrics(
        df=df,
        group_cols=group_cols,
    )

    derived_df = build_trial_level_with_derived_features_for_correctness(
        df=df,
        group_cols=group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
    )

    merged = area_df.merge(
        derived_df,
        on=list(group_cols),
        how="inner",
        suffixes=("", "_derived"),
    )

    if Con.IS_CORRECT_COLUMN not in merged.columns:
        candidates = []
        for c in [f"{Con.IS_CORRECT_COLUMN}_x", f"{Con.IS_CORRECT_COLUMN}_y",
                  f"{Con.IS_CORRECT_COLUMN}_derived"]:
            if c in merged.columns:
                candidates.append(c)

        if not candidates:
            raise KeyError(
                f"After merging, no '{Con.IS_CORRECT_COLUMN}' column found. "
                f"Columns present: {list(merged.columns)[:30]}..."
            )

        chosen = candidates[0]
        merged[Con.IS_CORRECT_COLUMN] = merged[chosen].astype(int)

    c1 = Con.IS_CORRECT_COLUMN
    c2 = f"{Con.IS_CORRECT_COLUMN}_derived"
    if c1 in merged.columns and c2 in merged.columns:
        if (merged[c1].astype(int) != merged[c2].astype(int)).any():
            bad = merged.loc[merged[c1].astype(int) != merged[c2].astype(int), list(group_cols) + [c1, c2]].head(10)
            raise ValueError(f"is_correct mismatch between area vs derived tables. Examples:\n{bad}")

        merged = merged.drop(columns=[c2])

    for col in [f"{Con.IS_CORRECT_COLUMN}_x", f"{Con.IS_CORRECT_COLUMN}_y"]:
        if col in merged.columns and col != Con.IS_CORRECT_COLUMN:
            merged = merged.drop(columns=[col])


    return merged

