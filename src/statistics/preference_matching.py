from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src import constants as C


Direction = Literal["high", "low"]
ExtremeMode = Literal["polarity", "relative"]


def compute_trial_matching(
    df: pd.DataFrame,
    metric_col: str,
    direction: Direction = "high",
    extreme_mode: ExtremeMode = "polarity",
    out_col: str = "pref_group",
    keep_cols: [Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Trial-level matching label (one row per TRIAL_ID x PARTICIPANT_ID),
    while carrying along correctness and optionally other trial-level columns.

    predicted answer can be defined in two ways:

      1) extreme_mode="polarity"
         - direction="high": argmax(metric) among A–D
         - direction="low" : argmin(metric) among A–D

      2) extreme_mode="relative"
         - argmax(|metric - trial_mean(metric)|) among A–D
           (direction ignored)

    matching := selected_answer_label == pred_label

    Required columns:
      C.TRIAL_ID
      C.PARTICIPANT_ID
      C.AREA_LABEL_COLUMN
      C.SELECTED_ANSWER_LABEL_COLUMN
      metric_col

    If present, this function will also keep:
      - C.IS_CORRECT_COLUMN

    """

    required = [
        C.TRIAL_ID,
        C.PARTICIPANT_ID,
        C.AREA_LABEL_COLUMN,
        C.SELECTED_ANSWER_LABEL_COLUMN,
        metric_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if extreme_mode not in ("polarity", "relative"):
        raise ValueError("extreme_mode must be 'polarity' or 'relative'")
    if direction not in ("high", "low"):
        raise ValueError("direction must be 'high' or 'low'")

    has_correct = C.IS_CORRECT_COLUMN in df.columns

    cols = required.copy()
    if has_correct:
        cols.append(C.IS_CORRECT_COLUMN)

    area = (
        df[cols]
        .groupby([C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN], as_index=False)
        .agg(
            metric_value=(metric_col, "first"),
            selected_label=(C.SELECTED_ANSWER_LABEL_COLUMN, "first"),
            **({C.IS_CORRECT_COLUMN: (C.IS_CORRECT_COLUMN, "first")} if has_correct else {}),
        )
    )

    answer_areas = [f"{C.ANSWER_PREFIX}{lab}" for lab in C.ANSWER_LABELS]
    area = area[area[C.AREA_LABEL_COLUMN].isin(answer_areas)].copy()

    area["answer_label"] = area[C.AREA_LABEL_COLUMN].str.replace(C.ANSWER_PREFIX, "", regex=False)

    wide = (
        area.pivot_table(
            index=[C.TRIAL_ID, C.PARTICIPANT_ID],
            columns="answer_label",
            values="metric_value",
            aggfunc="first",
        )
        .reset_index()
    )

    for lab in C.ANSWER_LABELS:
        if lab not in wide.columns:
            wide[lab] = np.nan

    M = wide[C.ANSWER_LABELS].to_numpy(dtype=float)

    if extreme_mode == "polarity":
        idx = np.nanargmax(M, axis=1) if direction == "high" else np.nanargmin(M, axis=1)
    elif extreme_mode == "relative":
        mu = np.nanmean(M, axis=1)
        dev = np.abs(M - mu[:, None])
        idx = np.nanargmax(dev, axis=1)
    else:
        raise ValueError("extreme_mode must be 'polarity' or 'relative'")

    wide["pred_label"] = np.array([C.ANSWER_LABELS[i] for i in idx], dtype=object)

    meta_cols = [C.TRIAL_ID, C.PARTICIPANT_ID, "selected_label"]
    if has_correct:
        meta_cols.append(C.IS_CORRECT_COLUMN)

    meta = area[meta_cols].drop_duplicates(subset=[C.TRIAL_ID, C.PARTICIPANT_ID])

    out = wide.merge(meta, on=[C.TRIAL_ID, C.PARTICIPANT_ID], how="left")

    out[out_col] = np.where(out["selected_label"] == out["pred_label"], "matching", "not_matching")

    core = [C.TRIAL_ID, C.PARTICIPANT_ID, "selected_label", "pred_label", out_col]
    if has_correct:
        core.append(C.IS_CORRECT_COLUMN)

    return out[core + C.ANSWER_LABELS]