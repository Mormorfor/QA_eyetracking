"""Did the participant choose the answer their eyes "preferred"?

For one per-area measurement, find the answer that stands out most on it --
e.g. the one looked at longest -- and label each trial "matching" if that is
the answer the participant chose, otherwise "not_matching".
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.config import columns as C


Direction = Literal["high", "low"]
ExtremeMode = Literal["polarity", "relative"]


def compute_trial_matching(
    df: pd.DataFrame,
    metric_col: str,
    direction: Direction = "high",
    extreme_mode: ExtremeMode = "polarity",
    out_col: str = "pref_group",
) -> pd.DataFrame:
    """One row per trial: the "preferred" answer, and whether it was chosen.

    The preferred answer (`pred_label`) is picked one of two ways:

      extreme_mode="polarity"  the highest value (direction="high", e.g. dwell
                               time) or the lowest (direction="low", e.g. skip
                               rate) among A-D
      extreme_mode="relative"  the answer furthest from the trial's average,
                               in either direction (`direction` is ignored)

    `out_col` is "matching" if the chosen answer is the preferred one, else
    "not_matching". Also returns the four answers' values, `n_preferred`, and
    `is_correct` if the frame has it.

    **A tie is not broken.** Every answer tied for the extreme is preferred:
    `pred_label` lists them ("A/C"), `n_preferred` counts them, and the trial is
    "matching" if the chosen answer is among them. Breaking the tie by letter
    would leak the answer key, since answer A is always the correct one: on a
    tied trial "matching" would then mean exactly "correct". Ties are common on
    the counted measures (on Study 1, `num_label_visits` ties at its top on 45%
    of trials, `skip_rate` at its bottom on 18%).

    Answers with no value are skipped. A trial where none of the four has a
    value has no preferred answer: its `pred_label` and `out_col` are missing,
    and the summaries and the test leave it out.
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
            **(
                {C.IS_CORRECT_COLUMN: (C.IS_CORRECT_COLUMN, "first")}
                if has_correct
                else {}
            ),
        )
    )

    answer_areas = [f"{C.ANSWER_PREFIX}{lab}" for lab in C.ANSWER_LABELS]
    area = area[area[C.AREA_LABEL_COLUMN].isin(answer_areas)].copy()

    area["answer_label"] = area[C.AREA_LABEL_COLUMN].str.replace(
        C.ANSWER_PREFIX, "", regex=False
    )

    # `pivot`, not `pivot_table`: pivot_table drops a trial whose four values are
    # all missing, and such a trial is kept here, with no preferred answer.
    wide = area.pivot(
        index=[C.TRIAL_ID, C.PARTICIPANT_ID],
        columns="answer_label",
        values="metric_value",
    ).reset_index()
    wide.columns.name = None

    for lab in C.ANSWER_LABELS:
        if lab not in wide.columns:
            wide[lab] = np.nan

    M = wide[C.ANSWER_LABELS].to_numpy(dtype=float)
    missing = np.isnan(M)

    # How far each answer stands out; the preferred answers are those with the top
    # score. An answer with no value cannot be preferred.
    if extreme_mode == "polarity":
        score = M if direction == "high" else -M
    elif extreme_mode == "relative":
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(missing, 0.0, M).sum(axis=1) / (~missing).sum(axis=1)
        score = np.abs(M - mu[:, None])
    else:
        raise ValueError("extreme_mode must be 'polarity' or 'relative'")
    score = np.where(missing, -np.inf, score)
    preferred = (score == score.max(axis=1)[:, None]) & ~missing

    labels = np.array(C.ANSWER_LABELS, dtype=object)
    wide["pred_label"] = [
        "/".join(labels[row]) if row.any() else np.nan for row in preferred
    ]
    wide["n_preferred"] = preferred.sum(axis=1)

    meta_cols = [C.TRIAL_ID, C.PARTICIPANT_ID, "selected_label"]
    if has_correct:
        meta_cols.append(C.IS_CORRECT_COLUMN)

    meta = area[meta_cols].drop_duplicates(subset=[C.TRIAL_ID, C.PARTICIPANT_ID])

    out = wide.merge(meta, on=[C.TRIAL_ID, C.PARTICIPANT_ID], how="left")
    # A left merge keeps `wide`'s rows in order, so `preferred` still lines up.
    assert len(out) == len(wide), "a trial has more than one selected answer"

    chosen = out["selected_label"].map({lab: i for i, lab in enumerate(C.ANSWER_LABELS)})
    assert chosen.notna().all(), (
        f"selected answers outside {C.ANSWER_LABELS}: "
        f"{sorted(out.loc[chosen.isna(), 'selected_label'].unique())}"
    )
    is_preferred = preferred[np.arange(len(out)), chosen.to_numpy(int)]
    out[out_col] = np.where(is_preferred, "matching", "not_matching").astype(object)
    out.loc[~preferred.any(axis=1), out_col] = np.nan

    core = [C.TRIAL_ID, C.PARTICIPANT_ID, "selected_label", "pred_label", "n_preferred", out_col]
    if has_correct:
        core.append(C.IS_CORRECT_COLUMN)

    return out[core + C.ANSWER_LABELS]
