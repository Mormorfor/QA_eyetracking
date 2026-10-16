"""Each participant's dominant opening scan, set against their dominant eye.

For every participant: their most common opening scan (by default, their first
four answer visits), how often they used it, and their dominant eye
(`EYE_TRACKED`, left or right: the tracker records the dominant eye). Draws one bar chart per eye,
counting how many participants have each dominant scan, and saves a scan-by-eye
count table.

Descriptive only: no significance test is run, and the paper does not report it.
The dominant scans come from `features/strategies.py`, like every other
dominant-strategy number in the project.
"""

from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import columns as Con
from src.features.strategies import (
    attach_dominant_eye,
    build_starting_strategies,
    dominant_strategy_by_eye_crosstab,
    dominant_strategy_by_participant,
    has_dominant_strategy,
)
from src.lib.plotting.output import save_output
from src.features.scope import split_participant_groups



def build_dominant_strategy_by_eye(
    df: pd.DataFrame,
    kind: str = "location",          # "location" or "label"
    window_len: int = 4,
    drop_question: bool = True,
    strat_col: str = Con.STRATEGY_COL,
    eye_col: str = Con.DOMINANT_EYE_COLUMN,
) -> pd.DataFrame:
    """
    Each participant's dominant opening scan, how often they used it, and their
    dominant eye.

    The scan is the first `window_len` entries of the merged sequence, by place
    (`kind="location"`) or by meaning (`kind="label"`), with the question's
    entries removed first if `drop_question` is set. It is worked out over
    whatever trials are in `df`; `n_total` says how many that is for each
    person.

    Returns one row per participant:
        [participant_id, eye_col, dominant_strategy, n_trials, n_total, dominant_prop]
    where `n_trials` of the person's `n_total` trials use the dominant scan, and
    `dominant_prop` is that share (the dominance score).
    """
    df_strat = build_starting_strategies(
        df,
        kind=kind,
        window_len=window_len,
        drop_question=drop_question,
        out_col=strat_col,
    )

    dominant_rows = dominant_strategy_by_participant(
        df_strat, id_col=Con.PARTICIPANT_ID, strat_col=strat_col
    ).rename(
        columns={
            Con.DOMINANT_STARTING_STRATEGY: "dominant_strategy",
            Con.DOMINANCE_SCORE: "dominant_prop",
            Con.N_STRATEGY_TRIALS: "n_total",
        }
    )

    # Trials on the dominant strategy, recovered from the score.
    dominant_rows["n_trials"] = (
        (dominant_rows["dominant_prop"] * dominant_rows["n_total"])
        .round()
        .astype(int)
    )
    dominant_rows = attach_dominant_eye(
        dominant_rows, df, id_col=Con.PARTICIPANT_ID, eye_col=eye_col
    )

    return dominant_rows[
        [
            Con.PARTICIPANT_ID,
            eye_col,
            "dominant_strategy",
            "n_trials",
            "n_total",
            "dominant_prop",
        ]
    ]


def plot_dominant_strategies_by_eye_sorted(
    dom_df: pd.DataFrame,
    eye_col: str = Con.DOMINANT_EYE_COLUMN,
    strat_col: str = "dominant_strategy",
    group_name: str = "hunters",
    save: Optional[bool] = None,
    min_count: int = 1,
    to_paper=None,
):
    """
    One horizontal bar chart per dominant eye: how many participants have each
    dominant scan, most common at the top.

    `dom_df` is the table from `build_dominant_strategy_by_eye`. With
    `min_count` above 1, scans held by fewer participants are pooled as
    "OTHER". `group_name` goes into the titles and file names.

    Saves each chart with its counts, and the whole scan-by-eye count table on
    its own, under `reports/scan_strategies/`. Returns that count table.
    """

    crosstab = dominant_strategy_by_eye_crosstab(
        dom_df, eye_col=eye_col, strat_col=strat_col, min_count=min_count
    )

    # Each eye's chart is that eye's column of the count table.
    for eye in crosstab.columns:
        freq = crosstab[eye]
        freq = freq[freq > 0].sort_values(ascending=False, kind="stable")

        fig, ax = plt.subplots(figsize=(10, 6))
        freq.plot(kind="barh", ax=ax)
        ax.invert_yaxis()  # most frequent at top
        ax.set_xlabel("Number of participants")
        ax.set_ylabel("Dominant strategy (first 4 visits)")
        ax.set_title(
            f"Dominant strategies by eye dominance ({eye}-eye, {group_name})"
        )

        for i, v in enumerate(freq.values):
            ax.text(v + 0.2, i, str(v), va="center")

        fig.tight_layout()

        save_output(
            fig,
            analysis="scan_strategies",
            plot="dominant_strategies_by_eye",
            tables={"counts": freq.rename("n_participants").reset_index()},
            save=save,
            to_paper=to_paper,
            group=group_name,
            eye=eye,
        )

        plt.show()

    save_output(
        None,
        analysis="scan_strategies",
        plot="dominant_strategy_by_eye_crosstab",
        tables={"crosstab": crosstab.reset_index()},
        save=save,
        to_paper=to_paper,
        group=group_name,
    )

    return crosstab


def run_dominant_strategy_eye_analysis(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    kind: str = "location",
    window_len: int = 4,
    drop_question: bool = True,
    threshold: float = 0.0,  # min dominant_prop to include participants
    save: Optional[bool] = None,
    to_paper=None,
):
    """
    Run the analysis for hunters, gatherers and all participants together
    (only the last with `split_groups=False`).

    For each group: build the per-participant table, keep only participants
    whose dominance score is at least `threshold` (when it is above 0; by
    default everyone is kept), then draw and save the per-eye charts.

    Returns `{group: {"dominant_df": ..., "crosstab": ...}}`.
    """
    results = {}

    groups = split_participant_groups(all_participants, split=split_groups)

    for group_key, df in groups.items():
        # Human label for titles/filenames
        group_label = "all participants" if group_key == "all_participants" else group_key

        dom_df = build_dominant_strategy_by_eye(
            df,
            kind=kind,
            window_len=window_len,
            drop_question=drop_question,
            strat_col=Con.STRATEGY_COL,
            eye_col=Con.DOMINANT_EYE_COLUMN,
        )

        if threshold > 0.0:
            dom_df = dom_df[
                has_dominant_strategy(dom_df["dominant_prop"], threshold=threshold)
            ].copy()

        crosstab = plot_dominant_strategies_by_eye_sorted(
            dom_df,
            eye_col=Con.DOMINANT_EYE_COLUMN,
            strat_col="dominant_strategy",
            group_name=group_label,      # pretty label in plot title + filename
            save=save,
            to_paper=to_paper,
        )

        results[group_key] = {
            "dominant_df": dom_df,
            "crosstab": crosstab,
        }

    return results




