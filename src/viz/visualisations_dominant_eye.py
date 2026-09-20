from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con
from src.derived.pattern_breaking import (
    attach_dominant_eye,
    build_starting_strategies,
    dominant_strategy_by_eye_crosstab,
    dominant_strategy_by_participant,
    has_dominant_strategy,
)
from src.viz.plot_output import save_output
from src.viz.viz_helpers import split_participant_groups



def build_dominant_strategy_by_eye(
    df: pd.DataFrame,
    kind: str = "location",          # "location" or "label"
    window_len: int = 4,
    drop_question: bool = True,
    strat_col: str = Con.STRATEGY_COL,
    eye_col: str = Con.DOMINANT_EYE_COLUMN,
) -> pd.DataFrame:
    """
    For each participant, compute their dominant strategy (most common
    first `window_len` visits) and how often it occurs, and attach
    their dominant eye (EYE_TRACKED).

    Returns one row per participant:
        [participant_id, eye_col, dominant_strategy, n_trials, n_total, dominant_prop]
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
    Produce separate horizontal barplots for each eye group (e.g. Left, Right),
    sorted from most to least frequent dominant strategy.
    """

    crosstab = dominant_strategy_by_eye_crosstab(
        dom_df, eye_col=eye_col, strat_col=strat_col, min_count=min_count
    )

    # The per-eye bars ARE the crosstab's columns -- no second tally needed.
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

        # annotate counts
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
    For hunters, gatherers, and all participants:
      - compute dominant strategy per participant (first `window_len` visits)
      - filter participants by dominant_prop if threshold > 0
      - produce sorted per-eye barplots
      - return dominant tables + crosstabs
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




