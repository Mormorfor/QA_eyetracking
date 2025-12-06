import os
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con
from src.viz.visualisations_strategies import build_strategy_dataframe



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
    if eye_col not in df.columns:
        raise KeyError(f"Column '{eye_col}' not found in df.")

    # 1) Per-trial strategies from simplified sequences
    df_strat = build_strategy_dataframe(
        df,
        kind=kind,
        window_len=window_len,
        drop_question=drop_question,
        strat_col=strat_col,
    )

    # 2) Unique mapping: participant -> eye
    eye_map = (
        df[[Con.PARTICIPANT_ID, eye_col]]
        .dropna(subset=[Con.PARTICIPANT_ID])
        .groupby(Con.PARTICIPANT_ID)[eye_col]
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else np.nan)
    )

    df_strat = df_strat.copy()
    df_strat[eye_col] = df_strat[Con.PARTICIPANT_ID].map(eye_map)

    # 3) Count strategies per (participant, eye, strategy)
    counts = (
        df_strat
        .groupby([Con.PARTICIPANT_ID, eye_col, strat_col])
        .size()
        .reset_index(name="n_trials")
    )

    # 4) Total trials per (participant, eye)
    totals = (
        counts
        .groupby([Con.PARTICIPANT_ID, eye_col])["n_trials"]
        .sum()
        .rename("n_total")
        .reset_index()
    )

    merged = counts.merge(
        totals,
        on=[Con.PARTICIPANT_ID, eye_col],
        how="left",
    )
    merged["dominant_prop"] = merged["n_trials"] / merged["n_total"]

    # 5) Keep only most frequent strategy per participant+eye
    dominant_rows = (
        merged
        .sort_values("dominant_prop", ascending=False)
        .groupby([Con.PARTICIPANT_ID, eye_col])
        .head(1)
        .reset_index(drop=True)
    )
    dominant_rows = dominant_rows.rename(
        columns={strat_col: "dominant_strategy"}
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
    output_root: str = "../reports/plots/dominant_eye",
    save: bool = True,
    min_count: int = 1,
):
    """
    Produce separate horizontal barplots for each eye group (e.g. Left, Right),
    sorted from most to least frequent dominant strategy.
    """

    os.makedirs(output_root, exist_ok=True)

    # string representation for plotting
    def _strat_to_str(s):
        if isinstance(s, (list, tuple)):
            return " → ".join(map(str, s))
        return str(s)

    df = dom_df.copy()
    df["strategy_str"] = df[strat_col].apply(_strat_to_str)

    # Optionally collapse very rare strategies into OTHER
    strat_counts_all = df["strategy_str"].value_counts()
    rare_strats = strat_counts_all[strat_counts_all < min_count].index
    df.loc[df["strategy_str"].isin(rare_strats), "strategy_str"] = "OTHER"

    crosstab = pd.crosstab(df["strategy_str"], df[eye_col])

    for eye in df[eye_col].dropna().unique():
        df_eye = df[df[eye_col] == eye]
        freq = df_eye["strategy_str"].value_counts().sort_values(ascending=False)

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
        if save:
            filename = f"dominant_strategies_sorted_{group_name}_{eye}.png"
            fig.savefig(os.path.join(output_root, filename), dpi=300)

        plt.show()

    return crosstab


def run_dominant_strategy_eye_analysis(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    kind: str = "location",
    window_len: int = 4,
    drop_question: bool = True,
    threshold: float = 0.0,  # min dominant_prop to include participants
    output_root: str = "../reports/plots/dominant_eye",
    save: bool = True,
):
    """
    For hunters and gatherers:
      - compute dominant strategy per participant (first `window_len` visits)
      - filter participants by dominant_prop if threshold > 0
      - produce sorted per-eye barplots
      - return dominant tables + crosstabs
    """
    results = {}

    for group_name, df in {"hunters": hunters, "gatherers": gatherers}.items():
        dom_df = build_dominant_strategy_by_eye(
            df,
            kind=kind,
            window_len=window_len,
            drop_question=drop_question,
            strat_col=Con.STRATEGY_COL,
            eye_col=Con.DOMINANT_EYE_COLUMN,
        )

        if threshold > 0.0:
            dom_df = dom_df[dom_df["dominant_prop"] >= threshold].copy()

        crosstab = plot_dominant_strategies_by_eye_sorted(
            dom_df,
            eye_col=Con.DOMINANT_EYE_COLUMN,
            strat_col="dominant_strategy",
            group_name=group_name,
            output_root=output_root,
            save=save,
        )

        results[group_name] = {
            "dominant_df": dom_df,
            "crosstab": crosstab,
        }

    return results



