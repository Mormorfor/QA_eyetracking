import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con


# ---------------------------------------------------------------------------
# Base Statistics Bar-charts + Mixed Models
# ---------------------------------------------------------------------------

def plot_area_ci_bar(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols=(Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN),
    area_col: str = Con.AREA_LABEL_COLUMN,
    figsize=(8, 5),
    save: bool = False,
    h_or_g: str = "hunters",
    selected: str = "A",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/basic_stats_barcharts",
):
    """
    Plot mean ± 95% CI of a metric by area (answer_A/B/C/D).

    Parameters
    ----------
    df : DataFrame
        Row-level data (one row per IA) already filtered
        to a subset (e.g. only selected_answer_label == 'A').
    stat_col : str
        Column with the metric to plot (e.g. Con.MEAN_DWELL_TIME).
    trial_cols : tuple[str]
        Columns that define a unique trial-level observation.
        Defaults match constants: TRIAL_ID, PARTICIPANT_ID, TEXT_ID_COLUMN.
    area_col : str
        Area column to plot on the x-axis (typically Con.AREA_LABEL_COLUMN).
    figsize : tuple
        Figure size passed to matplotlib.
    save : bool
        If True, save PNG under output_root / stat_col / ...
    h_or_g : {"hunters","gatherers"}
        Tag used in filenames.
    selected : {"A","B","C","D"}
        Which answer label this subset represents (for filenames/titles).
    title : str or None
        Custom title; if None, a default is used.
    output_root : str
        Root directory for saving plots.
    """

    dedup = (
        df[list(trial_cols) + [area_col, stat_col]]
        .drop_duplicates(subset=list(trial_cols) + [area_col])
    )

    area_order = [
        a for a in ["answer_A", "answer_B", "answer_C", "answer_D"]
        if a in dedup[area_col].unique()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=dedup,
        x=area_col,
        y=stat_col,
        order=area_order if area_order else None,
        estimator=np.mean,
        errorbar=("ci", 95),
        capsize=0.1,
        ax=ax,
    )

    ax.set_xlabel(area_col)
    ax.set_ylabel(stat_col)
    ax.set_title(title or f"{stat_col}: mean ± 95% CI by {area_col}")
    ax.margins(x=0.02)

    summary_df_basic = (
        dedup.groupby(area_col)[stat_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    if area_order:
        summary_df_basic = (
            summary_df_basic
            .set_index(area_col)
            .loc[area_order]
            .reset_index()
        )

    if save:
        out_dir = os.path.join(output_root, stat_col)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{h_or_g}__{selected}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)

    return fig, summary_df_basic




def run_all_area_barplots(
        hunters: pd.DataFrame,
        gatherers: pd.DataFrame,
        metrics=None,
        output_root: str = "../reports/plots/basic_stats_barcharts",
        save_plots: bool = True,
        print_summaries: bool = False,
):
    """
    For each metric and each selected answer label (A–D), create
    area-level barplots (mean ± 95% CI).

    No mixed models, no statistics beyond descriptive summaries.

    Parameters
    ----------
    hunters, gatherers : DataFrame
        Row-level data.
    metrics : list[str] or None
        Which metric columns to plot. If None: Con.AREA_METRIC_COLUMNS.
    output_root : str
        Where to save plots. Each metric gets its own subfolder.
    save_plots : bool
        Save PNGs using plot_area_ci_bar().
    print_summaries : bool
        Print the descriptive table (mean/sd/n).

    Returns
    -------
    results : dict
        results[group][metric][label] = {
            "fig": figure,
            "summary": DataFrame
        }
    """

    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS

    def _run_for_group(df: pd.DataFrame, group_name: str) -> dict:
        df_noq = df[df[Con.AREA_LABEL_COLUMN] != "question"].copy()
        group_results = {}

        for metric in metrics:
            metric_results = {}

            available_labels = [
                lab for lab in ["A", "B", "C", "D"]
                if lab in df_noq[Con.SELECTED_ANSWER_LABEL_COLUMN].unique()
            ]

            if print_summaries:
                print(f"\n=== {group_name.upper()} — metric: {metric} ===")

            for ans in available_labels:
                subset = df_noq[
                    df_noq[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans
                    ].copy()

                if subset.empty:
                    continue

                fig, summary = plot_area_ci_bar(
                    subset,
                    stat_col=metric,
                    h_or_g=group_name,
                    selected=ans,
                    save=save_plots,
                    output_root=output_root,
                )

                if print_summaries:
                    print(f"\n--- {group_name.upper()}, selected = {ans} ---")
                    print(summary)

                metric_results[ans] = {
                    "fig": fig,
                    "summary": summary,
                }

            group_results[metric] = metric_results

        return group_results

    results = {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
    }

    return results
