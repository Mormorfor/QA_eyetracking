from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con
from src.viz.plot_output import save_plot

# ---------------------------------------------------------------------------
# Base Statistics Bar-charts + Mixed Models
# ---------------------------------------------------------------------------

def plot_area_ci_bar(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols = (Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN),
    area_col: str = Con.AREA_LABEL_COLUMN,
    figsize=(8, 5),
    save: bool = False,
    paper_dirs = None,
    h_or_g: str = "hunters",
    selected: str = "A",
    title: Optional[str] = None,
):
    """
    Plot mean ± 95% CI of a metric by area (answer_A/B/C/D).

    If save=True, always saves to:
        reports/plots/basic_stats_barcharts/<stat_col>/<h_or_g>__<selected>.png

    If paper_dirs is a list, also mirrors to:
        <paper_dir>/basic_stats_barcharts/<stat_col>/<h_or_g>__<selected>.png
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
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"{stat_col}: mean ± 95% CI by {area_col}\n"
            f"Selected answer = {selected}"
        )
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
        save_plot(
            fig=fig,
            rel_dir=f"basic_stats_barcharts/{stat_col}",
            filename=f"{h_or_g}__{selected}",
            ext="png",
            dpi=300,
            paper_dirs=paper_dirs,
        )

    return fig, summary_df_basic




def run_all_area_barplots(
        hunters: pd.DataFrame,
        gatherers: pd.DataFrame,
        metrics=None,
        save_plots: bool = True,
        paper_dirs = None,
        print_summaries: bool = False,
):
    """
    For each metric and each selected answer label (A–D),
    for hunters, gatherers, and all participants combined,
    create area-level barplots (mean ± 95% CI).

    """
    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS_MODELING

    def _run_for_group(df: pd.DataFrame, group_name: str) -> dict:
        df = df.copy()
        group_results = {}

        for metric in metrics:
            metric_results = {}

            available_labels = [
                lab for lab in ["A", "B", "C", "D"]
                if lab in df[Con.SELECTED_ANSWER_LABEL_COLUMN].unique()
            ]

            if print_summaries:
                print(f"\n=== {group_name.upper()} — metric: {metric} ===")

            for ans in available_labels:
                subset = df[df[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans].copy()
                if subset.empty:
                    continue

                fig, summary = plot_area_ci_bar(
                    subset,
                    stat_col=metric,
                    h_or_g=group_name,
                    selected=ans,
                    save=save_plots,
                    paper_dirs=paper_dirs,
                )

                if print_summaries:
                    print(f"\n--- {group_name.upper()}, selected = {ans} ---")
                    print(summary)

                metric_results[ans] = {"fig": fig, "summary": summary}

            group_results[metric] = metric_results

        return group_results

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    results = {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all participants"),
    }

    return results

