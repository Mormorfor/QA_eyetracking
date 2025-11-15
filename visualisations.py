import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re

from typing import Optional

import constants as Con

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pymer4.models import Lmer


# ---------------------------------------------------------------------------
#  Base Statistics Heatmaps
# ---------------------------------------------------------------------------

def matrix_plot_ABCD(
    df: pd.DataFrame,
    stat: str,
    selected: str = "A",
    h_or_g: str = "hunters",
    drop_questions: bool = True,
    output_root: str = "plots/basic_stats_heatmaps",
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Draw a heatmap of a metric by (area_label x area_screen_loc)
    for participants who selected a given answer label (A/B/C/D).

    Parameters
    ----------
    df : DataFrame
        Row-level data already filtered to the desired subset
        (e.g. only trials where selected_answer_label == 'A').
    stat : str
        Column name of the metric to visualize (e.g. 'mean_dwell_time')
        Should be selected from C.AREA_METRIC_COLUMNS
    selected : str, optional
        Which answer label was selected ('A', 'B', 'C', 'D').
    h_or_g : str, optional
        Tag for hunters/gatherers, used in the plot title and filename.
    drop_questions : bool, optional
        If True, exclude rows where AREA_LABEL_COLUMN == 'question'.
    output_root : str, optional
        Root directory where plots will be saved.
    show : bool, optional
        If True, display the plot.
    save : bool, optional
        If True, save the plot as a PNG file under output_root/stat/.
    """
    df = df[
        [Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.AREA_LABEL_COLUMN, Con.AREA_SCREEN_LOCATION, stat]
    ].drop_duplicates().copy()

    if drop_questions:
        df = df[df[Con.AREA_LABEL_COLUMN] != "question"]

    matrix = pd.pivot_table(
        data=df,
        index=Con.AREA_LABEL_COLUMN,
        columns=Con.AREA_SCREEN_LOCATION,
        values=stat,
        aggfunc="mean",
    )

    if drop_questions:
        label_order = [lbl for lbl in Con.ANSWER_LABEL_CHOICES if lbl != "question"]
    else:
        label_order = list(Con.ANSWER_LABEL_CHOICES)

    row_order = [lbl for lbl in label_order if lbl in matrix.index]
    col_order = [loc for loc in Con.AREA_LABEL_CHOICES if loc in matrix.columns]

    matrix = matrix.reindex(index=row_order, columns=col_order)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar_kws={"label": stat},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    title_suffix = " (questions removed)" if drop_questions else ""
    plt.title(f"{stat} of those who chose {selected}{title_suffix}")
    plt.xlabel(Con.AREA_SCREEN_LOCATION)
    plt.ylabel(Con.AREA_SCREEN_LOCATION)
    plt.tight_layout()

    if save:
        out_dir = os.path.join(output_root, stat)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{h_or_g} - {selected} selected - (questions removed - {drop_questions}).png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()



def label_vs_loc_mat(
    metric: str,
    dfh: pd.DataFrame,
    dfg: pd.DataFrame,
    drop_questions: bool = False,
    **plot_kwargs,
) -> None:
    """
    For a given metric, plot hunters/gatherers heatmaps
    for each selected answer label (A, B, C, D).

    Parameters
    ----------
    metric : str
        Column name of the metric to visualize.
    dfh : DataFrame
        Hunters DataFrame.
    dfg : DataFrame
        Gatherers DataFrame.
    drop_questions : bool, optional
        Whether to drop question areas in the plots.
    plot_kwargs : dict
        Additional keyword arguments passed to matrix_plot_ABCD
        (e.g. output_root, show, save).
    """
    print(f"HUNTERS (drop_questions={drop_questions})")
    for ans in ["A", "B", "C", "D"]:
        subset_h = dfh[dfh[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans]
        matrix_plot_ABCD(
            subset_h,
            metric,
            selected=ans,
            h_or_g="hunters",
            drop_questions=drop_questions,
            **plot_kwargs,
        )

    print(f"GATHERERS (drop_questions={drop_questions})")
    for ans in ["A", "B", "C", "D"]:
        subset_g = dfg[dfg[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans]
        matrix_plot_ABCD(
            subset_g,
            metric,
            selected=ans,
            h_or_g="gatherers",
            drop_questions=drop_questions,
            **plot_kwargs,
        )



def run_all_area_metric_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metrics=None,
    drop_question_variants=(False, True),
    output_root="plots/basic_stats_heatmaps",
    show=True,
    save=True,
):
    """
    Convenience wrapper: for every metric in metrics (or C.AREA_METRIC_COLUMNS),
    generate label-vs-location matrices for hunters & gatherers,
    with and without questions.

    Parameters
    ----------
    hunters : DataFrame
    gatherers : DataFrame
    metrics : list[str] or None
        If None, uses C.AREA_METRIC_COLUMNS.
    drop_question_variants : iterable of bool
        Which values of drop_questions to run (e.g. (False, True)).
    output_root : str
        Root directory for saving plots.
    show : bool
        Whether to show plots interactively.
    save : bool
        Whether to save plots to disk.
    """
    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS

    for metric in metrics:
        for dq in drop_question_variants:
            print(f"\n=== {metric} (drop_questions={dq}) ===")
            label_vs_loc_mat(
                metric,
                hunters,
                gatherers,
                drop_questions=dq,
                output_root=output_root,
                show=show,
                save=save,
            )


# ---------------------------------------------------------------------------
#  Base Statistics Bar-charts
# ---------------------------------------------------------------------------
#
#
# def plot_area_ci_bar(
#     df: pd.DataFrame,
#     stat_col: str = Con.AREA_METRIC_COLUMNS[0],
#     trial_cols: tuple[str, ...] = (Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN, Con.TRIAL_ID),
#     area_col: str = Con.AREA_LABEL_COLUMN,
#     figsize: tuple[int, int] = (8, 5),
#     h_or_g: str = "hunters",
#     selected: str = "A",
#     title: Optional[str] = None,
#     output_root: str = "plots/basic_stats_barcharts",
#     show: bool = True,
#     save: bool = True,
# ) -> tuple[plt.Figure, pd.DataFrame]:
#     """
#     Plot mean ± 95% CI of a metric per area (answer_A/B/C/D) as a bar chart.
#
#     Steps
#     -----
#     1. De-duplicate rows by (trial_cols + area_col) to avoid double-counting.
#     2. Order areas using ANSWER_LABEL_CHOICES (excluding 'question').
#     3. Draw a seaborn barplot with mean + 95% CI.
#     4. Return the figure and a summary table (mean, sd, n per area).
#
#     Parameters
#     ----------
#     df : DataFrame
#         Row-level data already filtered to the desired subset
#         (e.g. only trials where selected_answer_label == 'A').
#     stat_col : str
#         Name of the metric column to plot (e.g. 'mean_dwell_time').
#         Should be selected from C.AREA_METRIC_COLUMNS
#     trial_cols : tuple of str
#         Columns identifying a unique trial (participant, text, trial index).
#     area_col : str
#         Column containing logical area labels (e.g. 'answer_A', 'answer_B', ...).
#     figsize : (int, int)
#         Figure size passed to matplotlib.
#     h_or_g : str
#         'hunters' or 'gatherers' tag used in the saved filename.
#     selected : str
#         Selected answer label ('A', 'B', 'C', 'D').
#     title : str or None
#         Optional custom title; if None, a default title is constructed.
#     output_root : str
#         Root directory where plots will be saved.
#     show : bool
#         If True, show the plot.
#     save : bool
#         If True, save the plot as PNG to disk.
#
#     Returns
#     -------
#     (fig, summary_df_basic)
#         fig : matplotlib Figure
#         summary_df_basic : DataFrame with columns [area_col, mean, sd, n]
#     """
#
#     dedup = (
#         df[list(trial_cols) + [area_col, stat_col]]
#         .drop_duplicates(subset=list(trial_cols) + [area_col])
#         .dropna(subset=[stat_col])
#         .copy()
#     )
#
#     answer_areas = [
#         lbl for lbl in Con.ANSWER_LABEL_CHOICES
#         if lbl != "question" and lbl in dedup[area_col].unique()
#     ]
#
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.barplot(
#         data=dedup,
#         x=area_col,
#         y=stat_col,
#         order=answer_areas if answer_areas else None,
#         estimator=np.mean,
#         errorbar=("ci", 95),
#         capsize=0.1,
#         ax=ax,
#     )
#
#     ax.set_xlabel(area_col)
#     ax.set_ylabel(stat_col)
#
#     ax.set_title(title or f"{stat_col}: mean ± 95% CI by {area_col} (selected {selected})")
#     ax.margins(x=0.02)
#
#     summary_df_basic = (
#         dedup.groupby(area_col)[stat_col]
#         .agg(mean="mean", sd="std", n="count")
#         .reset_index()
#     )
#     if answer_areas:
#         summary_df_basic = (
#             summary_df_basic
#             .set_index(area_col)
#             .loc[answer_areas]
#             .reset_index()
#         )
#
#     if save:
#         out_dir = os.path.join(output_root, stat_col)
#         os.makedirs(out_dir, exist_ok=True)
#         fname = f"{h_or_g} - {selected}.png"
#         plt.savefig(os.path.join(out_dir, fname))
#
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)
#
#     return fig, summary_df_basic
#
#
#
# def mixed_area_analysis(
#     df: pd.DataFrame,
#     stat_col: str = Con.AREA_METRIC_COLUMNS[0],
#     trial_cols: tuple[str, ...] = (Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN, Con.TRIAL_ID),
#     area_col: str = Con.AREA_LABEL_COLUMN,
#     alpha: float = 0.05,
# ):
#     """
#     Run a linear mixed-effects model (pymer4::Lmer) on an area-level metric and
#     compute pairwise differences between areas (answer_A/B/C/D) with Holm-adjusted p-values.
#
#     Model (lme4-style formula)
#     --------------------------
#         stat_col ~ 0 + area_col + (1|participant_id) + (1|text_id)
#
#     i.e.:
#       - one fixed-effect coefficient per area (no global intercept)
#       - random intercept for participant_id
#       - random intercept for text_id
#
#     Steps
#     -----
#     1. De-duplicate metric values.
#     2. Set area_col as an ordered categorical (answer_A/B/C/D if present).
#     3. Fit an Lmer model.
#     4. Extract fixed-effect estimates + CIs.
#     5. Use Lmer.post_hoc() to get pairwise contrasts and then add Holm-adjusted p.
#
#     Returns
#     -------
#     (model, fe_table, pairwise)
#         model    : pymer4.models.Lmer (already fit)
#         fe_table : DataFrame with columns
#                    [area, estimate, ci_low, ci_high]
#         pairwise : DataFrame with at least:
#                    [contrast, estimate, SE, DF, T, p, p_adj_holm, ci_low, ci_high]
#     """
#
#     dedup = (
#         df[list(trial_cols) + [area_col, stat_col]]
#         .drop_duplicates()
#         .copy()
#     )
#
#     area_order = [
#         lbl for lbl in Con.ANSWER_LABEL_CHOICES
#         if lbl != "question" and lbl in dedup[area_col].unique()
#     ]
#     if not area_order:
#         area_order = sorted(dedup[area_col].unique())
#
#     dedup[area_col] = pd.Categorical(
#         dedup[area_col],
#         categories=area_order,
#         ordered=True,
#     )
#
#
#     #    Example: "mean_dwell_time ~ 0 + area_label + (1|participant_id) + (1|text_id)"
#     formula = (
#         f"{stat_col} ~ 0 + {area_col} "
#         f"+ (1|{Con.PARTICIPANT_ID}) "
#         f"+ (1|{Con.TEXT_ID_COLUMN})"
#     )
#
#     model = Lmer(formula, data=dedup)
#     fit_res = model.fit(factors={area_col: area_order})
#
#     coefs = model.coefs.copy()
#     fe_table = (
#         coefs
#         .reset_index()
#         .rename(columns={"index": "term"})
#     )
#
#     # Helper: extract area level from term name
#     # pymer4 usually uses something like:
#     #   term = f"{area_col}{level}"  e.g. "area_labelanswer_A"
#     # or "area_label:answer_A" depending on version
#     def extract_area(term: str) -> str:
#         prefix = f"{area_col}"
#         if term.startswith(prefix):
#             lvl = term[len(prefix):]
#             # strip a leading ':' if present: "area_label:answer_A"
#             if lvl.startswith(":"):
#                 lvl = lvl[1:]
#             return lvl or None
#
#         # fallback: look for bracketed forms like "[answer_A]" if they ever occur
#         m = re.search(r"\[(?:T\.)?([^\]]+)\]", term)
#         if m:
#             return m.group(1)
#         return None
#
#     fe_table["area"] = fe_table["term"].apply(extract_area)
#
#     # Keep only rows that correspond to our area levels
#     fe_table = fe_table[fe_table["area"].isin(area_order)].copy()
#
#     # Map pymer4 column names to our neutral names
#     col_map = {
#         "Estimate": "estimate",
#         "2.5_ci": "ci_low",
#         "97.5_ci": "ci_high",
#     }
#     for old, new in col_map.items():
#         if old in fe_table.columns:
#             fe_table[new] = fe_table[old]
#
#     fe_table = (
#         fe_table[["area", "estimate", "ci_low", "ci_high"]]
#         .set_index("area")
#         .loc[area_order]  # <— now this should work
#         .reset_index()
#     )
#
#     ph = model.post_hoc(marginal_vars=area_col)
#     pairwise = ph.copy()
#
#     rename_cols = {
#         "Estimate": "estimate",
#         "SE": "se",
#         "T-stat": "t",
#         "p": "p_unc",
#         "lower": "ci_low",
#         "upper": "ci_high",
#     }
#     pairwise.rename(columns={k: v for k, v in rename_cols.items() if k in pairwise.columns},
#                     inplace=True)
#
#     if "p_unc" in pairwise.columns:
#         pairwise["p_adj_holm"] = multipletests(pairwise["p_unc"], method="holm")[1]
#         pairwise["sig"] = np.where(pairwise["p_adj_holm"] < alpha, "★", "")
#     else:
#         pairwise["p_adj_holm"] = np.nan
#         pairwise["sig"] = ""
#
#     return model, fe_table, pairwise











