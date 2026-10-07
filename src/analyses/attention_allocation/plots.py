"""Figures: where attention goes across the five screen areas.

Merged from three `viz/visualisations_*` modules in stage E. They were split by
figure type (bars, matrices, significance heatmaps); they are one question."""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import columns as Con
from src.lib.plotting.output import collect_tables, save_output
from src.features.scope import split_participant_groups


# ==========================================================================
# from src/viz/visualisations_area_bars.py
# ==========================================================================

# ---------------------------------------------------------------------------
# Base Statistics Bar-charts + Mixed Models
# ---------------------------------------------------------------------------


def plot_area_ci_bar(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols=(Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN),
    area_col: str = Con.AREA_LABEL_COLUMN,
    figsize=(8, 5),
    save: Optional[bool] = None,
    to_paper=None,
    h_or_g: str = "hunters",
    selected: str = "A",
    title: Optional[str] = None,
):
    """
    Plot mean ± 95% CI of a metric by area (answer_A/B/C/D).

    Saves through ``plot_output.save_output`` to
    ``reports/attention_allocation/{figures,tables}/<stat_col>/``.
    """

    dedup = df[list(trial_cols) + [area_col, stat_col]].drop_duplicates(
        subset=list(trial_cols) + [area_col]
    )

    area_order = [
        a
        for a in ["answer_A", "answer_B", "answer_C", "answer_D"]
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
            f"{stat_col}: mean ± 95% CI by {area_col}\n" f"Selected answer = {selected}"
        )
    ax.margins(x=0.02)

    summary_df_basic = (
        dedup.groupby(area_col)[stat_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    if area_order:
        summary_df_basic = (
            summary_df_basic.set_index(area_col).loc[area_order].reset_index()
        )

    save_output(
        fig,
        analysis="attention_allocation",
        plot="area_bars",
        tables={"summary": summary_df_basic},
        save=save,
        to_paper=to_paper,
        subdir=stat_col,
        group=h_or_g,
        metric=stat_col,
        selected=selected,
    )

    return fig, summary_df_basic


def run_all_area_barplots(
    all_participants: pd.DataFrame,
    metrics=None,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = True,
    split_groups: bool = True,
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
                lab
                for lab in ["A", "B", "C", "D"]
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
                    save=save,
                    to_paper=to_paper,
                )

                if print_summaries:
                    print(f"\n--- {group_name.upper()}, selected = {ans} ---")
                    print(summary)

                metric_results[ans] = {"fig": fig, "summary": summary}

            group_results[metric] = metric_results

        return group_results

    groups = split_participant_groups(all_participants, split=split_groups)

    # A sweep over group x metric x selected: up to 120 figures. Their summaries
    # pool into one long table rather than 120 small ones -- see
    # plot_output.collect_tables.
    with collect_tables("attention_allocation", plots=["area_bars"]):
        results = {name: _run_for_group(df, name) for name, df in groups.items()}

    return results


# ==========================================================================
# from src/viz/visualisations_area_matrices.py
# ==========================================================================

# ---------------------------------------------------------------------------
#  Base Statistics Heatmaps
# ---------------------------------------------------------------------------

def matrix_plot_ABCD(
    df: pd.DataFrame,
    stat: str,
    selected: str = "A",
    h_or_g: str = "hunters",
    drop_questions: bool = True,
    show: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
) -> pd.DataFrame:
    """
    Draw a heatmap of a metric by (area_label x area_screen_loc)
    for participants who selected a given answer label (A/B/C/D).

    Returns the pivoted matrix, which is also saved alongside the figure —
    before T1.3 this function returned ``None`` and the numbers behind all 320
    of these heatmaps existed only inside the PNGs.

    Parameters
    ----------
    df : DataFrame
        Row-level data already filtered to the desired subset
        (e.g. only trials where selected_answer_label == 'A').
    stat : str
        Column name of the metric to visualize (e.g. 'mean_dwell_time')
        Should be selected from C.AREA_METRIC_COLUMNS_VIZES
    selected : str, optional
        Which answer label was selected ('A', 'B', 'C', 'D').
    h_or_g : str, optional
        Tag for hunters/gatherers, used in the plot title and filename.
    drop_questions : bool, optional
        If True, exclude rows where AREA_LABEL_COLUMN == 'question'.
    show : bool, optional
        If True, display the plot.
    save : bool, optional
        If True, write the figure and its matrix through ``save_output``.
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
        label_order = [lbl for lbl in Con.LABEL_CHOICES if lbl != "question"]
    else:
        label_order = list(Con.LABEL_CHOICES)

    row_order = [lbl for lbl in label_order if lbl in matrix.index]
    col_order = [loc for loc in Con.LOC_CHOICES if loc in matrix.columns]

    matrix = matrix.reindex(index=row_order, columns=col_order)

    fig = plt.figure(figsize=(8, 6))
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
    # The y axis is indexed by area_label (the pivot's index), not by screen
    # location -- both axes were labelled area_screen_loc until 2026-09-20.
    plt.ylabel(Con.AREA_LABEL_COLUMN)
    plt.tight_layout()

    save_output(
        fig,
        analysis="attention_allocation",
        plot="area_label_by_loc_heatmap",
        tables={"matrix": matrix.reset_index()},
        save=save,
        to_paper=to_paper,
        subdir=stat,
        group=h_or_g,
        metric=stat,
        selected=selected,
        questions="removed" if drop_questions else "included",
    )

    if show:
        plt.show()
    else:
        plt.close()

    return matrix



def label_vs_loc_mat(
    metric: str,
    all_participants: pd.DataFrame,
    drop_questions: bool = False,
    split_groups: bool = True,
    **plot_kwargs,
) -> None:
    """
    For a given metric, plot heatmaps for each selected answer label (A-D).

    Splits the single ``all_participants`` frame on request and runs:
      - hunters
      - gatherers
      - all participants

    When ``split_groups=False`` only the combined all-participants group is run.
    """
    def _run(df: pd.DataFrame, group_label: str) -> None:
        print(f"{group_label.upper()} (drop_questions={drop_questions})")
        for ans in ["A", "B", "C", "D"]:
            subset = df[df[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans]
            matrix_plot_ABCD(
                subset,
                metric,
                selected=ans,
                h_or_g=group_label,          # used in filename
                drop_questions=drop_questions,
                **plot_kwargs,
            )

    groups = split_participant_groups(all_participants, split=split_groups)
    for group_label, df in groups.items():
        _run(df, group_label)




def run_all_area_metric_plots(
    all_participants: pd.DataFrame,
    metrics=None,
    drop_question_variants=(False, True),
    show=True,
    save=None,
    to_paper=None,
    split_groups: bool = True,
):
    """
    Convenience wrapper: for every metric, generate label-vs-location matrices
    for hunters, gatherers, and all participants, with and without questions.

    """
    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS_VIZES

    # A sweep over metric x drop_questions x group x selected: up to 240 figures.
    # Their matrices pool into one long table rather than 240 small ones -- see
    # plot_output.collect_tables.
    with collect_tables("attention_allocation", plots=["area_label_by_loc_heatmap"]):
        for metric in metrics:
            for dq in drop_question_variants:
                print(f"\n=== {metric} (drop_questions={dq}) ===")

                label_vs_loc_mat(
                    metric,
                    all_participants,
                    drop_questions=dq,
                    split_groups=split_groups,
                    show=show,
                    save=save,
                    to_paper=to_paper,
                )


# ==========================================================================
# from src/viz/visualisations_area_significance_heatmaps.py
# ==========================================================================

def _stars_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_pairwise_significance_heatmap(
    pairwise,
    title,
    alpha=0.05,
    areas=None,
    show=False,
):
    """
    Heatmap for pairwise comparisons table.

    Color = -log10(p_adj_holm)
    Annotation (upper triangle) = stars + direction of diff_i_minus_j

    pairwise must have: area_i, area_j, p_adj_holm, diff_i_minus_j

    Returns the figure; persistence belongs to the caller, which also holds the
    fixed-effects table that belongs with it (see
    ``statistics.mixed_area_comparisons.run_models_for_group``).
    """
    if areas is None:
        canonical = ["answer_A", "answer_B", "answer_C", "answer_D"]
        present = set(pairwise["area_i"]).union(set(pairwise["area_j"]))
        areas = [a for a in canonical if a in present]
        if not areas:
            areas = sorted(present)

    areas = list(areas)
    if len(areas) < 2:
        return None

    idx = {a: i for i, a in enumerate(areas)}
    n = len(areas)

    P = np.full((n, n), np.nan, dtype=float)
    E = np.full((n, n), np.nan, dtype=float)

    for _, r in pairwise.iterrows():
        a = r["area_i"]
        b = r["area_j"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        p = r.get("p_adj_holm", np.nan)
        eff = r.get("diff_i_minus_j", np.nan)

        P[i, j] = p
        P[j, i] = p
        E[i, j] = eff
        E[j, i] = -eff

    P_clip = np.clip(P, 1e-300, 1.0)
    intensity = -np.log10(P_clip)
    np.fill_diagonal(intensity, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(intensity, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_yticklabels(areas)
    ax.set_title(title)

    # grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(p_adj_holm)")

    # annotate only upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            p = P[i, j]
            eff = E[i, j]
            if pd.isna(p):
                continue
            if p < alpha:
                stars = _stars_from_p(p)
                sign = "↑" if eff > 0 else ("↓" if eff < 0 else "0")
                ax.text(j, i, "{}\n{}".format(stars, sign),
                        ha="center", va="center", fontsize=9)

    fig.tight_layout()

    if show:
        plt.show()

    return fig
