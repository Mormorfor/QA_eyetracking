import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast
from collections import defaultdict, Counter

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
    output_root: str = "plots/basic_stats_barcharts",
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



## Would be ideal to transition to LMER, but I struggle with pairwise comparisons there.
## So for now we keep using statsmodels MixedLM, even thought text ids are not treated exactly correctly
## (or maybe its even better this way?)
def mixed_area_analysis(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
    area_col: str = Con.AREA_LABEL_COLUMN,
    alpha: float = 0.05,
):
    """
    Mixed-effects model: area effect on a given metric, with random intercepts
    for participants and texts.

    Model:
        stat_col ~ 0 + C(area_col)
        random intercept for participant_id
        variance component for text_id

        (approximately: stat ~ 0 + area + (1|participant) + (1|text) )

    Parameters
    ----------
    df : DataFrame
        Row-level data (one row per IA), already filtered to a subset
        (e.g. selected_answer_label == 'A').
    stat_col : str
        Continuous outcome to model (e.g. Con.MEAN_DWELL_TIME).
    trial_cols : tuple[str]
        Columns that define a unique trial observation.
        Not used directly in the formula, but dedup uses them to ensure
        one row per (trial, area).
    area_col : str
        Area factor (e.g. Con.AREA_LABEL_COLUMN).
    alpha : float
        Significance level for CIs and Holm correction.

    Returns
    -------
    result : statsmodels MixedLMResults
    fe_table : DataFrame
        Per-area fixed effect estimates and CIs.
    pairwise : DataFrame
        Pairwise comparisons (Holm-adjusted p-values).
    """

    dedup = (
        df[list(trial_cols) + [area_col, stat_col]]
        .drop_duplicates()
        .copy()
    )

    area_order = [
        a for a in ["answer_A", "answer_B", "answer_C", "answer_D"]
        if a in dedup[area_col].unique()
    ]
    if not area_order:
        area_order = sorted(dedup[area_col].unique())

    dedup[area_col] = pd.Categorical(
        dedup[area_col],
        categories=area_order,
        ordered=True,
    )

    formula = f"{stat_col} ~ 0 + C({area_col})"
    model = smf.mixedlm(
        formula,
        data=dedup,
        groups=dedup[Con.PARTICIPANT_ID],
        re_formula="1",
        vc_formula={"text": "0 + C(text_id)"},
    )
    result = model.fit(method="lbfgs", reml=True)

    fe = result.fe_params.copy()
    ci = result.conf_int(alpha=alpha).loc[fe.index]
    fe_idx = fe.index.tolist()

    def pname(level: str) -> str:
        for nm in fe_idx:
            if nm.endswith(f"[{level}]") or nm.endswith(f"[T.{level}]"):
                return nm

    fe_table = (
        pd.DataFrame(
            {
                "term": fe.index,
                "estimate": fe.values,
                "ci_low": ci[0].values,
                "ci_high": ci[1].values,
            }
        )
        .assign(
            area=lambda d: d["term"].str.extract(
                r"\[(?:T\.)?([^\]]+)\]"
            )
        )
        .set_index("area")
        .loc[area_order]
        .reset_index()
    )

    # Pairwise comparisons (Holm-adjusted)
    pairs = []
    k_fe = len(fe_idx)

    for i in range(len(area_order)):
        for j in range(i + 1, len(area_order)):
            a, b = area_order[i], area_order[j]
            L = np.zeros((1, k_fe))
            L[0, fe_idx.index(pname(a))] = 1.0
            L[0, fe_idx.index(pname(b))] = -1.0

            t_res = result.t_test(L)
            eff = float(np.asarray(t_res.effect).ravel()[0])
            se = float(np.asarray(t_res.sd).ravel()[0])
            tval = float(np.asarray(t_res.tvalue).ravel()[0])
            pval = float(np.asarray(t_res.pvalue).ravel()[0])
            ci_lo, ci_hi = np.asarray(
                t_res.conf_int(alpha=alpha)
            ).ravel()

            pairs.append(
                {
                    "area_i": a,
                    "area_j": b,
                    "diff_i_minus_j": eff,
                    "se": se,
                    "t": tval,
                    "p_unc": pval,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                }
            )

    pairwise = pd.DataFrame(pairs)
    pairwise["p_adj_holm"] = multipletests(
        pairwise["p_unc"], method="holm"
    )[1]
    pairwise["sig"] = np.where(
        pairwise["p_adj_holm"] < alpha, "★", ""
    )

    return result, fe_table, pairwise


def run_all_area_barplots_and_models(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metrics=None,
    alpha: float = 0.05,
    output_root: str = "plots/basic_stats_barcharts",
    save_plots: bool = True,
    save_tables: bool = True,
    tables_root: str = "output_data/area_barplots",
    print_summaries: bool = True,
):
    """
    For each metric and each selected answer label (A–D), run area-level
    barplots and mixed-effects models for hunters & gatherers.

    All filtering happens inside this function:
    - rows with area_label == "question" are removed
    - data are split by selected_answer_label ('A', 'B', 'C', 'D')

    For each (group, metric, selected_label) combination, this function:
    - creates a bar plot (mean +/- 95% CI per area; saved as PNG if requested)
    - fits a mixed-effects model with area as fixed effect
    - computes pairwise contrasts between areas (Holm-corrected)
    - optionally saves the fixed-effect table and pairwise table as CSV

    Parameters
    ----------
    hunters : DataFrame
        Row-level hunters data (output of the preprocessing pipeline).
    gatherers : DataFrame
        Row-level gatherers data (output of the preprocessing pipeline).
    metrics : list[str] or None
        List of metric column names to analyse (e.g. [Con.MEAN_DWELL_TIME]).
        If None, uses Con.AREA_METRIC_COLUMNS.
    alpha : float
        Significance level used for confidence intervals and Holm correction.
    output_root : str
        Root directory for saving bar plots. Each metric will get its own
        subfolder under this directory (handled by plot_area_ci_bar).
    save_plots : bool
        If True, save PNG barplots for each (group, selected_label) combo.
    save_tables : bool
        If True, save fixed-effect and pairwise tables as CSV under tables_root.
    tables_root : str
        Root directory for saving result tables. Files are stored as:
        tables_root / <metric> / <group> /
            <group>__<label>__fe.csv
            <group>__<label>__pairwise.csv
    print_summaries : bool
        If True, print model summaries and pairwise comparison tables
        to stdout (useful for interactive exploration in notebooks).

    Returns
    -------
    results : dict
        Nested dictionary with structure:
        results[group][metric][selected_label] = {
            "fig": matplotlib Figure,
            "summary": DataFrame (descriptive stats per area),
            "model": MixedLMResults,
            "fe_table": DataFrame (fixed-effect estimates per area),
            "pairwise": DataFrame (pairwise area comparisons)
        }
    """
    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS

    def _save_model_tables(
        fe_table: pd.DataFrame,
        pairwise: pd.DataFrame,
        group_name: str,
        metric: str,
        selected_label: str,
    ) -> None:
        """
        Save fe_table and pairwise tables as CSV under a structured directory:

        tables_root / metric / group_name /
            <group>__<label>__fe.csv
            <group>__<label>__pairwise.csv
        """
        base_dir = os.path.join(tables_root, metric, group_name)
        os.makedirs(base_dir, exist_ok=True)

        fe_path = os.path.join(
            base_dir, f"{group_name}__{selected_label}__fe.csv"
        )
        pw_path = os.path.join(
            base_dir, f"{group_name}__{selected_label}__pairwise.csv"
        )

        fe_table.to_csv(fe_path, index=False)
        pairwise.to_csv(pw_path, index=False)

    def _run_for_group(df: pd.DataFrame, group_name: str) -> dict:
        """
        Run barplots + mixed models for one group (hunters/gatherers).
        Returns a dict: metric -> selected_label -> results dict.
        """
        df_noq = df[df[Con.AREA_LABEL_COLUMN] != "question"].copy()

        group_results: dict = {}

        for metric in metrics:
            metric_results: dict = {}

            available_labels = [
                lab
                for lab in ["A", "B", "C", "D"]
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

                model, fe_table, pairwise = mixed_area_analysis(
                    subset,
                    stat_col=metric,
                    alpha=alpha,
                )

                if save_tables:
                    _save_model_tables(
                        fe_table=fe_table,
                        pairwise=pairwise,
                        group_name=group_name,
                        metric=metric,
                        selected_label=ans,
                    )

                if print_summaries:
                    print(f"\n--- {group_name.upper()}, selected = {ans} ---")
                    print(model.summary())
                    print("\nFixed effects (per area):")
                    print(fe_table)
                    print("\nPairwise comparisons (Holm-corrected):")
                    print(pairwise.sort_values("p_adj_holm"))

                metric_results[ans] = {
                    "fig": fig,
                    "summary": summary,
                    "model": model,
                    "fe_table": fe_table,
                    "pairwise": pairwise,
                }

            group_results[metric] = metric_results

        return group_results

    results = {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
    }

    return results



# ---------------------------------------------------------------------------
# First/Last Visits Heatmaps
# ---------------------------------------------------------------------------

def matrix_plot_simplified_visits(
    df: pd.DataFrame,
    kind: str = "location",        # "label" or "location"
    which: str = "first",          # "first" or "last"
    drop_question: bool = True,
    h_or_g: str = "hunters",
    selected: str = "A",
    figsize: tuple = (8, 5),
    save: bool = False,
    output_root: str = "plots/simpl_visit_matrices",
    show: bool = True,
) -> None:
    """
    Plot a heatmap of visit frequencies for a fixed-length window taken from
    the simplified fixation sequence.

    For each trial/participant row:
      - Take the simplified sequence (label or location).
      - Optionally remove 'question' tokens.
      - Take either:
          * the first X tokens      (which = 'first'), or
          * the last  X tokens      (which = 'last'),
        where X = 4 if drop_question=True, else X = 5.
      - Re-index those tokens as positions 0..len(window)-1.
      - Count how often each area occurs at each position across trials.

    Result: a small matrix:
      rows   = visit position (0..3 or 0..4)
      cols   = areas (answers, and optionally question)
      values = counts.

    Parameters
    ----------
    df : DataFrame
        Data filtered to a specific group (hunters/gatherers) and selected
        answer label, typically one row per (trial, participant).
    kind : {"label", "location"}
        Which sequence to visualise:
        - "label"    -> Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
        - "location" -> Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    which : {"first", "last"}
        Whether to use the first X or last X entries of the sequence.
    drop_question : bool
        If True, remove 'question' tokens before taking the window.
    h_or_g : str
        Group label: 'hunters' or 'gatherers', used in the title/filename.
    selected : str
        Selected answer label ('A', 'B', 'C', 'D'), used in the title/filename.
    figsize : tuple
        Figure size for the heatmap.
    save : bool
        If True, save the figure as a PNG under output_root.
    output_root : str
        Root directory where the plot will be saved.
    show : bool
        If True, display the plot; otherwise close it after saving.
    """
    if kind == "label":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
        base_areas = list(Con.ANSWER_LABEL_CHOICES)
    elif kind == "location":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
        base_areas = list(Con.AREA_LABEL_CHOICES)
    else:
        raise ValueError("kind must be 'label' or 'location'")

    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'")

    window_len = 4 if drop_question else 5

    df_sel = (
        df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col]]
        .drop_duplicates()
        .copy()
    )

    def _parse_seq(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
        return x

    df_sel[seq_col] = df_sel[seq_col].apply(_parse_seq)
    df_sel = df_sel[df_sel[seq_col].notna()].copy()

    def _clean_and_window(seq):
        if not isinstance(seq, (list, tuple)):
            return []
        seq = list(seq)
        if drop_question:
            seq = [tok for tok in seq if tok != "question"]
        if not seq:
            return []
        if which == "first":
            return seq[:window_len]
        else:
            return seq[-window_len:]

    df_sel["window"] = df_sel[seq_col].apply(_clean_and_window)
    df_sel = df_sel[df_sel["window"].map(len) > 0].copy()

    if df_sel.empty:
        print(
            f"[info] No non-empty windows for kind='{kind}', which='{which}', "
            f"drop_question={drop_question}, group={h_or_g}, selected={selected}."
        )
        return

    df_sel["position"] = df_sel["window"].apply(
        lambda lst: list(range(len(lst)))
    )

    df_expl = df_sel.explode("position")
    df_expl = df_expl[df_expl["position"].notna()].copy()

    df_expl["area"] = df_expl.apply(
        lambda row: row["window"][int(row["position"])],
        axis=1,
    )

    agg = (
        df_expl.groupby(["position", "area"])
        .size()
        .reset_index(name="count")
    )

    pivot = (
        agg.pivot(index="position", columns="area", values="count")
        .fillna(0)
        .sort_index()
    )

    if drop_question:
        area_order = [a for a in base_areas if a != "question"]
    else:
        area_order = base_areas

    col_order = [c for c in area_order if c in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(pivot, annot=True, fmt="g", cmap="viridis")
    q_flag = " (no question)" if drop_question else " (with question)"
    ax.set_title(
        f"{which.capitalize()} {window_len} visits ({kind}){q_flag}\n"
        f"{h_or_g}, selected={selected}"
    )
    ax.set_xlabel("Area")
    ax.set_ylabel("Visit Order (position)")
    plt.tight_layout()

    if save:
        mode_dir = f"{which}_{kind}" + ("_noq" if drop_question else "_withq")
        out_dir = os.path.join(output_root, mode_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{h_or_g} - {selected}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()



def run_all_simplified_visit_matrices(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    drop_question: bool = True,
    kinds: tuple = ("label", "location"),
    which_list: tuple = ("first", "last"),
    answers: tuple = ("A", "B", "C", "D"),
    output_root: str = "plots/simpl_visit_matrices",
    save: bool = True,
    show: bool = True,
) -> None:
    """
    Generate visit-order heatmaps (first/last visits) for simplified sequences:
    - hunters and gatherers,
    - each selected answer (A–D),
    - label-based and/or location-based sequences,
    - optionally with 'question' included or excluded.

    Parameters
    ----------
    hunters : DataFrame
        Row-level hunters data.
    gatherers : DataFrame
        Row-level gatherers data.
    drop_question : bool
        If True, remove 'question' from the visit-order lists (4 positions max).
        If False, include 'question' (5 positions max).
    kinds : tuple of {"label", "location"}
        Which sequence types to plot.
    which_list : tuple of {"first", "last"}
        Whether to plot first and/or last visit orders.
    answers : tuple of str
        Selected answer labels to loop over ('A', 'B', 'C', 'D').
    output_root : str
        Root directory for saving heatmaps.
    save : bool
        If True, save PNG files to disk.
    show : bool
        If True, display plots interactively.
    """
    groups = {
        "hunters": hunters,
        "gatherers": gatherers,
    }

    for group_name, df in groups.items():
        df_group = df.copy()

        for ans in answers:
            subset = df_group[
                df_group[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans
            ].copy()

            if subset.empty:
                continue

            print(
                f"\n{group_name.upper()} — participants who selected {ans} "
                f"(drop_question={drop_question}):"
            )
            print("-" * 72)

            for which in which_list:
                for kind in kinds:
                    matrix_plot_simplified_visits(
                        subset,
                        kind=kind,
                        which=which,
                        drop_question=drop_question,
                        h_or_g=group_name,
                        selected=ans,
                        figsize=(8, 5),
                        save=save,
                        output_root=output_root,
                        show=show,
                    )


# ---------------------------------------------------------------------------
# Dominant Strategies
# ---------------------------------------------------------------------------

def build_strategy_dataframe(
    df: pd.DataFrame,
    kind: str = "location",          # "location" or "label"
    window_len: int = 4,
    drop_question: bool = True,
    strat_col: str = Con.STRATEGY_COL,
) -> pd.DataFrame:
    """
    Build a per-trial 'strategy' DataFrame from the simplified fixation sequences.

    - parses the list stored in Con.SIMPLIFIED_FIX_SEQ_BY_*
    - optionally removes 'question' tokens
    - takes the FIRST `window_len` entries
    - stores them as tuples in `strat_col`

    Returns a DataFrame with columns:
        [Con.TRIAL_ID, Con.PARTICIPANT_ID, strat_col]
    """
    if kind == "label":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
    elif kind == "location":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    else:
        raise ValueError("kind must be 'label' or 'location'")

    df_sel = (
        df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col]]
        .drop_duplicates()
        .copy()
    )

    def _parse_seq(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
        return x

    df_sel[seq_col] = df_sel[seq_col].apply(_parse_seq)

    def _first_window(seq):
        if not isinstance(seq, (list, tuple)):
            return ()
        seq = list(seq)
        if drop_question:
            seq = [tok for tok in seq if tok != "question"]
        if not seq:
            return ()
        return tuple(seq[:window_len])

    df_sel[strat_col] = df_sel[seq_col].apply(_first_window)
    return df_sel[[Con.TRIAL_ID, Con.PARTICIPANT_ID, strat_col]].copy()



def proportion_with_dominant_strategy(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    threshold: float = 0.8,
) -> float:
    """
    Proportion of participants whose most frequent strategy
    accounts for > threshold of their trials.
    """
    counts = df.groupby([id_col, strat_col]).size()
    total = counts.groupby(level=0).sum()
    top = counts.groupby(level=0).max()
    prop = top / total
    is_dominant = prop > threshold
    return float(is_dominant.mean())



def plot_dominant_strategy_hist(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    bins: int = 20,
    figsize=(6, 4),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/strategies",
    completed_flag_col: Optional[str] = None,
):
    """
    Histogram of the dominant-strategy proportion per participant:
    - What percentage of trials did the participant use their most common strategy?
    - How many participants fall into each bin?

    """
    counts = df.groupby([id_col, strat_col]).size()
    total = counts.groupby(level=0).sum()
    top = counts.groupby(level=0).max()
    dominant_prop = top / total

    if isinstance(bins, int):
        bin_edges = np.linspace(0, 1, bins + 1)
    else:
        bin_edges = np.asarray(bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(dominant_prop, bins=bin_edges)
    ax.set_xlabel("Proportion of trials in dominant strategy")
    ax.set_ylabel("Number of participants")
    ax.set_title(f"Distribution of Dominant-Strategy Usage ({h_or_g}) - {strat_col}")
    ticks = bin_edges
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(x*100)}%" for x in ticks], rotation=45)

    fig.tight_layout()
    if save:
        os.makedirs(output_root, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_root,
                f"dominant_prop_{strat_col}_{h_or_g}).png",
            ),
            dpi=300,
        )
    plt.show()

    return dominant_prop



def plot_dominance_gap(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    bins: int = 20,
    figsize=(12, 5),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/strategies",
    hist_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
):
    """
    For each participant:
      P1 = most frequent strategy proportion
      P2 = second-most frequent strategy proportion
      gap = P1 - P2

    Plots a histogram of gaps and a P2 vs P1 scatter.
    """
    hist_kwargs = hist_kwargs or {"edgecolor": "k"}
    scatter_kwargs = scatter_kwargs or {"alpha": 0.7}

    counts = df.groupby([id_col, strat_col]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    p1 = props.max(axis=1)

    def second_largest(row):
        vals = row[row > 0].nlargest(2)
        return vals.iloc[-1] if len(vals) > 1 else 0

    p2 = props.apply(second_largest, axis=1)
    gap = p1 - p2

    result = pd.DataFrame({"p1": p1, "p2": p2, "gap": gap})

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(gap, bins=bins, **hist_kwargs)
    axes[0].set_xlabel("Gap (P1 – P2)")
    axes[0].set_ylabel("Number of participants")
    axes[0].set_title(f"Histogram of Dominance Gaps ({h_or_g})")

    axes[1].scatter(p2, p1, **scatter_kwargs)
    axes[1].plot([0, 1], [0, 1], "r--", label="P1=P2")
    axes[1].set_xlabel("2nd-most common proportion (P2)")
    axes[1].set_ylabel("Most common proportion (P1)")
    axes[1].set_title(f"P2 vs. P1 per Participant ({h_or_g})")
    axes[1].legend()

    plt.tight_layout()
    if save:
        os.makedirs(output_root, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_root,
                f"dominance_gap_{strat_col}_{h_or_g}.png",
            ),
            dpi=300,
        )
    plt.show()

    return result



def plot_strategy_count_distribution(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    figsize=(6, 4),
    bins=None,
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/strategies",
    **plot_kwargs,
):
    """
    Distribution of how many distinct strategies each participant uses.
    """
    strat_counts = df.groupby(id_col)[strat_col].nunique()

    plt.figure(figsize=figsize)
    if bins is None:
        max_strat = strat_counts.max()
        bins = np.arange(0.5, max_strat + 1.5, 1.0)
    plt.hist(strat_counts, bins=bins, **plot_kwargs)
    plt.xlabel("Number of distinct strategies used")
    plt.ylabel("Number of participants")
    plt.title(f"Distribution of Strategy Counts ({h_or_g})")
    ticks = np.arange(1, strat_counts.max() + 1)
    plt.xticks(ticks)

    plt.tight_layout()
    if save:
        os.makedirs(output_root, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_root,
                f"dom_str_counts_{strat_col}_{h_or_g}.png",
            ),
            dpi=300,
        )

    plt.show()

    return strat_counts



def plot_dominant_strategy_counts_above_threshold(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    threshold: float = 0.5,
    figsize=(8, 4),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/strategies",
    **bar_kwargs,
):
    """
    For participants whose dominant strategy ≥ threshold of trials:
    barplot of which strategies are dominant and how many participants
    use each.
    """
    counts = df.groupby([id_col, strat_col]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    dominant_prop = props.max(axis=1)
    dominant_strat = props.idxmax(axis=1)
    mask = dominant_prop >= threshold
    filtered = dominant_strat[mask]
    freq = filtered.value_counts().sort_values(ascending=False)

    plt.figure(figsize=figsize)
    freq.plot(kind="bar", **bar_kwargs)
    plt.xlabel(strat_col)
    plt.ylabel("Number of participants")
    pct = int(threshold * 100)
    plt.title(
        f"Dominant Strategies (≥ {pct}% of trials) — Count of Participants ({h_or_g})"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        os.makedirs(output_root, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_root,
                f"str_above_thresh_{strat_col}_{h_or_g}.png",
            ),
            dpi=300,
        )

    plt.show()
    return freq



def build_prefix_completion_map_from_series(series: pd.Series, full_len: int = 4):
    """
    From fully observed strategies (length == full_len), learn how
    prefixes tend to be completed.

    Returns a dict prefix -> most frequent full sequence.
    """
    full_counts = series[series.map(len).eq(full_len)].value_counts()
    by_prefix = defaultdict(Counter)
    for full_seq, c in full_counts.items():
        for k in range(1, full_len):
            pref = full_seq[:k]
            by_prefix[pref][full_seq] += c
    prefix2full = {
        pref: max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for pref, counter in by_prefix.items()
    }
    return prefix2full



def add_completed_sequence_column(
    df: pd.DataFrame,
    strat_col: str = Con.STRATEGY_COL,
    full_len: int = 4,
    col_suffix: str = "_completed",
    prefix2full: dict = None,
):
    """
    Use prefix-completion map to fill shorter strategies up to full_len.
    Sequences are assumed to be tuples.

    Adds two columns:
    - <strat_col><col_suffix>: the completed sequence
    - <strat_col>_was_completed: bool indicating whether the original
      sequence was shorter than full_len (i.e. completion attempted)
    """
    df = df.copy()
    series = df[strat_col]

    if prefix2full is None:
        prefix2full = build_prefix_completion_map_from_series(
            series, full_len=full_len
        )

    was_completed_col = f"{strat_col}_was_completed"
    df[was_completed_col] = series.map(lambda t: len(t) < full_len)

    def _complete(t):
        if len(t) >= full_len:
            return t[:full_len]
        return prefix2full.get(t, t)

    comp = series.map(_complete)
    comp_col = f"{strat_col}{col_suffix}"
    df[comp_col] = comp

    return df, prefix2full


def summarize_before_after(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    raw_col: str = Con.STRATEGY_COL,
    comp_col: Optional[str] = None,
    threshold: float = 0.5,
    bins: int = 20,
    figsize=(8, 5),
    h_or_g: str = "hunters",
    save: bool = True,
    out_prefix: str = "plots/strategies",
    density: bool = False,
    hist_kwargs: Optional[dict] = None,
    full_len: int = 4,
):
    """
    Compare dominant-strategy proportions BEFORE vs AFTER completion,
    per participant, and plot overlapping histograms.

    Parameters
    ----------
    df : DataFrame
        Must contain at least:
          - id_col (e.g. Con.PARTICIPANT_ID)
          - raw_col (e.g. 'strategy')
          - comp_col (e.g. 'strategy_completed')
    id_col : str
        Participant ID column.
    raw_col : str
        Column with raw strategies (tuples, length <= full_len).
    comp_col : str or None
        Column with completed strategies. If None, defaults to
        f"{raw_col}_completed".
    threshold : float
        Threshold for “dominant strategy” (e.g. 0.5 for 50% of trials).
    bins : int or array
        Bins for the histograms.
    figsize : tuple
        Figure size.
    h_or_g : str
        Group label for titles/filenames ('hunters' / 'gatherers').
    save : bool
        Save the figure to disk.
    out_prefix : str
        Directory where the PNG will be stored.
    density : bool
        If True, plot density instead of counts.
    hist_kwargs : dict or None
        Extra kwargs for plt.hist (applied to both histograms).
    full_len : int
        Full strategy length (used for normalising sequences in change stats).

    Returns
    -------
    summary : dict
        Aggregate statistics about raw vs completed dominance.
    both : DataFrame
        Per-participant table with raw/comp proportions and change info.
    fig, ax : matplotlib Figure and Axes
        The histogram figure and axes.
    """
    if comp_col is None:
        comp_col = f"{raw_col}_completed"

    hist_kwargs = hist_kwargs or {"alpha": 0.5, "edgecolor": "k"}

    counts_raw = df.groupby([id_col, raw_col]).size().unstack(fill_value=0)
    prop_raw = counts_raw.max(axis=1) / counts_raw.sum(axis=1)
    dom_raw = counts_raw.idxmax(axis=1)

    counts_comp = df.groupby([id_col, comp_col]).size().unstack(fill_value=0)
    prop_comp = counts_comp.max(axis=1) / counts_comp.sum(axis=1)
    dom_comp = counts_comp.idxmax(axis=1)

    both = pd.DataFrame({"raw": prop_raw, "comp": prop_comp}).dropna()
    both["delta"] = both["comp"] - both["raw"]

    both["raw_label"] = dom_raw.reindex(both.index)
    both["comp_label"] = dom_comp.reindex(both.index)
    both["changed_label"] = both["raw_label"] != both["comp_label"]

    comp_series = df[comp_col]
    raw_series = df[raw_col]
    mask_valid = comp_series.notna() & raw_series.notna()

    def _norm(t):
        if t is None:
            return None
        t = tuple(t)
        return t[:full_len] if len(t) > full_len else t

    changed_rows = (
        comp_series[mask_valid].map(_norm)
        != raw_series[mask_valid].map(_norm)
    )

    per_part_changed = (
        pd.DataFrame(
            {
                "changed": changed_rows,
                "total": True,
                id_col: df.loc[mask_valid, id_col].values,
            }
        )
        .groupby(id_col)
        .agg(
            seq_pct_changed=(
                "changed",
                lambda s: float(s.mean()) if len(s) else np.nan,
            ),
            seq_changed_n=("changed", "sum"),
            seq_total_n=("total", "sum"),
        )
    )

    both = both.join(per_part_changed, how="left")

    changed_n = int(both["changed_label"].sum())
    changed_pct = (
        float(changed_n / len(both) * 100) if len(both) else np.nan
    )
    mean_seq_pct_changed = (
        float(both["seq_pct_changed"].mean() * 100)
        if both["seq_pct_changed"].notna().any()
        else np.nan
    )
    median_seq_pct_changed = (
        float(both["seq_pct_changed"].median() * 100)
        if both["seq_pct_changed"].notna().any()
        else np.nan
    )

    summary = {
        "participants": int(len(both)),
        "mean_raw": float(both["raw"].mean()) if len(both) else np.nan,
        "mean_completed": float(both["comp"].mean())
        if len(both)
        else np.nan,
        "mean_delta": float(both["delta"].mean())
        if len(both)
        else np.nan,
        f"raw_≥{int(threshold*100)}%": float(
            (both["raw"] >= threshold).mean() * 100
        )
        if len(both)
        else np.nan,
        f"comp_≥{int(threshold*100)}%": float(
            (both["comp"] >= threshold).mean() * 100
        )
        if len(both)
        else np.nan,
        "changed_label_n": changed_n,
        "changed_label_pct": changed_pct,
        "mean_seq_pct_changed": mean_seq_pct_changed,
        "median_seq_pct_changed": median_seq_pct_changed,
    }

    bin_edges = (
        np.linspace(0, 1, bins + 1)
        if isinstance(bins, int)
        else np.asarray(bins)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        both["raw"],
        bins=bin_edges,
        density=density,
        label="Raw",
        **hist_kwargs,
    )
    ax.hist(
        both["comp"],
        bins=bin_edges,
        density=density,
        label="Completed",
        **hist_kwargs,
    )
    ax.set_xlabel("Proportion of trials in dominant strategy")
    ax.set_ylabel("Density" if density else "Number of participants")
    ax.set_title(
        f"Dominant-Strategy Proportion: Raw vs Completed ({h_or_g})"
    )
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(
        [f"{int(x*100)}%" for x in bin_edges], rotation=45
    )
    ax.legend()
    fig.tight_layout()

    if save:
        os.makedirs(out_prefix, exist_ok=True)
        fig.savefig(
            os.path.join(
                out_prefix,
                f"dominant_prop_raw_vs_completed_{h_or_g}.png",
            ),
            dpi=300,
        )

    plt.show()

    return summary, both, fig, ax




def plot_strategies(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/strategies",
):
    """
    Convenience wrapper: runs all strategy plots on one DataFrame.
    Assumes `strat_col` and `strat_col + "_completed"` exist.
    """
    # Histogram of dominant usage (raw)
    dominant = plot_dominant_strategy_hist(
        df,
        id_col=id_col,
        strat_col=strat_col,
        bins=20,
        figsize=(8, 5),
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
    )

    # Histogram of dominant usage (completed)
    comp_dom = plot_dominant_strategy_hist(
        df,
        id_col=id_col,
        strat_col=strat_col + "_completed",
        bins=20,
        figsize=(8, 5),
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        completed_flag_col=strat_col + "_was_completed",
    )

    # Dominance gaps
    gaps = plot_dominance_gap(
        df,
        id_col=id_col,
        strat_col=strat_col,
        bins=20,
        figsize=(10, 4),
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
    )

    # How many strategies per participant?
    counts = plot_strategy_count_distribution(
        df,
        id_col=id_col,
        strat_col=strat_col,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
    )

    # Which strategies dominate above 50%?
    strategies = plot_dominant_strategy_counts_above_threshold(
        df,
        id_col=id_col,
        strat_col=strat_col + "_completed",
        threshold=0.5,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
    )

    return dominant, comp_dom, gaps, counts, strategies


def run_all_strategy_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    kind: str = "location",
    window_len: int = 4,
    threshold: float = 0.5,
    output_root: str = "plots/strategies",
    save: bool = True,
) -> dict:
    """
    Build strategy data from simplified sequences and run all strategy analyses
    for hunters and gatherers.

    - uses FIRST `window_len` entries of the simplified sequence
    - always drops 'question' tokens
    - `kind` chooses between SIMPLIFIED_FIX_SEQ_BY_LOCATION / ...BY_LABEL

    Returns nested dict:
        results[group_name] = {
            "df": df_strat_with_completed,
            "prefix_map": prefix_map,
            "dominant_prop": dominant_prop,
            "plots": (dh, cdh, gh, ch, sh),
        }
    """
    results = {}

    for group_name, df in {"hunters": hunters, "gatherers": gatherers}.items():
        df_strat = build_strategy_dataframe(
            df,
            kind=kind,
            window_len=window_len,
            drop_question=True,
            strat_col=Con.STRATEGY_COL,
        )

        dom_prop_raw = proportion_with_dominant_strategy(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            strat_col=Con.STRATEGY_COL,
            threshold=threshold,
        )
        print(
            f"{dom_prop_raw:.1%} of {group_name} participants had a dominant "
            f"strategy (>{threshold * 100:.0f}% of trials) before completion."
        )

        df_strat, prefix_map = add_completed_sequence_column(
            df_strat,
            strat_col=Con.STRATEGY_COL,
            full_len=window_len,
            col_suffix="_completed",
            prefix2full=None,
        )

        completed_col = f"{Con.STRATEGY_COL}_completed"
        dom_prop_completed = proportion_with_dominant_strategy(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            strat_col=completed_col,
            threshold=threshold,
        )
        print(
            f"{dom_prop_completed:.1%} of {group_name} participants had a dominant "
            f"strategy (>{threshold * 100:.0f}% of trials) after completion."
        )

        ba_summary, ba_table, ba_fig, ba_ax = summarize_before_after(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            raw_col=Con.STRATEGY_COL,
            comp_col=completed_col,
            threshold=threshold,
            bins=20,
            figsize=(8, 5),
            h_or_g=group_name,
            save=save,
            out_prefix=output_root,
            density=False,
            full_len=window_len,
        )

        plots = plot_strategies(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            strat_col=Con.STRATEGY_COL,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        results[group_name] = {
            "df": df_strat,
            "prefix_map": prefix_map,
            "dominant_prop_raw": dom_prop_raw,
            "dominant_prop_completed": dom_prop_completed,
            "before_after_summary": ba_summary,
            "before_after_table": ba_table,
            "before_after_fig": ba_fig,
            "before_after_ax": ba_ax,
            "plots": plots,
        }

    return results


# ---------------------------------------------------------------------------
# Time-Segment Analyses (before / during / after selected answer first encountered)
# ---------------------------------------------------------------------------

def _assign_time_segment(
    group: pd.DataFrame,
    area_col: str,
    selected_col: str,
    segment_col: str = Con.SEGMENT_COLUMN,
) -> pd.Series:
    """
    Helper: given all rows for one (participant, trial), assign a time segment
    ('before', 'during', 'after') based on when the selected answer is fixated.

    Logic:
      - area_col contains e.g. 'question', 'answer_A', 'answer_B', ...
      - selected_col contains 'A' / 'B' / 'C' / 'D'
      - 'before': all rows before first fixation on the selected answer
      - 'during': all fixations on the selected answer from its first
                  occurrence onwards
      - 'after' : only NON-selected areas from the first non-selected
                  fixation after the selected answer onwards
                  (later revisits to the selected answer stay 'during')
    """
    area = group[area_col].astype(str).str.lower().to_numpy()
    selected = str(group[selected_col].iloc[0]).lower()
    target = f"answer_{selected}"

    mask = (area == target)
    n = len(group)
    out = np.full(n, "before", dtype=object)

    # No fixation on the selected answer at all -> everything 'before'
    if not mask.any():
        return pd.Series(out, index=group.index, name=segment_col)

    first_pos = np.flatnonzero(mask)[0]

    out[first_pos:] = "during"

    # Look for the first NON-target after that
    after_first = mask[first_pos + 1:]
    non_target_rel = np.flatnonzero(~after_first)

    if non_target_rel.size > 0:
        interruption = first_pos + 1 + non_target_rel[0]

        # From interruption onwards, mark ONLY non-selected areas as "after".
        # Any later visits to the selected answer remain "during".
        after_range = np.arange(interruption, n)
        non_target_after = after_range[~mask[interruption:]]
        out[non_target_after] = "after"

    return pd.Series(out, index=group.index, name=segment_col)




def add_time_segment_column(
    df: pd.DataFrame,
    group_cols=(Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    segment_col: str = Con.SEGMENT_COLUMN,
) -> pd.DataFrame:
    """
    Add a SEGMENT_COLUMN column to a row-level DF with IA features.

    Parameters
    ----------
    df : DataFrame
        Row-level data (one row per IA).
    group_cols : tuple[str]
        Columns that uniquely identify a trial, e.g.
        (PARTICIPANT_ID, TRIAL_ID).
    area_col : str
        Column with area labels such as 'question', 'answer_A', ...
    selected_col : str
        Column with the selected answer label ('A','B','C','D').
    segment_col : str
        Name of the resulting time-segment column.

    Returns
    -------
    df_out : DataFrame
        Copy of df with an added categorical column `segment_col`
        taking values in ['before', 'during', 'after'].
    """
    df_out = df.copy()

    df_out[segment_col] = (
        df_out
        .groupby(list(group_cols), group_keys=False)
        .apply(
            lambda g: _assign_time_segment(
                g,
                area_col=area_col,
                selected_col=selected_col,
                segment_col=segment_col,
            )
        )
    )

    order = ["before", "during", "after"]
    df_out[segment_col] = pd.Categorical(
        df_out[segment_col],
        categories=order,
        ordered=True,
    )
    return df_out


def _plot_time_segment_bar(
    df: pd.DataFrame,
    value_col: Optional[str],
    metric_col_name: str,
    metric_label: str,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    transform=None,
    subdir: str = "",
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/time_segments",
    title: Optional[str] = None,
):
    """
    Generic helper to make time-segment barplots.

    Parameters
    ----------
    df : DataFrame
        Row-level data with a time-segment column (SEGMENT_COLUMN).
    value_col : str or None
        If not None, we aggregate the mean of this column per
        (participant, trial, segment). If None, we use the group size
        (.size()) as the metric.
    metric_col_name : str
        Name of the metric column in the aggregated DataFrame.
    metric_label : str
        Label to show on the y-axis and in the title.
    id_col : str
        Participant ID column.
    trial_index_col : str
        Trial ID column.
    segment_col : str
        Column indicating the time segment (e.g. SEGMENT_COLUMN).
    transform : callable or None
        Optional function applied to df before aggregation
        (e.g., to add a 'skipped' column).
    subdir : str
        Subdirectory under output_root for saving plots.
    figsize : tuple
        Figure size.
    h_or_g : str
        Group label for titles/filenames ('hunters' / 'gatherers').
    save : bool
        If True, save the figure as PNG.
    output_root : str
        Root directory for plot saving.
    title : str or None
        Custom title; if None, a default is used.

    Returns
    -------
    fig : Figure
    summary : DataFrame
        Mean, SD, and N per segment.
    """
    order = ["before", "during", "after"]

    if transform is not None:
        df = transform(df.copy())

    if value_col is None:
        df_metric = (
            df
            .groupby([id_col, trial_index_col, segment_col], observed=False)
            .size()
            .reset_index(name=metric_col_name)
        )
    else:
        df_metric = (
            df
            .groupby([id_col, trial_index_col, segment_col], observed=False)[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: metric_col_name})
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_metric,
        x=segment_col,
        y=metric_col_name,
        order=order,
        estimator=np.mean,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("Time segment")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} by time segment ({h_or_g})")

    summary = (
        df_metric
        .groupby(segment_col)[metric_col_name]
        .agg(mean="mean", sd="std", n="count")
        .reindex(order)
        .reset_index()
    )

    fig.tight_layout()
    if save:
        out_dir = os.path.join(output_root, subdir or metric_col_name)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{h_or_g}__{subdir or metric_col_name}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300)

    plt.show()
    return fig, summary



def plot_time_segment_mean_dwell(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean dwell time (per trial) by time segment ('before', 'during', 'after').
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=dwell_col,
        metric_col_name=dwell_col,
        metric_label=f"Mean {dwell_col}",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="mean_dwell_time",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )


def plot_time_segment_sequence_length(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/time_segments",
    title: Optional[str] = None,
):
    """
    Average sequence length (number of IA rows) by time segment.
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=None,
        metric_col_name=Con.SEQUENCE_LENGTH_COLUMN,
        metric_label="Average number of rows",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="sequence_length",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )



def plot_time_segment_fixation_count(
    df: pd.DataFrame,
    fix_col: str = Con.IA_FIXATIONS_COUNT,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean fixation count (per trial) by time segment.
    """
    return _plot_time_segment_bar(
        df=df,
        value_col=fix_col,
        metric_col_name=fix_col,
        metric_label=f"Mean {fix_col}",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=None,
        subdir="fixation_count",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )


def plot_time_segment_skip_rate(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    id_col: str = Con.PARTICIPANT_ID,
    trial_index_col: str = Con.TRIAL_ID,
    segment_col: str = Con.SEGMENT_COLUMN,
    figsize=(8, 6),
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "plots/time_segments",
    title: Optional[str] = None,
):
    """
    Mean skip rate (proportion of IA rows with dwell == 0) by time segment.
    """
    def _add_skipped(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        # IA-level flag for time-segment computation
        d[Con.SKIPPED_COLUMN] = (d[dwell_col] == 0).astype(int)
        return d

    return _plot_time_segment_bar(
        df=df,
        value_col=Con.SKIPPED_COLUMN,
        metric_col_name=Con.SKIP_RATE,          # <- existing "skip_rate"
        metric_label="Skip rate (proportion)",
        id_col=id_col,
        trial_index_col=trial_index_col,
        segment_col=segment_col,
        transform=_add_skipped,
        subdir="skip_rate",
        figsize=figsize,
        h_or_g=h_or_g,
        save=save,
        output_root=output_root,
        title=title,
    )




def run_all_time_segment_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    group_cols=(Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    selected_col: str = Con.SELECTED_ANSWER_LABEL_COLUMN,
    dwell_col: str = Con.IA_DWELL_TIME,
    fix_col: str = Con.IA_FIXATIONS_COUNT,
    segment_col: str = Con.SEGMENT_COLUMN,
    output_root: str = "plots/time_segments",
    save: bool = True,
) -> dict:
    """
    Convenience wrapper:
    - add SEGMENT_COLUMN to hunters & gatherers
    - plot:
        * mean dwell time by segment
        * average sequence length by segment
        * mean fixation count by segment
        * mean skip rate by segment

    Returns
    -------
    results : dict
        results["hunters"] / results["gatherers"] each contain:
            {
              "df": df_with_segment,
              "mean_dwell": (fig, summary_df),
              "sequence_length": (fig, summary_df),
              "fixation_count": (fig, summary_df),
              "skip_rate": (fig, summary_df),
            }
    """
    results = {}

    for group_name, df in {"hunters": hunters, "gatherers": gatherers}.items():
        df_seg = add_time_segment_column(
            df,
            group_cols=group_cols,
            area_col=area_col,
            selected_col=selected_col,
            segment_col=segment_col,
        )

        md_fig, md_summary = plot_time_segment_mean_dwell(
            df_seg,
            dwell_col=dwell_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        sl_fig, sl_summary = plot_time_segment_sequence_length(
            df_seg,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        fc_fig, fc_summary = plot_time_segment_fixation_count(
            df_seg,
            fix_col=fix_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        sr_fig, sr_summary = plot_time_segment_skip_rate(
            df_seg,
            dwell_col=dwell_col,
            id_col=group_cols[0],
            trial_index_col=group_cols[1],
            segment_col=segment_col,
            h_or_g=group_name,
            save=save,
            output_root=output_root,
        )

        results[group_name] = {
            "df": df_seg,
            "mean_dwell": (md_fig, md_summary),
            "sequence_length": (sl_fig, sl_summary),
            "fixation_count": (fc_fig, fc_summary),
            "skip_rate": (sr_fig, sr_summary),
        }

    return results
