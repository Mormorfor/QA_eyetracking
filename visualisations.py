import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast

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







