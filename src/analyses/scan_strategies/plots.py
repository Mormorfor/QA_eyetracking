"""Figures: the opening scan over the four answer options.

Backs the paper's First-scan behavior subsection. The strategy computation itself is
`features/strategies.py` -- unified there in T1.1/T1.6, after two implementations of
it drifted apart precisely because the plots sat this far from the features."""

from typing import Optional
import ast
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import columns as Con
from src.features.strategies import (
    DEFAULT_DOMINANCE_THRESHOLD,
    DEFAULT_WINDOW_LEN,
    add_completed_strategy_column,
    build_prefix_completion_map,
    build_starting_strategies,
    dominance_gap_by_participant,
    dominant_strategy_by_participant,
    dominant_strategy_counts,
    has_dominant_strategy,
    strategy_variety_by_participant,
    summarize_completion_effect,
)
from src.lib.plotting.output import save_output
from src.features.scope import split_participant_groups
import seaborn as sns


# ==========================================================================
# from src/viz/visualisations_strategies.py
# ==========================================================================

# ---------------------------------------------------------------------------
# Dominant Strategies
# ---------------------------------------------------------------------------

# NOTE: build_strategy_dataframe used to live here. It was a byte-identical
# duplicate of features.strategies.build_starting_strategies (verified on
# both L1 groups: same rows, keys and strategy tuples), so it was removed on
# 2026-09-20 and callers now use the derived one. Only the keyword changed --
# `strat_col=` became `out_col=`.


def proportion_with_dominant_strategy(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> float:
    """
    Proportion of participants whose most frequent strategy
    accounts for AT LEAST `threshold` of their trials.
    """
    dominant = dominant_strategy_by_participant(
        df, id_col=id_col, strat_col=strat_col
    )
    is_dominant = has_dominant_strategy(
        dominant[Con.DOMINANCE_SCORE], threshold=threshold
    )
    return float(is_dominant.mean())



def plot_dominant_strategy_hist(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    bins: int = 20,
    figsize=(6, 4),
    h_or_g: str = "hunters",
    save: Optional[bool] = None,
    to_paper=None,
    completed_flag_col: Optional[str] = None,
):
    """
    Histogram of the dominant-strategy proportion per participant:
    - What percentage of trials did the participant use their most common strategy?
    - How many participants fall into each bin?

    """
    dominant_prop = dominant_strategy_by_participant(
        df, id_col=id_col, strat_col=strat_col
    )[Con.DOMINANCE_SCORE]

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
    save_output(
        fig,
        analysis="scan_strategies",
        plot="dominant_strategy_proportion",
        tables={"dominance": dominant_prop.rename("dominance_score").reset_index()},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
        kind=strat_col,
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
    save: Optional[bool] = None,
    to_paper=None,
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

    result = dominance_gap_by_participant(df, id_col=id_col, strat_col=strat_col)
    p1, p2, gap = result["p1"], result["p2"], result["gap"]

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
    save_output(
        fig,
        analysis="scan_strategies",
        plot="dominance_gap",
        tables={"gap": result.reset_index()},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
        kind=strat_col,
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
    save: Optional[bool] = None,
    to_paper=None,
    **plot_kwargs,
):
    """
    Distribution of how many distinct strategies each participant uses.
    """
    strat_counts = strategy_variety_by_participant(
        df, id_col=id_col, strat_col=strat_col
    )

    fig = plt.figure(figsize=figsize)
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
    save_output(
        fig,
        analysis="scan_strategies",
        plot="strategy_count_distribution",
        tables={"counts": strat_counts.rename("n_strategies").reset_index()},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
        kind=strat_col,
    )

    plt.show()

    return strat_counts



def plot_dominant_strategy_counts_above_threshold(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    figsize=(8, 4),
    h_or_g: str = "hunters",
    save: Optional[bool] = None,
    to_paper=None,
    **bar_kwargs,
):
    """
    For participants whose dominant strategy ≥ threshold of trials:
    barplot of which strategies are dominant and how many participants
    use each.
    """
    freq = dominant_strategy_counts(
        df, id_col=id_col, strat_col=strat_col, threshold=threshold
    )

    fig = plt.figure(figsize=figsize)
    freq.plot(kind="bar", **bar_kwargs)
    plt.xlabel(strat_col)
    plt.ylabel("Number of participants")
    pct = int(threshold * 100)
    plt.title(
        f"Dominant Strategies (≥ {pct}% of trials) — Count of Participants ({h_or_g})"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_output(
        fig,
        analysis="scan_strategies",
        plot="dominant_strategies_above_threshold",
        tables={"counts": freq.rename("n_participants").reset_index()},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
        kind=strat_col,
    )

    plt.show()
    return freq



# NOTE: build_prefix_completion_map_from_series and add_completed_sequence_column
# moved to features.strategies on 2026-09-20 (as build_prefix_completion_map
# and add_completed_strategy_column) -- they are the interrupted-scan completion
# method, not plotting. Behaviour unchanged, including the population-wide scope
# of the learned map (todo.md T3.21 row 5).


def summarize_before_after(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    raw_col: str = Con.STRATEGY_COL,
    comp_col: Optional[str] = None,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    bins: int = 20,
    figsize=(8, 5),
    h_or_g: str = "hunters",
    save: Optional[bool] = None,
    to_paper=None,
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
    to_paper : bool or None
        Mirror into the Overleaf-synced papers/ tree (off by default).
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

    summary, both = summarize_completion_effect(
        df,
        id_col=id_col,
        raw_col=raw_col,
        comp_col=comp_col,
        threshold=threshold,
        full_len=full_len,
    )

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

    save_output(
        fig,
        analysis="scan_strategies",
        plot="dominance_raw_vs_completed",
        tables={"summary": summary, "per_participant": both.reset_index()},
        save=save,
        to_paper=to_paper,
        group=h_or_g,
    )

    plt.show()

    return summary, both, fig, ax




def plot_strategies(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    h_or_g: str = "hunters",
    save: Optional[bool] = None,
    to_paper=None,
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
        to_paper=to_paper,
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
        to_paper=to_paper,
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
        to_paper=to_paper,
    )

    # How many strategies per participant?
    counts = plot_strategy_count_distribution(
        df,
        id_col=id_col,
        strat_col=strat_col,
        h_or_g=h_or_g,
        save=save,
        to_paper=to_paper,
    )

    # Which strategies dominate above 50%?
    strategies = plot_dominant_strategy_counts_above_threshold(
        df,
        id_col=id_col,
        strat_col=strat_col + "_completed",
        threshold=0.5,
        h_or_g=h_or_g,
        save=save,
        to_paper=to_paper,
    )

    return dominant, comp_dom, gaps, counts, strategies


def run_all_strategy_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    kind: str = "location",
    window_len: int = 4,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    save: Optional[bool] = None,
    to_paper=None,
    include_all: bool = True,
) -> dict:
    """
    Build strategy data from simplified sequences and run all strategy analyses
    for hunters and gatherers.

    - uses FIRST `window_len` entries of the simplified sequence
    - always drops 'question' tokens
    - `kind` chooses between SIMPLIFIED_FIX_SEQ_BY_LOCATION / ...BY_LABEL

    The interrupted-scan completion map is learned ONCE, over every participant
    in ``all_participants``, and the same map is then applied to each group.
    Diana, 2026-09-25: the completion should use the whole population of the
    dataset. It used to be learned inside the group loop, so hunters and
    gatherers repaired the same short scan differently -- a property of the
    grouping rather than of the behaviour (`todo.md` T3.21 row 5).

    One map per DATASET, not across datasets: learning it over L1 and KnowQA
    together would make Study 1's descriptive numbers depend on Study 2 data.

    ``include_all`` adds the all-participants group. It is on by default because
    the paper reports a dominant-strategy prevalence for the whole sample, and
    that number had never been produced -- this function used to hardcode
    ``include_all=False`` (`findings.md` 1.2).

    Returns nested dict:
        results[group_name] = {
            "df": df_strat_with_completed,
            "prefix_map": prefix_map,
            "dominant_prop": dominant_prop,
            "plots": (dh, cdh, gh, ch, sh),
        }
    """
    results = {}

    # One map for the whole dataset, learned before the group loop so every
    # group repairs short scans the same way.
    population_strat = build_starting_strategies(
        all_participants,
        kind=kind,
        window_len=window_len,
        drop_question=True,
        out_col=Con.STRATEGY_COL,
    )
    population_prefix_map = build_prefix_completion_map(
        population_strat[Con.STRATEGY_COL], full_len=window_len
    )

    groups = split_participant_groups(
        all_participants, split=split_groups, include_all=include_all
    )
    for group_name, df in groups.items():
        df_strat = build_starting_strategies(
            df,
            kind=kind,
            window_len=window_len,
            drop_question=True,
            out_col=Con.STRATEGY_COL,
        )

        dom_prop_raw = proportion_with_dominant_strategy(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            strat_col=Con.STRATEGY_COL,
            threshold=threshold,
        )
        print(
            f"{dom_prop_raw:.1%} of {group_name} participants had a dominant "
            f"strategy (≥{threshold * 100:.0f}% of trials) before completion."
        )

        df_strat, prefix_map = add_completed_strategy_column(
            df_strat,
            strat_col=Con.STRATEGY_COL,
            full_len=window_len,
            col_suffix="_completed",
            prefix2full=population_prefix_map,
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
            f"strategy (≥{threshold * 100:.0f}% of trials) after completion."
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
            to_paper=to_paper,
            density=False,
            full_len=window_len,
        )

        plots = plot_strategies(
            df_strat,
            id_col=Con.PARTICIPANT_ID,
            strat_col=Con.STRATEGY_COL,
            h_or_g=group_name,
            save=save,
            to_paper=to_paper,
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


# ==========================================================================
# from src/viz/visualisations_simplified_visits.py
# ==========================================================================

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
    save: Optional[bool] = None,
    to_paper=None,
    show: bool = True,
    # ---- presentation touch-ups (optional; defaults preserve behavior) ----
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = "g",
    area_label_map: Optional[dict] = None,
    position_labels: Optional[list] = None,
    cbar_label: Optional[str] = None,
    normalize: Optional[str] = None,   # None | "row" | "col" | "all"
    title_fontsize: Optional[float] = None,
    label_fontsize: Optional[float] = None,
    tick_fontsize: Optional[float] = None,
    annot_fontsize: Optional[float] = None,
):
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
        If True, write the figure and its pivot through ``save_output``.
    show : bool
        If True, display the plot; otherwise close it after saving.
    """
    if kind == "label":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
        base_areas = list(Con.LABEL_CHOICES)
    elif kind == "location":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
        base_areas = list(Con.LOC_CHOICES)
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

    # Optional normalization (e.g. row -> share of visits at each position).
    if normalize == "row":
        pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "col":
        pivot = pivot.div(pivot.sum(axis=0).replace(0, np.nan), axis=1)
    elif normalize == "all":
        total = pivot.values.sum()
        if total:
            pivot = pivot / total
    elif normalize is not None:
        raise ValueError("normalize must be one of None, 'row', 'col', 'all'.")

    # Pretty column labels for display (does not affect the returned pivot).
    plot_pivot = pivot.rename(columns=area_label_map) if area_label_map else pivot

    fig = plt.figure(figsize=figsize)
    heatmap_kwargs = dict(annot=annot, fmt=fmt, cmap=cmap)
    if cbar_label is not None:
        heatmap_kwargs["cbar_kws"] = {"label": cbar_label}
    if annot_fontsize is not None:
        heatmap_kwargs["annot_kws"] = {"size": annot_fontsize}
    ax = sns.heatmap(plot_pivot, **heatmap_kwargs)

    if title is None:
        q_flag = " (no question)" if drop_question else " (with question)"
        title = (
            f"{which.capitalize()} {window_len} visits ({kind}){q_flag}\n"
            f"{h_or_g}, selected={selected}"
        )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel if xlabel is not None else "Area", fontsize=label_fontsize)
    ax.set_ylabel(
        ylabel if ylabel is not None else "Visit Order (position)",
        fontsize=label_fontsize,
    )

    if position_labels is not None:
        ax.set_yticklabels(position_labels[: len(plot_pivot.index)], rotation=0)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    plt.tight_layout()

    save_output(
        fig,
        analysis="scan_strategies",
        plot=f"{which}_visits_matrix",
        tables={"matrix": pivot.reset_index()},
        save=save,
        to_paper=to_paper,
        subdir=f"{which}_visits",
        group=h_or_g,
        kind=kind,
        selected=selected,
        questions="removed" if drop_question else "included",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, pivot



def run_all_simplified_visit_matrices(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    drop_question_variants: tuple = (True, False),
    kinds: tuple = ("label", "location"),
    which_list: tuple = ("first", "last"),
    answers: tuple = ("A", "B", "C", "D"),
    save: Optional[bool] = None,
    to_paper=None,
    show: bool = True,
) -> None:
    """
    Generate visit-order heatmaps (first/last visits) for simplified sequences:
      - hunters, gatherers, and all participants
      - each selected answer (A–D)
      - label-based and/or location-based sequences
      - with and without 'question'
    """
    groups = split_participant_groups(all_participants, split=split_groups)

    for drop_question in drop_question_variants:
        for group_key, df in groups.items():
            group_label = "all participants" if group_key == "all_participants" else group_key

            for ans in answers:
                subset = df[df[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans].copy()
                if subset.empty:
                    continue

                print(
                    f"\n{group_label.upper()} — selected {ans} "
                    f"(drop_question={drop_question})"
                )
                print("-" * 72)

                for which in which_list:
                    for kind in kinds:
                        matrix_plot_simplified_visits(
                            subset,
                            kind=kind,
                            which=which,
                            drop_question=drop_question,
                            h_or_g=group_label,
                            selected=ans,
                            figsize=(8, 5),
                            save=save,
                            to_paper=to_paper,
                            show=show,
                        )
