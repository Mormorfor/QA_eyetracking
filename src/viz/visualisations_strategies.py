import os
from typing import Optional
import ast
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants as Con
from src.derived.pattern_breaking import (
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
from src.viz.plot_output import save_fig
from src.viz.viz_helpers import split_participant_groups


# ---------------------------------------------------------------------------
# Dominant Strategies
# ---------------------------------------------------------------------------

# NOTE: build_strategy_dataframe used to live here. It was a byte-identical
# duplicate of derived.pattern_breaking.build_starting_strategies (verified on
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
    save: bool = True,
    output_root: str = "../reports/plots/strategies",
    paper_dirs=None,
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
    if save:
        save_fig(
            fig,
            output_root,
            f"dominant_prop_{strat_col}_{h_or_g}",
            paper_dirs=paper_dirs,
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
    output_root: str = "../reports/plots/strategies",
    paper_dirs=None,
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
    if save:
        save_fig(
            fig,
            output_root,
            f"dominance_gap_{strat_col}_{h_or_g}",
            paper_dirs=paper_dirs,
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
    output_root: str = "../reports/plots/strategies",
    paper_dirs=None,
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
    if save:
        save_fig(
            fig,
            output_root,
            f"dom_str_counts_{strat_col}_{h_or_g}",
            paper_dirs=paper_dirs,
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
    save: bool = True,
    output_root: str = "../reports/plots/strategies",
    paper_dirs=None,
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
    if save:
        save_fig(
            fig,
            output_root,
            f"str_above_thresh_{strat_col}_{h_or_g}",
            paper_dirs=paper_dirs,
        )

    plt.show()
    return freq



# NOTE: build_prefix_completion_map_from_series and add_completed_sequence_column
# moved to derived.pattern_breaking on 2026-09-20 (as build_prefix_completion_map
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
    save: bool = True,
    out_prefix: str = "../reports/plots/strategies",
    paper_dirs=None,
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

    if save:
        save_fig(
            fig,
            out_prefix,
            f"dominant_prop_raw_vs_completed_{h_or_g}",
            paper_dirs=paper_dirs,
        )

    plt.show()

    return summary, both, fig, ax




def plot_strategies(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    h_or_g: str = "hunters",
    save: bool = True,
    output_root: str = "../reports/plots/strategies",
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
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    kind: str = "location",
    window_len: int = 4,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    output_root: str = "../reports/plots/strategies",
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

    groups = split_participant_groups(
        all_participants, split=split_groups, include_all=False
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
