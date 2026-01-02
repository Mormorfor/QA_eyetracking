import os
from typing import Optional
import ast
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants as Con


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
    output_root: str = "../reports/plots/strategies",
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
                f"dominant_prop_{strat_col}_{h_or_g}.png",
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
    output_root: str = "../reports/plots/strategies",
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
    output_root: str = "../reports/plots/strategies",
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
    output_root: str = "../reports/plots/strategies",
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
    out_prefix: str = "../reports/plots/strategies",
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
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    kind: str = "location",
    window_len: int = 4,
    threshold: float = 0.5,
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
