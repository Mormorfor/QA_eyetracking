"""Figures for the opening scan: the first four answers a participant visits,
repeats included.

Backs the paper's First-scan behavior subsection. `run_all_strategy_plots` gives,
for each group, the share of participants with a dominant strategy (their most
common opening scan covers at least half their trials), before and after
interrupted scans are completed, and draws the dominance figures.
`run_all_simplified_visit_matrices` draws heatmaps of which area is at each of a
trial's first (or last) visits.

The numbers are computed in `features/strategies.py`; this module draws and
saves them, under `reports/scan_strategies/`. A function that takes a `df` with a
strategy column expects one row per trial, as `build_starting_strategies` returns
it. The terms are explained in `glossary.md` §8."""

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


# ---------------------------------------------------------------------------
# Dominant strategies
# ---------------------------------------------------------------------------


def proportion_with_dominant_strategy(
    df: pd.DataFrame,
    id_col: str = Con.PARTICIPANT_ID,
    strat_col: str = Con.STRATEGY_COL,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> float:
    """
    Share of participants with a dominant strategy: their most common strategy
    covers at least `threshold` of their trials.
    """
    dominant = dominant_strategy_by_participant(
        df, id_col=id_col, strat_col=strat_col
    )
    is_dominant = has_dominant_strategy(
        dominant[Con.DOMINANCE_SCORE], threshold=threshold
    )
    return float(is_dominant.mean())



def _group_label(h_or_g: str) -> str:
    """"all_participants" -> "all participants", for figure titles."""
    return str(h_or_g).replace("_", " ")


def _most_common_axes(ax, threshold: float) -> None:
    """The shared x-axis of the most-common-strategy histograms: the share of a
    participant's trials that use their most common scan, with a line where that
    share makes it a dominant strategy."""
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"dominant strategy (≥ {threshold:.0%})")
    ax.set_xlim(0, 1)
    ticks = np.linspace(0, 1, 11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Share of the participant's trials that use their most common scan")


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
    Histogram of the dominance score: for each participant, the share of their
    trials that use their most common strategy.

    `bins` is a number of equal-width bins from 0 to 1, or the bin edges.
    `strat_col` goes into the title and file name, which keeps the raw and the
    completed versions apart. Returns the scores.

    `completed_flag_col` is accepted but not used. The saved table holds the
    scores without participant ids.
    """
    dominant_prop = dominant_strategy_by_participant(
        df, id_col=id_col, strat_col=strat_col
    )[Con.DOMINANCE_SCORE]

    if isinstance(bins, int):
        bin_edges = np.linspace(0, 1, bins + 1)
    else:
        bin_edges = np.asarray(bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(dominant_prop, bins=bin_edges, edgecolor="k")
    _most_common_axes(ax, DEFAULT_DOMINANCE_THRESHOLD)
    ax.set_ylabel("Number of participants")
    scans = "completed" if str(strat_col).endswith("_completed") else "raw"
    ax.set_title(f"Most common strategy proportion ({_group_label(h_or_g)}, {scans} scans)")
    ax.legend()

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
    How far each participant's most common strategy is ahead of their second.

    P1 is the share of their trials on the most common strategy, P2 the share
    on the second (0 if they used only one), and the gap is P1 - P2. Left: a
    histogram of the gaps. Right: P1 against P2, one point per participant,
    with the line P1 = P2. Saves and returns the per-participant table.
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
    Histogram of how many different strategies each participant used.

    It shows that "dominant" means most common, not the only one: a person
    with a dominant strategy can still have used many others. Saves and
    returns the per-participant counts.
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
    Bar chart: among participants with a dominant strategy (at least
    `threshold` of their trials), how many have each strategy as their
    dominant one. Saves and returns the counts.
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
    Compare each participant's dominance score before and after interrupted-scan
    completion, as two overlapping histograms.

    `df` has one row per trial, with the raw strategies in `raw_col` and the
    completed ones in `comp_col` (default `<raw_col>_completed`, the name
    `add_completed_strategy_column` gives them). `threshold` decides what
    counts as a dominant strategy in the summary, `full_len` is the length of a
    complete scan, and `density` plots densities instead of counts.

    Saves the figure with both tables. Returns `(summary, both, fig, ax)`, the
    first two from `summarize_completion_effect`: `summary` is a dict (average
    scores, the share with a dominant strategy before and after, how many
    participants' dominant strategy changed); `both` has one row per
    participant, with both scores, their difference, both dominant strategies,
    and how many of the person's trials completion changed.
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
        label="Raw scans",
        **hist_kwargs,
    )
    ax.hist(
        both["comp"],
        bins=bin_edges,
        density=density,
        label="Completed scans",
        **hist_kwargs,
    )
    _most_common_axes(ax, threshold)
    ax.set_ylabel("Density" if density else "Number of participants")
    ax.set_title(
        f"Most common strategy proportion, raw vs completed ({_group_label(h_or_g)})"
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
    Draw the dominance figures for one group, in this order: the dominance-score
    histogram, raw and then completed; the dominance gap and the number of
    strategies per participant, both raw; and which strategies are dominant,
    completed.

    `df` has one row per trial, with `strat_col` and `<strat_col>_completed`.
    The last chart always uses a threshold of 0.5, even if
    `run_all_strategy_plots` was given another one.

    Returns the five results in the same order.
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
    Build the opening scans and draw every dominance figure, for hunters,
    gatherers and all participants together.

    The scan is the first `window_len` answer visits of the merged sequence,
    with the question's entries removed; `kind` picks the sequence by place
    ("location", the default) or by meaning ("label"). For each group it prints
    the share of participants with a dominant strategy (at least `threshold` of
    their trials) before and after completion, then runs `summarize_before_after`
    and `plot_strategies`.

    The completion map is learned once, from every trial in `all_participants`,
    and used for every group, so a short scan is completed the same way
    whichever group the person is in. Pass one dataset at a time: a map learned
    over Study 1 and Study 2 together would make Study 1's numbers depend on
    Study 2's data.

    `split_groups=False` gives only the all-participants group, and
    `include_all=False` leaves it out.

    Returns `{group: {...}}`, each with: `df` (the per-trial strategies, raw and
    completed), `prefix_map` (the shared completion map), `dominant_prop_raw` and
    `dominant_prop_completed`; `before_after_summary`, `before_after_table`,
    `before_after_fig` and `before_after_ax` (from `summarize_before_after`); and
    `plots` (the five results of `plot_strategies`).
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


# ---------------------------------------------------------------------------
# First and last visits heatmaps
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
    # ---- styling (optional) ----
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
    Heatmap of which area is at each of a trial's first (or last) visits.

    For each trial: take the merged sequence (by meaning with `kind="label"`,
    by place with `kind="location"`), remove the question's entries if
    `drop_question` is set, and keep the first or last 4 entries (5 when the
    question is kept), as `which` says. The matrix counts, over trials, how
    often each area is at each position: rows are positions 0-3 (or 0-4),
    columns are areas. First visits count from the start, so row 0 is the
    first visit; last visits count back from the end, so the bottom row is the
    final visit, also in a trial with fewer visits than the window. Removing the
    question does not merge the entries on either side of it, so
    (A, question, A) becomes (A, A).

    `df` may have several rows per trial (e.g. the word table); it is reduced to
    one row per trial. The caller filters it to one group and one chosen
    answer; `h_or_g` and `selected` only label the figure and the file name.
    `normalize` turns the counts into shares of each position ("row"), of each
    area ("col") or of all ("all"). The remaining arguments are styling, and
    `show=False` closes the figure instead of showing it.

    Trials whose sequence is missing or cannot be read, or is empty once the
    question is removed, are left out without a message. If no trial is left,
    it prints a note and returns None.

    Saves the figure and the matrix (after `normalize`). Returns
    `(fig, ax, matrix)`.
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

    # The row a trial's window starts on: row 0 for first visits. For last visits
    # the window ends on the last row, so the final visit is always the bottom row,
    # even in a trial with fewer than `window_len` visits.
    def _start(lst):
        return 0 if which == "first" else window_len - len(lst)

    df_sel["position"] = df_sel["window"].apply(
        lambda lst: list(range(_start(lst), _start(lst) + len(lst)))
    )

    df_expl = df_sel.explode("position")
    df_expl = df_expl[df_expl["position"].notna()].copy()

    df_expl["area"] = df_expl.apply(
        lambda row: row["window"][int(row["position"]) - _start(row["window"])],
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
    Draw every visit heatmap: for each group (hunters, gatherers, all
    participants), each chosen answer (A-D), first and last visits, by meaning
    and by place, and with the question removed and kept -- 96 heatmaps with
    the defaults.

    A chosen answer nobody in a group picked is skipped. Returns nothing; each
    heatmap and its matrix are saved by `matrix_plot_simplified_visits`.
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
