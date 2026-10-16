"""Figures: accuracy against simple measures of how a trial's answers were read.

Seven families of figures, each saved with the numbers behind it under
reports/correctness_associations/, one subfolder per family:

    correctness_by_seq_len_threshold            bars: at most N visits vs more
    correctness_by_seq_len_continuous           line: accuracy at each number of visits
    correctness_by_back_and_forth               bars: ends in a back-and-forth vs not
    correctness_by_trial_mean_dwell_threshold   bars: mean dwell per word, low vs high
    correctness_by_trial_mean_dwell_continuous  line: accuracy by mean dwell per word
    correctness_by_total_answering_rt           line: accuracy by time to answer
    correctness_by_matching                     bars: chose the gaze-preferred answer vs not

Each has a `run_all_*` function that draws it for hunters, gatherers and all
participants. Every figure takes the word-level table and collapses it to one row
per trial before counting. Intervals are 95% Wilson intervals. The bar charts
carry a Fisher test from `stats.py`; its p-values are too small, because it
treats trials as independent (see that module).
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.config import columns as Con
from src.analyses.correctness_associations.compute import (
    compute_seq_len_threshold_summary,
    compute_back_and_forth_pattern_summary,
    compute_trial_mean_dwell_threshold_summary,
    wilson_ci,
)
from src.lib.plotting.annotate import (
    add_significance_bracket,
    add_wilson_errorbars_and_ns,
    barplot_accuracy,
)
from src.lib.plotting.tables import correctness_tables
from src.lib.stats.proportions import p_to_stars
from src.features.scope import split_participant_groups
from src.lib.plotting.output import save_output
from pathlib import Path
from typing import Optional, Dict, List, Sequence
from src.config import columns as C
from src.analyses.correctness_associations.stats import correctness_by_pref_group_test
from src.features import preference as PM
from src.lib.plotting.annotate import (
    add_wilson_errorbars_and_ns,
    barplot_accuracy,
)
from src.analyses.correctness_associations.compute import summarize_binary_by_group


ANALYSIS = "correctness_associations"

# ------------------------------------------
# Sequence Length Measures
# ------------------------------------------


def plot_correctness_by_sequence_len_threshold(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    add_significance: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, Optional[Dict]]:
    """Bar chart: accuracy of trials with at most `threshold` visits vs more.

    Bars show 95% Wilson intervals and n; with `add_significance`, a bracket
    carries the Fisher test's stars. Saves the figure, the summary and the Fisher
    result. Returns (fig, summary, Fisher result or None).
    """
    summary_df, test_res = compute_seq_len_threshold_summary(
        df=df,
        threshold=int(threshold),
        seq_col=seq_col,
        correct_col=correct_col,
        add_significance=add_significance,
    )

    order = [f"≤ {threshold}", f"> {threshold}"]
    fig, ax = barplot_accuracy(summary_df, order=order, figsize=figsize)
    add_wilson_errorbars_and_ns(ax, summary_df)

    default_title = f"{h_or_g}: Correctness by sequence length (threshold={threshold})"
    ax.set_title(title or default_title)
    ax.set_xlabel("Sequence length bin")
    ax.set_ylabel("Correctness rate")

    if add_significance and test_res is not None:
        stars = p_to_stars(test_res.get("p_value"))
        add_significance_bracket(ax, stars, x1=0, x2=1)

    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_seq_len_threshold",
        tables=correctness_tables(summary_df, test_res),
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_seq_len_threshold",
        group=h_or_g,
        threshold=threshold,
    )

    return fig, summary_df, test_res


def run_all_correctness_seq_len_threshold_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    thresholds: Tuple[int, ...] = (2, 3, 4, 5),
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Dict[str, Dict[int, Dict[str, object]]]:
    """Draw `plot_correctness_by_sequence_len_threshold` for each threshold and each
    group: hunters, gatherers and all participants, or only all participants if
    `split_groups=False`.

    Returns results[group][threshold] = {"fig", "summary", "test"}.
    """

    def _run_for_group(
        df: pd.DataFrame, group_name: str
    ) -> Dict[int, Dict[str, object]]:
        group_results: Dict[int, Dict[str, object]] = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_sequence_len_threshold(
                df=df,
                threshold=int(t),
                seq_col=seq_col,
                correct_col=correct_col,
                h_or_g=group_name,
                save=save,
                to_paper=to_paper,
                add_significance=add_significance,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)
                if test_res is not None and test_res.get("p_value") is not None:
                    print(f"p={test_res.get('p_value')}")

            group_results[int(t)] = {"fig": fig, "summary": summary, "test": test_res}

        return group_results

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


def plot_correctness_by_sequence_len_continuous(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (7, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    max_len: Optional[int] = None,
    min_n_per_len: int = 5,
    show_ci: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Line plot: accuracy at each number of visits, with a 95% Wilson band.

    Two exclusions, both set by the caller: trials with more than `max_len` visits
    are left out, and so is any number of visits with fewer than `min_n_per_len`
    trials, from the figure and the saved table alike. Saves the figure and the
    summary (seq_len, n, k_correct, accuracy, ci_low, ci_high). Returns
    (fig, summary).
    """
    # Visits per trial, by the same rule as `features.sequences.sequence_len_literal_eval`:
    # a missing or unreadable sequence counts as 0.
    import ast
    import numpy as np

    def _seq_len(x) -> int:
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return 0
        return len(x) if isinstance(x, (list, tuple)) else 0

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq_len"] = d[seq_col].apply(_seq_len)

    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        seq_len=("_seq_len", "first"), is_correct=(correct_col, "first")
    )

    if max_len is not None:
        trial_df = trial_df[trial_df["seq_len"] <= int(max_len)].copy()

    agg = (
        trial_df.groupby("seq_len", as_index=False)
        .agg(
            n=("is_correct", "size"),
            k_correct=("is_correct", "sum"),
        )
        .sort_values("seq_len")
        .reset_index(drop=True)
    )

    # Leave out lengths with too few trials.
    if min_n_per_len is not None and min_n_per_len > 1:
        agg = agg[agg["n"] >= int(min_n_per_len)].copy()

    # Accuracy and its Wilson interval
    agg["accuracy"] = agg["k_correct"] / agg["n"]
    cis = agg.apply(lambda r: wilson_ci(int(r["k_correct"]), int(r["n"])), axis=1)
    agg["ci_low"] = [c[0] for c in cis]
    agg["ci_high"] = [c[1] for c in cis]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(agg["seq_len"], agg["accuracy"], marker="o")

    # The Wilson interval as a band
    if show_ci and len(agg) > 0 and agg["ci_low"].notna().any():
        ax.fill_between(
            agg["seq_len"].to_numpy(),
            agg["ci_low"].to_numpy(),
            agg["ci_high"].to_numpy(),
            alpha=0.2,
        )

    default_title = f"{h_or_g}: Correctness by sequence length (continuous)"
    default_x_label = "Sequence length"
    default_y_label = "Correctness rate"

    ax.set_title(title or default_title)
    ax.set_xlabel(x_label or default_x_label)
    ax.set_ylabel(y_label or default_y_label)

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_seq_len_continuous",
        tables={"summary": agg},
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_seq_len_continuous",
        group=h_or_g,
        maxlen=max_len if max_len is not None else "all",
        minn=min_n_per_len,
    )

    return fig, agg


def run_all_correctness_seq_len_continuous_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    max_len: Optional[int] = None,
    min_n_per_len: int = 5,
    show_ci: bool = True,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Draw `plot_correctness_by_sequence_len_continuous` for each group: hunters,
    gatherers and all participants, or only all participants if
    `split_groups=False`.

    Returns results[group] = {"fig", "summary"}.
    """

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[str, object]:
        fig, summary = plot_correctness_by_sequence_len_continuous(
            df=df,
            seq_col=seq_col,
            correct_col=correct_col,
            h_or_g=group_name,
            save=save,
            to_paper=to_paper,
            max_len=max_len,
            min_n_per_len=min_n_per_len,
            show_ci=show_ci,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        if print_summaries:
            print(f"\n=== {group_name.upper()} — continuous seq_len ===")
            print(summary.head(15))
            if len(summary) > 0:
                print(f"... ({len(summary)} lengths total)")

        return {"fig": fig, "summary": summary}

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


# ------------------------------------------
# XYX / XYXY detection
# ------------------------------------------


def plot_correctness_by_back_and_forth_pattern(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    use_xyxy: bool = False,
    add_significance: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, Optional[Dict]]:
    """Bar chart: accuracy of trials whose sequence of visits ends in a back-and-forth
    vs not.

    XYX (default): the last three visits are x, y, x. XYXY (`use_xyxy=True`): the
    last four are x, y, x, y. Any area counts, the question included, and a
    sequence that will not parse counts as "absent". Bars show 95% Wilson intervals
    and n; with `add_significance`, a bracket carries the Fisher test's stars.
    Saves the figure, the summary and the Fisher result. Returns
    (fig, summary, Fisher result or None).
    """
    summary_df, test_res, pattern_name = compute_back_and_forth_pattern_summary(
        df=df,
        seq_col=seq_col,
        correct_col=correct_col,
        use_xyxy=use_xyxy,
        add_significance=add_significance,
    )

    order = [f"{pattern_name} absent", f"{pattern_name} present"]
    fig, ax = barplot_accuracy(summary_df, order=order, figsize=figsize)
    add_wilson_errorbars_and_ns(ax, summary_df)

    default_title = f"{h_or_g}: Correctness by end-pattern ({pattern_name})"
    ax.set_title(title or default_title)
    ax.set_xlabel("Pattern bin")
    ax.set_ylabel("Correctness rate")

    if add_significance and test_res is not None:
        stars = p_to_stars(test_res.get("p_value"))
        add_significance_bracket(ax, stars, x1=0, x2=1)

    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_back_and_forth",
        tables=correctness_tables(summary_df, test_res),
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_back_and_forth",
        group=h_or_g,
        pattern="xyxy" if use_xyxy else "xyx",
    )

    return fig, summary_df, test_res


def run_all_back_and_forth_pattern_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    use_xyxy: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Draw `plot_correctness_by_back_and_forth_pattern` for each group: hunters,
    gatherers and all participants, or only all participants if
    `split_groups=False`.

    Returns results[group] = {"fig", "summary", "test"}.
    """

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[str, object]:
        fig, summary, test_res = plot_correctness_by_back_and_forth_pattern(
            df=df,
            seq_col=seq_col,
            correct_col=correct_col,
            h_or_g=group_name,
            save=save,
            to_paper=to_paper,
            use_xyxy=use_xyxy,
            add_significance=add_significance,
        )

        if print_summaries:
            print(
                f"\n=== {group_name.upper()} — pattern: {'XYXY' if use_xyxy else 'XYX'} ==="
            )
            print(summary)
            if test_res is not None and test_res.get("p_value") is not None:
                print(f"p={test_res.get('p_value')}")

        return {"fig": fig, "summary": summary, "test": test_res}

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


# ------------------------------------------
# Mean dwell per word
# ------------------------------------------


def plot_correctness_by_trial_mean_dwell_threshold(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    add_significance: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, Optional[Dict]]:
    """Bar chart: accuracy of trials whose mean dwell time per word, over the whole
    answer screen, is at most `threshold` ms vs more.

    Bars show 95% Wilson intervals and n; with `add_significance`, a bracket
    carries the Fisher test's stars. Saves the figure, the summary and the Fisher
    result. Returns (fig, summary, Fisher result or None).
    """
    summary_df, test_res = compute_trial_mean_dwell_threshold_summary(
        df=df,
        threshold=float(threshold),
        dwell_col=dwell_col,
        correct_col=correct_col,
        add_significance=add_significance,
    )

    order = [f"≤ {threshold}", f"> {threshold}"]
    fig, ax = barplot_accuracy(summary_df, order=order, figsize=figsize)
    add_wilson_errorbars_and_ns(ax, summary_df)

    default_title = (
        f"{h_or_g}: Correctness by whole-trial mean dwell per word "
        f"(threshold={threshold})"
    )
    ax.set_title(title or default_title)
    ax.set_xlabel("Trial mean dwell per word bin")
    ax.set_ylabel("Correctness rate")

    if add_significance and test_res is not None:
        stars = p_to_stars(test_res.get("p_value"))
        add_significance_bracket(ax, stars, x1=0, x2=1)

    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_trial_mean_dwell_threshold",
        tables=correctness_tables(summary_df, test_res),
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_trial_mean_dwell_threshold",
        group=h_or_g,
        threshold=threshold,
    )

    return fig, summary_df, test_res


def run_all_trial_mean_dwell_threshold_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    thresholds: Tuple[float, ...] = (50.0, 75.0, 100.0),
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Dict[str, Dict[float, Dict[str, object]]]:
    """Draw `plot_correctness_by_trial_mean_dwell_threshold` for each threshold (in
    ms) and each group: hunters, gatherers and all participants, or only all
    participants if `split_groups=False`.

    Returns results[group][threshold] = {"fig", "summary", "test"}.
    """

    def _run_for_group(
        df: pd.DataFrame, group_name: str
    ) -> Dict[float, Dict[str, object]]:
        group_results: Dict[float, Dict[str, object]] = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_trial_mean_dwell_threshold(
                df=df,
                threshold=float(t),
                dwell_col=dwell_col,
                correct_col=correct_col,
                h_or_g=group_name,
                save=save,
                to_paper=to_paper,
                add_significance=add_significance,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)
                if test_res is not None and test_res.get("p_value") is not None:
                    print(f"p={test_res.get('p_value')}")

            group_results[float(t)] = {"fig": fig, "summary": summary, "test": test_res}

        return group_results

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


def plot_correctness_by_trial_mean_dwell_continuous(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (7, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    bin_width: Optional[float] = 10.0,
    n_bins: Optional[int] = None,
    min_n_per_bin: int = 10,
    x_max: Optional[float] = None,
    show_ci: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Line plot: accuracy against whole-trial mean dwell time per word, in bins,
    with a 95% Wilson band.

    The bins are `n_bins` equal bins from the lowest value to the highest if
    `n_bins` is 2 or more, otherwise bins `bin_width` ms wide (default 10). Trials
    above `x_max` are left out first, and bins with fewer than `min_n_per_bin`
    trials are left out of the figure and the saved table.

    A bin holds its left edge but not its right one, so a trial exactly on the last
    edge is in no bin and is dropped without a message. With `n_bins` that is
    always the trial (or trials) with the highest value; with `bin_width`, only
    when the highest value is a multiple of the width.

    Saves the figure and the summary (bin_left, bin_right, bin_center, n,
    k_correct, accuracy, ci_low, ci_high). If no trials are left, returns an empty
    figure and table and saves nothing. Returns (fig, summary).
    """

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)

    # Mean dwell per word over the trial, as in
    # `features.sequences.compute_trial_mean_dwell_per_word`.
    total_dwell = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "sum"
    )
    n_words = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "count"
    )
    d["_trial_mean_dwell"] = total_dwell / n_words

    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        trial_mean_dwell=("_trial_mean_dwell", "first"),
        is_correct=(correct_col, "first"),
    )

    if x_max is not None:
        trial_df = trial_df[trial_df["trial_mean_dwell"] <= float(x_max)].copy()

    x = trial_df["trial_mean_dwell"].to_numpy()
    if len(x) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(
            title or f"{h_or_g}: Correctness by mean dwell per word (continuous)"
        )
        ax.set_xlabel("Trial mean dwell per word")
        ax.set_ylabel("Correctness rate")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig, pd.DataFrame(
            columns=[
                "bin_left",
                "bin_right",
                "bin_center",
                "n",
                "k_correct",
                "accuracy",
                "ci_low",
                "ci_high",
            ]
        )

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))

    if n_bins is not None and n_bins >= 2:
        edges = np.linspace(xmin, xmax, int(n_bins) + 1)
    else:
        bw = float(bin_width) if bin_width is not None else 10.0
        start = np.floor(xmin / bw) * bw
        end = np.ceil(xmax / bw) * bw
        if end == start:
            end = start + bw
        edges = np.arange(start, end + bw, bw)

        if len(edges) < 3:
            edges = np.array([start, start + bw, start + 2 * bw], dtype=float)

    # Each trial's bin; a bin holds its left edge, not its right one.
    bin_idx = (
        np.digitize(trial_df["trial_mean_dwell"].to_numpy(), edges, right=False) - 1
    )
    # A trial outside every bin -- including one exactly on the last edge -- is
    # dropped here.
    valid = (bin_idx >= 0) & (bin_idx < len(edges) - 1)
    trial_df = trial_df.loc[valid].copy()
    trial_df["_bin"] = bin_idx[valid]

    agg = (
        trial_df.groupby("_bin", as_index=False)
        .agg(
            n=("is_correct", "size"),
            k_correct=("is_correct", "sum"),
        )
        .sort_values("_bin")
        .reset_index(drop=True)
    )

    agg["bin_left"] = agg["_bin"].apply(lambda i: float(edges[int(i)]))
    agg["bin_right"] = agg["_bin"].apply(lambda i: float(edges[int(i) + 1]))
    agg["bin_center"] = (agg["bin_left"] + agg["bin_right"]) / 2.0

    if min_n_per_bin is not None and min_n_per_bin > 1:
        agg = agg[agg["n"] >= int(min_n_per_bin)].copy()

    agg["accuracy"] = agg["k_correct"] / agg["n"]

    cis = agg.apply(lambda r: wilson_ci(int(r["k_correct"]), int(r["n"])), axis=1)
    agg["ci_low"] = [c[0] for c in cis]
    agg["ci_high"] = [c[1] for c in cis]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(agg["bin_center"], agg["accuracy"], marker="o")

    if show_ci and len(agg) > 0 and agg["ci_low"].notna().any():
        ax.fill_between(
            agg["bin_center"].to_numpy(),
            agg["ci_low"].to_numpy(),
            agg["ci_high"].to_numpy(),
            alpha=0.2,
        )

    default_title = f"{h_or_g}: Correctness by mean dwell per word (continuous)"
    ax.set_title(title or default_title)
    ax.set_xlabel("Trial mean dwell per word (binned)")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_trial_mean_dwell_continuous",
        tables={"summary": agg},
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_trial_mean_dwell_continuous",
        group=h_or_g,
        nbins=int(n_bins) if n_bins is not None else None,
        bw=bin_width if n_bins is None else None,
        xmax=x_max,
        minn=min_n_per_bin,
    )

    return fig, agg


def run_all_trial_mean_dwell_continuous_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    bin_width: Optional[float] = 10.0,
    n_bins: Optional[int] = None,
    min_n_per_bin: int = 10,
    x_max: Optional[float] = None,
    show_ci: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Draw `plot_correctness_by_trial_mean_dwell_continuous` for each group:
    hunters, gatherers and all participants, or only all participants if
    `split_groups=False`.

    Returns results[group] = {"fig", "summary"}.
    """

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[str, object]:
        fig, summary = plot_correctness_by_trial_mean_dwell_continuous(
            df=df,
            dwell_col=dwell_col,
            correct_col=correct_col,
            h_or_g=group_name,
            save=save,
            to_paper=to_paper,
            bin_width=bin_width,
            n_bins=n_bins,
            min_n_per_bin=min_n_per_bin,
            x_max=x_max,
            show_ci=show_ci,
        )

        if print_summaries:
            print(f"\n=== {group_name.upper()} — continuous mean dwell ===")
            print(summary.head(10))
            if len(summary) > 0:
                print(f"... ({len(summary)} bins total)")

        return {"fig": fig, "summary": summary}

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


# ------------------------------------------
# Time to answer
# ------------------------------------------


def plot_correctness_by_total_answering_rt_continuous(
    df: pd.DataFrame,
    rt_col: str = Con.CONFIRM_FINAL_ANSWER_RT,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (7, 4),
    save: Optional[bool] = None,
    h_or_g: str = "hunters",
    to_paper=None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    bin_width: Optional[float] = None,
    n_bins: Optional[int] = 10,
    min_n_per_bin: int = 10,
    x_max: Optional[float] = None,
    show_ci: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Line plot: accuracy against the time taken to answer, in bins, with a 95%
    Wilson band.

    The time is `rt_col`, by default the tracker's time from the answers appearing
    to the confirming press, in ms. It is divided by 1000 and shown in seconds, so
    the column should be in ms. The bins are `bin_width` seconds wide if that is
    given, otherwise `n_bins` equal bins (default 10) from the shortest time to the
    longest; a time exactly on the last edge goes in the last bin. Trials above
    `x_max` seconds are left out first, and bins with fewer than `min_n_per_bin`
    trials are left out of the figure and the saved table. Trials whose time or
    correctness is missing or not a number are dropped without a message.

    Saves the figure and the summary (bin_left, bin_right, bin_center, n,
    k_correct, accuracy, ci_low, ci_high), with the RT column in the file name. If
    no trials are left, returns an empty figure and table and saves nothing.
    Returns (fig, summary).
    """
    required_cols = [Con.TRIAL_ID, Con.PARTICIPANT_ID, rt_col, correct_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    d = df[required_cols].copy()
    d[rt_col] = pd.to_numeric(d[rt_col], errors="coerce")
    d[correct_col] = pd.to_numeric(d[correct_col], errors="coerce")
    d = d.dropna(subset=[rt_col, correct_col]).copy()
    d[correct_col] = d[correct_col].astype(int)

    # The word-level table repeats each trial's time on every word row, so collapse
    # to one row per trial before counting.
    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        total_answering_rt=(rt_col, "first"),
        is_correct=(correct_col, "first"),
    )

    trial_df["total_answering_rt_sec"] = trial_df["total_answering_rt"] / 1000

    if x_max is not None:
        trial_df = trial_df[trial_df["total_answering_rt_sec"] <= float(x_max)].copy()

    x = trial_df["total_answering_rt_sec"].to_numpy()

    if len(x) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"{h_or_g}: Correctness by total answering RT")
        ax.set_xlabel("Total answering time (s)")
        ax.set_ylabel("Correctness rate")
        ax.set_ylim(0, 1)
        fig.tight_layout()

        empty_summary = pd.DataFrame(
            columns=[
                "bin_left",
                "bin_right",
                "bin_center",
                "n",
                "k_correct",
                "accuracy",
                "ci_low",
                "ci_high",
            ]
        )
        return fig, empty_summary

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))

    if bin_width is not None:
        bw = float(bin_width)
        start = np.floor(xmin / bw) * bw
        end = np.ceil(xmax / bw) * bw

        if end == start:
            end = start + bw

        edges = np.arange(start, end + bw, bw)

    else:
        nb = int(n_bins) if n_bins is not None else 10
        nb = max(nb, 2)
        edges = np.linspace(xmin, xmax, nb + 1)

    # If all RT values are identical, create one valid bin around that value.
    if len(np.unique(edges)) < 2:
        edges = np.array([xmin - 0.5, xmax + 0.5], dtype=float)

    bin_idx = (
        np.digitize(
            trial_df["total_answering_rt_sec"].to_numpy(),
            edges,
            right=False,
        )
        - 1
    )

    # Include values that fall exactly on the rightmost edge in the final bin.
    bin_idx[bin_idx == len(edges) - 1] = len(edges) - 2

    valid = (bin_idx >= 0) & (bin_idx < len(edges) - 1)
    trial_df = trial_df.loc[valid].copy()
    trial_df["_bin"] = bin_idx[valid]

    agg = (
        trial_df.groupby("_bin", as_index=False)
        .agg(
            n=("is_correct", "size"),
            k_correct=("is_correct", "sum"),
        )
        .sort_values("_bin")
        .reset_index(drop=True)
    )

    agg["bin_left"] = agg["_bin"].apply(lambda i: float(edges[int(i)]))
    agg["bin_right"] = agg["_bin"].apply(lambda i: float(edges[int(i) + 1]))
    agg["bin_center"] = (agg["bin_left"] + agg["bin_right"]) / 2.0

    if min_n_per_bin is not None and min_n_per_bin > 1:
        agg = agg[agg["n"] >= int(min_n_per_bin)].copy()

    agg["accuracy"] = agg["k_correct"] / agg["n"]

    cis = agg.apply(
        lambda r: wilson_ci(int(r["k_correct"]), int(r["n"])),
        axis=1,
    )
    agg["ci_low"] = [c[0] for c in cis]
    agg["ci_high"] = [c[1] for c in cis]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        agg["bin_center"],
        agg["accuracy"],
        marker="o",
    )

    if show_ci and len(agg) > 0 and agg["ci_low"].notna().any():
        ax.fill_between(
            agg["bin_center"].to_numpy(),
            agg["ci_low"].to_numpy(),
            agg["ci_high"].to_numpy(),
            alpha=0.2,
        )

    default_title = f"{h_or_g}: Correctness by total answering RT"
    default_x_label = "Total answering time (s)"
    default_y_label = "Correctness rate"

    ax.set_title(title or default_title)
    ax.set_xlabel(x_label or default_x_label)
    ax.set_ylabel(y_label or default_y_label)

    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="correctness_by_total_answering_rt",
        tables={"summary": agg},
        save=save,
        to_paper=to_paper,
        subdir="correctness_by_total_answering_rt",
        group=h_or_g,
        # The RT column is in the file name on purpose: the raw time (ms) and the
        # length-normalized one (ms per word on screen) are different measures, and
        # without this tag a run with one would silently overwrite the other's files.
        rt=rt_col,
        bw=bin_width,
        nbins=(int(n_bins) if n_bins is not None else 10) if bin_width is None else None,
        xmax=x_max,
        minn=min_n_per_bin,
    )

    return (
        fig,
        agg[
            [
                "bin_left",
                "bin_right",
                "bin_center",
                "n",
                "k_correct",
                "accuracy",
                "ci_low",
                "ci_high",
            ]
        ],
    )


def run_all_total_answering_rt_continuous_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
    rt_col: str = Con.CONFIRM_FINAL_ANSWER_RT,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    bin_width: Optional[float] = None,
    n_bins: Optional[int] = 10,
    min_n_per_bin: int = 10,
    x_max: Optional[float] = None,
    show_ci: bool = True,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Draw `plot_correctness_by_total_answering_rt_continuous` for each group:
    hunters, gatherers and all participants, or only all participants if
    `split_groups=False`.

    Returns results[group] = {"fig", "summary"}.
    """

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[str, object]:
        fig, summary = plot_correctness_by_total_answering_rt_continuous(
            df=df,
            rt_col=rt_col,
            correct_col=correct_col,
            h_or_g=group_name,
            save=save,
            to_paper=to_paper,
            bin_width=bin_width,
            n_bins=n_bins,
            min_n_per_bin=min_n_per_bin,
            x_max=x_max,
            show_ci=show_ci,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        if print_summaries:
            print(f"\n=== {group_name.upper()} — continuous total answering RT ===")
            print(summary.head(10))

            if len(summary) > 0:
                print(f"... ({len(summary)} bins total)")

        return {
            "fig": fig,
            "summary": summary,
        }

    groups = split_participant_groups(all_participants, split=split_groups)

    return {name: _run_for_group(df, name) for name, df in groups.items()}


# ------------------------------------------
# Preference matching
# ------------------------------------------

def summarize_correctness_by_pref_group(
    df: pd.DataFrame,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
    group_order: Sequence[str] = ("matching", "not_matching"),
) -> pd.DataFrame:
    """Accuracy of matching vs not-matching trials, with 95% Wilson intervals.

    Expects one row per trial, as `features.preference.compute_trial_matching`
    returns it. Trials with no group or no correctness value are left out, and so
    are groups not in `group_order`. Returns pref_group, n_trials, n_correct, acc,
    ci_low, ci_high.
    """
    sub = df[[pref_col, correct_col]].copy()
    sub = sub.dropna(subset=[pref_col, correct_col])
    sub[correct_col] = sub[correct_col].astype(int)

    # Fixed order, matching first.
    sub[pref_col] = pd.Categorical(sub[pref_col], categories=list(group_order), ordered=True)

    tmp = summarize_binary_by_group(
        trial_df=sub.rename(columns={pref_col: "group", correct_col: "is_correct"}),
        group_col="group",
        outcome_col="is_correct",
    ).sort_values("group")

    # The column names of the saved preference tables.
    out = tmp.rename(
        columns={
            "group": "pref_group",
            "n": "n_trials",
            "k_correct": "n_correct",
            "accuracy": "acc",
        }
    )

    return out[["pref_group", "n_trials", "n_correct", "acc", "ci_low", "ci_high"]].reset_index(drop=True)


def plot_correctness_by_matching(
    df: pd.DataFrame,
    metric_name: str,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
    group_order: Sequence[str] = ("matching", "not_matching"),
    title: Optional[str] = None,
    show_n: bool = True,
    show_test: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    h_or_g: str = "all_participants",
    mode: str = "polarity",
) -> pd.DataFrame:
    """Bar chart: accuracy of trials where the chosen answer was the gaze-preferred
    one vs not, with 95% Wilson intervals.

    `df` is one row per trial with `pref_group` already set (from
    `features.preference.compute_trial_matching`); `metric_name` and `mode` only
    label the figure and its file name. With `show_test`, the Fisher p-value and
    odds ratio are written at the top of the plot.

    Saves the figure and the summary. The Fisher result is only written on the
    figure, not saved. Returns the summary (pref_group, n_trials, n_correct, acc,
    ci_low, ci_high).
    """
    summary = summarize_correctness_by_pref_group(
        df=df,
        pref_col=pref_col,
        correct_col=correct_col,
        group_order=group_order,
    )

    # The plotting helpers expect the columns `group` and `accuracy`.
    plot_df = summary.rename(columns={"pref_group": "group", "acc": "accuracy"}).copy()

    fig, ax = barplot_accuracy(plot_df, order=list(group_order), figsize=(7, 4))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Correctness rate")
    ax.set_xlabel("Preference group")

    if title is None:
        title = f"Correctness by matching group ({metric_name})"
    ax.set_title(title)

    # Error bars and n labels.
    if show_n:
        add_wilson_errorbars_and_ns(
            ax,
            plot_df,
            y="accuracy",
            n_col="n_trials",
        )
    else:
        # Error bars only, without the n labels.
        for i, r in plot_df.reset_index(drop=True).iterrows():
            acc = float(r["accuracy"])
            if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) and np.isfinite(acc):
                ax.errorbar(
                    i,
                    acc,
                    yerr=[[acc - float(r["ci_low"])], [float(r["ci_high"]) - acc]],
                    fmt="none",
                    capsize=4,
                    ecolor="black",
                    elinewidth=1.5,
                )

    # ---- Fisher test, written at the top of the plot ----
    if show_test:
        test_res = correctness_by_pref_group_test(df=df, pref_col=pref_col, correct_col=correct_col)
        p = test_res.get("p_value", np.nan)
        or_ = test_res.get("odds_ratio", np.nan)
        stars = p_to_stars(p)

        txt = f"Fisher p={p:.3g} {stars}  (OR={or_:.2f})"
        ax.text(
            0.5,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.tight_layout()

    save_output(
        fig,
        analysis="correctness_associations",
        plot="correctness_by_matching",
        tables={"summary": summary},
        save=save,
        to_paper=to_paper,
        dpi=200,
        subdir="correctness_by_matching",
        group=h_or_g,
        mode=mode,
        metric=metric_name,
    )

    return summary


# For the polarity mode: whether the preferred answer is the one with the highest
# value on a measurement or the lowest. Keyed by the plots' columns (pupil size in
# mm); any other column, such as `num_label_visits` or a z-scored pupil column,
# counts as "high".
_DEFAULT_DIRECTION_BY_METRIC = {
    C.MEAN_DWELL_TIME: "high",
    C.MEAN_FIXATIONS_COUNT: "high",
    C.MEAN_FIRST_FIXATION_DURATION: "high",
    C.SKIP_RATE: "low",
    C.AREA_DWELL_PROPORTION: "high",
    C.MEAN_AVG_FIX_PUPIL_SIZE: "high",
    C.MEAN_MAX_FIX_PUPIL_SIZE: "high",
    C.MEAN_MIN_FIX_PUPIL_SIZE: "low",
    C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE: "high",
}


def run_all_matching_correctness_plots(
    all_participants: pd.DataFrame,
    split_groups: bool = True,
    metrics: List[str] = None,
    save: Optional[bool] = None,
    to_paper=None,
    print_summaries: bool = False,
) -> Dict:
    """Draw `plot_correctness_by_matching` for every measurement, both ways of
    picking the preferred answer, and each group.

    The two ways (`features.preference.compute_trial_matching`):
      polarity  the answer with the highest value, or the lowest for skip rate and
                smallest pupil size (`_DEFAULT_DIRECTION_BY_METRIC`)
      relative  the answer furthest from the trial's average, in either direction

    `metrics` defaults to the plots' list of per-area measurements (pupil size in
    mm). Groups are hunters, gatherers and all participants, or only all
    participants if `split_groups=False`.

    Returns results[mode][group][metric] = {"trial_df", "summary"}.
    """
    if metrics is None:
        metrics = list(C.AREA_METRIC_COLUMNS_VIZES)

    groups = split_participant_groups(all_participants, split=split_groups)

    modes = ["polarity", "relative"]
    results: Dict = {}

    for mode in modes:
        mode_results: Dict = {}

        for group_key, df in groups.items():
            group_label = "all participants" if group_key == "all_participants" else group_key
            group_results: Dict = {}

            for metric in metrics:
                direction = _DEFAULT_DIRECTION_BY_METRIC.get(metric, "high")

                if mode == "polarity":
                    trial_df = PM.compute_trial_matching(
                        df,
                        metric_col=metric,
                        direction=direction,
                        extreme_mode="polarity",
                    )
                    title = f"Correctness by matching ({metric}) — {group_label} — polarity ({direction})"
                else:
                    trial_df = PM.compute_trial_matching(
                        df,
                        metric_col=metric,
                        extreme_mode="relative",
                    )
                    title = f"Correctness by matching ({metric}) — {group_label} — relative"

                summary = plot_correctness_by_matching(
                    df=trial_df,
                    metric_name=metric,
                    title=title,
                    show_test=True,
                    save=save,
                    to_paper=to_paper,
                    h_or_g=group_key,
                    mode=mode,
                )

                if print_summaries:
                    print(f"\n=== {mode.upper()} | {group_label.upper()} | {metric} ===")
                    if mode == "polarity":
                        print(f"direction: {direction}")
                    print(summary)

                group_results[metric] = {
                    "trial_df": trial_df,
                    "summary": summary,
                }

            mode_results[group_key] = group_results

        results[mode] = mode_results

    return results
