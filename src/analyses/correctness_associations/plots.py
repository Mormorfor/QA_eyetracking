"""Figures: correctness against scan effort, dwell, response time and preference matching.

Seven `corr_by_*` families, which is how `reports/correctness_associations/` already
groups them -- `corr_by_matching` is a peer of the other six, not a separate analysis,
which is why the preference-matching plots merged in here rather than standing alone."""

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


# ==========================================================================
# from src/viz/visualisations_correctness_measures.py
# ==========================================================================

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
    """
    Plot correctness split by sequence length <=threshold vs >threshold,
    with Wilson 95% CI and optional Fisher exact significance annotation.

    Returns (fig, summary_df, test_result_dict_or_None).
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
    """
    For each threshold, plot correctness split by sequence length bin
    (<=threshold vs >threshold) for:
      - hunters
      - gatherers
      - all_participants

    Returns
    -------
    results[group][threshold] = {"fig": fig, "summary": summary_df, "test": test_dict}
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
    """
    Continuous version:
    Plot correctness rate as a function of sequence length.

    - Aggregates trials by seq_len: accuracy + Wilson 95% CI
    - Optionally filters to seq_len <= max_len
    - Optionally drops lengths with n < min_n_per_len

    Returns (fig, summary_df) where summary_df has:
      seq_len, n, k_correct, accuracy, ci_low, ci_high
    """
    # We reuse the derived logic indirectly by calling the threshold builder repeatedly would be slow,
    # so we just compute seq_len per trial here (same logic as derived.sequence_len_literal_eval).
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

    # Drop sparse lengths (optional)
    if min_n_per_len is not None and min_n_per_len > 1:
        agg = agg[agg["n"] >= int(min_n_per_len)].copy()

    # Accuracy + Wilson CI
    agg["accuracy"] = agg["k_correct"] / agg["n"]
    cis = agg.apply(lambda r: wilson_ci(int(r["k_correct"]), int(r["n"])), axis=1)
    agg["ci_low"] = [c[0] for c in cis]
    agg["ci_high"] = [c[1] for c in cis]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    # line
    ax.plot(agg["seq_len"], agg["accuracy"], marker="o")

    # CI as ribbon (optional)
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
    """
    Continuous seq-length plots for:
      - hunters
      - gatherers
      - all_participants

    Returns results[group] = {"fig": fig, "summary": df}
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
    """
    Compare correctness for trials with vs without a back-and-forth pattern
    at the end of the fixation sequence.

    Patterns:
      - XYX (default): last 3 entries form A B A
      - XYXY (use_xyxy=True): last 4 entries form A B A B

    Returns (fig, summary_df, test_res_or_None).
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
    """
    Run pattern-based correctness plots for:
      - hunters
      - gatherers
      - all_participants

    Returns results[group] = {"fig": fig, "summary": df, "test": dict}
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
# Mean Dwell per word
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
    """
    Compare correctness for trials with low vs high
    mean dwell time per word across the entire trial.

    Returns (fig, summary_df, test_res_or_None).
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
    """
    For each threshold, compare correctness by whole-trial
    mean dwell time per word for:
      - hunters
      - gatherers
      - all_participants

    Returns
    -------
    results[group][threshold] = {"fig": fig, "summary": summary_df, "test": test_dict}
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
    """
    Continuous version:
    Plot correctness rate as a function of whole-trial mean dwell per word.

    Since dwell is continuous, we bin it:
      - either fixed-width bins (bin_width)
      - or a fixed number of bins (n_bins)

    Returns (fig, summary_df) where summary_df has:
      bin_left, bin_right, bin_center, n, k_correct, accuracy, ci_low, ci_high
    """

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)

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

    # assign each trial into a bin index
    # right=False means [left, right)
    bin_idx = (
        np.digitize(trial_df["trial_mean_dwell"].to_numpy(), edges, right=False) - 1
    )
    # keep only valid bins
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
    """
    Continuous mean-dwell plots for:
      - hunters
      - gatherers
      - all_participants

    Returns results[group] = {"fig": fig, "summary": df}
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
# Total answering reading time
# ------------------------------------------


def plot_correctness_by_total_answering_rt_continuous(
    df: pd.DataFrame,
    # CONFIRM_FINAL_ANSWER_RT is the canonical column here (Diana, 2026-09-20).
    # It used to default to "total_answering_RT", which disagreed with the
    # run_all_* wrapper below AND does not exist in all_participants.csv at all --
    # that friendlier name is minted later, in features/build.py.
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
    """
    Plot correctness rate as a function of total answering reading time.

    Since total_answering_RT is continuous, the function bins trials by RT and
    computes correctness rate within each bin.

    Returns
    -------
    fig, summary_df
        summary_df contains:
        bin_left, bin_right, bin_center, n, k_correct, accuracy, ci_low, ci_high
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

    # If the dataframe is word-/IA-level, the same trial can appear in multiple rows.
    # We collapse to one row per participant-trial before calculating accuracy.
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
        # Which RT column is in the filename on purpose: raw confirm-press RT and
        # the length-normalized variant are different measures on different scales
        # (ms vs a 0-1 ratio), and without this tag a rerun under the other one
        # would silently overwrite the first under an identical name.
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
    """
    Continuous total-answering-RT plots for:
      - hunters
      - gatherers
      - all_participants

    Returns results[group] = {"fig": fig, "summary": df}
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


# ==========================================================================
# from src/viz/visualisations_preference_correctness.py
# ==========================================================================

def summarize_correctness_by_pref_group(
    df: pd.DataFrame,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
    group_order: Sequence[str] = ("matching", "not_matching"),
) -> pd.DataFrame:
    """
    Return a tidy summary table:
      pref_group, n_trials, n_correct, acc, ci_low, ci_high

    Notes
    -----
    - Expects trial-level df (one row per trial x participant).
    - Uses Wilson CI via shared derived utilities.
    """
    sub = df[[pref_col, correct_col]].copy()
    sub = sub.dropna(subset=[pref_col, correct_col])
    sub[correct_col] = sub[correct_col].astype(int)

    # enforce stable plotting order
    sub[pref_col] = pd.Categorical(sub[pref_col], categories=list(group_order), ordered=True)

    tmp = summarize_binary_by_group(
        trial_df=sub.rename(columns={pref_col: "group", correct_col: "is_correct"}),
        group_col="group",
        outcome_col="is_correct",
    ).sort_values("group")

    # match legacy column names used elsewhere in your pipeline
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
    """
    Plot correctness rate (mean) ± 95% CI for matching vs not_matching.
    Optionally annotates Fisher exact test (p-value + odds ratio).

    Returns
    -------
    summary : pd.DataFrame
        Columns: pref_group, n_trials, n_correct, acc, ci_low, ci_high
    """
    summary = summarize_correctness_by_pref_group(
        df=df,
        pref_col=pref_col,
        correct_col=correct_col,
        group_order=group_order,
    )

    # Build the standardized summary table expected by viz_helpers
    plot_df = summary.rename(columns={"pref_group": "group", "acc": "accuracy"}).copy()

    fig, ax = barplot_accuracy(plot_df, order=list(group_order), figsize=(7, 4))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Correctness rate")
    ax.set_xlabel("Preference group")

    if title is None:
        title = f"Correctness by matching group ({metric_name})"
    ax.set_title(title)

    # error bars + n labels (centralized)
    if show_n:
        add_wilson_errorbars_and_ns(
            ax,
            plot_df,
            y="accuracy",
            n_col="n_trials",
        )
    else:
        # still draw error bars (CI), but skip n labels
        # (small local tweak to avoid proliferating variants in viz_helpers yet)
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

    # ---- Statistical test annotation (Fisher exact) ----
    if show_test:
        test_res = correctness_by_pref_group_test(df=df, pref_col=pref_col, correct_col=correct_col)
        p = test_res.get("p_value", np.nan)
        or_ = test_res.get("odds_ratio", np.nan)
        stars = p_to_stars(p)

        # keep style consistent with your other plots: small text in the top area
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
    """
    For each metric in AREA_METRIC_COLUMNS_VIZES, compute trial-level matching labels
    and plot correctness by matching group, for both:
      - extreme_mode="polarity" (direction chosen per metric)
      - extreme_mode="relative" (direction ignored)

    Runs for:
      - hunters
      - gatherers
      - all_participants

    Folder structure:
      reports/correctness_associations/figures/correctness_by_matching/

    Returns
    -------
    results[mode][group][metric] = {
        "trial_df": DataFrame,
        "summary": DataFrame,
    }
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
