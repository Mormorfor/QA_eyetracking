# src/viz/visualisations_correctness_measures.py

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src import constants as Con
from src.derived.correctness_measures import (
    compute_seq_len_threshold_summary,
    compute_back_and_forth_pattern_summary,
    compute_trial_mean_dwell_threshold_summary,
)
from src.viz.viz_helpers import (
    add_significance_bracket,
    add_wilson_errorbars_and_ns,
    barplot_accuracy,
    p_to_stars,
    save_plot_and_report,
)


# ------------------------------------------
# Sequence Length Measures
# ------------------------------------------

def plot_correctness_by_sequence_len_threshold(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: bool = False,
    h_or_g: str = "hunters",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
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

    if save:
        plot_dir = os.path.join(output_root, "correctness_by_seq_len_threshold")
        data_dir = os.path.join(report_root, "correctness_by_seq_len_threshold")
        base_name = f"{h_or_g}__thresh_{threshold}"
        save_plot_and_report(
            fig=fig,
            summary_df=summary_df,
            test_res=test_res,
            plot_dir=plot_dir,
            data_dir=data_dir,
            base_name=base_name,
        )

    return fig, summary_df, test_res


def run_all_correctness_seq_len_threshold_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    thresholds: Tuple[int, ...] = (2, 3, 4, 5),
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
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

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[int, Dict[str, object]]:
        group_results: Dict[int, Dict[str, object]] = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_sequence_len_threshold(
                df=df,
                threshold=int(t),
                seq_col=seq_col,
                correct_col=correct_col,
                h_or_g=group_name,
                save=save_plots,
                output_root=output_root,
                report_root=report_root,
                add_significance=add_significance,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)
                if test_res is not None and test_res.get("p_value") is not None:
                    print(f"p={test_res.get('p_value')}")

            group_results[int(t)] = {"fig": fig, "summary": summary, "test": test_res}

        return group_results

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }



def plot_correctness_by_sequence_len_continuous(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (7, 4),
    save: bool = False,
    h_or_g: str = "hunters",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
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

    trial_df = (
        d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(seq_len=("_seq_len", "first"), is_correct=(correct_col, "first"))
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
    def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n <= 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + (z**2) / n
        center = (phat + (z**2) / (2 * n)) / denom
        half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
        return (max(0.0, center - half), min(1.0, center + half))

    agg["accuracy"] = agg["k_correct"] / agg["n"]
    cis = agg.apply(lambda r: _wilson_ci(int(r["k_correct"]), int(r["n"])), axis=1)
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
    ax.set_title(title or default_title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ---- Save ----
    if save:
        plot_dir = os.path.join(output_root, "correctness_by_seq_len_continuous")
        data_dir = os.path.join(report_root, "correctness_by_seq_len_continuous")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        suffix = f"maxlen_{max_len}" if max_len is not None else "all"
        base_name = f"{h_or_g}__{suffix}__minn_{min_n_per_len}"

        fig.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=300)
        agg.to_csv(os.path.join(data_dir, f"{base_name}__summary.csv"), index=False)

    return fig, agg


def run_all_correctness_seq_len_continuous_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    max_len: Optional[int] = None,
    min_n_per_len: int = 5,
    show_ci: bool = True,
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
            save=save_plots,
            output_root=output_root,
            report_root=report_root,
            max_len=max_len,
            min_n_per_len=min_n_per_len,
            show_ci=show_ci,
        )

        if print_summaries:
            print(f"\n=== {group_name.upper()} — continuous seq_len ===")
            print(summary.head(15))
            if len(summary) > 0:
                print(f"... ({len(summary)} lengths total)")

        return {"fig": fig, "summary": summary}

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }




# ------------------------------------------
# XYX / XYXY detection
# ------------------------------------------

def plot_correctness_by_back_and_forth_pattern(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: bool = False,
    h_or_g: str = "hunters",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
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

    if save:
        plot_dir = os.path.join(output_root, "correctness_by_back_and_forth_pattern")
        data_dir = os.path.join(report_root, "correctness_by_back_and_forth_pattern")
        suffix = "xyxy" if use_xyxy else "xyx"
        base_name = f"{h_or_g}__{suffix}"
        save_plot_and_report(
            fig=fig,
            summary_df=summary_df,
            test_res=test_res,
            plot_dir=plot_dir,
            data_dir=data_dir,
            base_name=base_name,
        )

    return fig, summary_df, test_res


def run_all_back_and_forth_pattern_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
    use_xyxy: bool = False,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
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
            save=save_plots,
            output_root=output_root,
            report_root=report_root,
            use_xyxy=use_xyxy,
            add_significance=add_significance,
        )

        if print_summaries:
            print(f"\n=== {group_name.upper()} — pattern: {'XYXY' if use_xyxy else 'XYX'} ===")
            print(summary)
            if test_res is not None and test_res.get("p_value") is not None:
                print(f"p={test_res.get('p_value')}")

        return {"fig": fig, "summary": summary, "test": test_res}

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }


# ------------------------------------------
# Mean Dwell per word
# ------------------------------------------

def plot_correctness_by_trial_mean_dwell_threshold(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (6, 4),
    save: bool = False,
    h_or_g: str = "hunters",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
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

    if save:
        plot_dir = os.path.join(output_root, "correctness_by_trial_mean_dwell_threshold")
        data_dir = os.path.join(report_root, "correctness_by_trial_mean_dwell_threshold")
        base_name = f"{h_or_g}__thresh_{threshold}"
        save_plot_and_report(
            fig=fig,
            summary_df=summary_df,
            test_res=test_res,
            plot_dir=plot_dir,
            data_dir=data_dir,
            base_name=base_name,
        )

    return fig, summary_df, test_res


def run_all_trial_mean_dwell_threshold_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    thresholds: Tuple[float, ...] = (50.0, 75.0, 100.0),
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
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

    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[float, Dict[str, object]]:
        group_results: Dict[float, Dict[str, object]] = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_trial_mean_dwell_threshold(
                df=df,
                threshold=float(t),
                dwell_col=dwell_col,
                correct_col=correct_col,
                h_or_g=group_name,
                save=save_plots,
                output_root=output_root,
                report_root=report_root,
                add_significance=add_significance,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)
                if test_res is not None and test_res.get("p_value") is not None:
                    print(f"p={test_res.get('p_value')}")

            group_results[float(t)] = {"fig": fig, "summary": summary, "test": test_res}

        return group_results

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }



def plot_correctness_by_trial_mean_dwell_continuous(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    figsize: Tuple[int, int] = (7, 4),
    save: bool = False,
    h_or_g: str = "hunters",
    title: Optional[str] = None,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
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
    def _wilson_ci(k: int, n: int, z: float = 1.96):
        if n <= 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + (z**2) / n
        center = (phat + (z**2) / (2 * n)) / denom
        half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
        return (max(0.0, center - half), min(1.0, center + half))

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)

    total_dwell = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform("sum")
    n_words = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform("count")
    d["_trial_mean_dwell"] = total_dwell / n_words

    trial_df = (
        d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(trial_mean_dwell=("_trial_mean_dwell", "first"), is_correct=(correct_col, "first"))
    )

    if x_max is not None:
        trial_df = trial_df[trial_df["trial_mean_dwell"] <= float(x_max)].copy()

    x = trial_df["trial_mean_dwell"].to_numpy()
    if len(x) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title or f"{h_or_g}: Correctness by mean dwell per word (continuous)")
        ax.set_xlabel("Trial mean dwell per word")
        ax.set_ylabel("Correctness rate")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig, pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_center", "n", "k_correct", "accuracy", "ci_low", "ci_high"]
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
    bin_idx = np.digitize(trial_df["trial_mean_dwell"].to_numpy(), edges, right=False) - 1
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

    cis = agg.apply(lambda r: _wilson_ci(int(r["k_correct"]), int(r["n"])), axis=1)
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

    if save:
        plot_dir = os.path.join(output_root, "correctness_by_trial_mean_dwell_continuous")
        data_dir = os.path.join(report_root, "correctness_by_trial_mean_dwell_continuous")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        if n_bins is not None:
            suffix = f"nbins_{int(n_bins)}"
        else:
            suffix = f"bw_{bin_width}"
        if x_max is not None:
            suffix += f"__xmax_{x_max}"

        base_name = f"{h_or_g}__{suffix}__minn_{min_n_per_bin}"

        fig.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=300)
        agg.to_csv(os.path.join(data_dir, f"{base_name}__summary.csv"), index=False)

    return fig, agg



def run_all_trial_mean_dwell_continuous_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
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
            save=save_plots,
            output_root=output_root,
            report_root=report_root,
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

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }

