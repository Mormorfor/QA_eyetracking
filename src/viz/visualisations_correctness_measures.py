# src/viz/visualisations_correctness_measures.py

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

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
