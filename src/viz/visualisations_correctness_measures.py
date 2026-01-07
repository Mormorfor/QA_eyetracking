# src/viz/visualisations_correctness_measures.py

from __future__ import annotations

import os
import ast
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src import constants as Con
from src.statistics.correctness_measures_tests import (
    correctness_by_seq_len_threshold_test,
    correctness_by_sequence_pattern_test,
    correctness_by_trial_mean_dwell_threshold_test,
)


def _p_to_stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ------------------------------------------
# Sequence Length Measures
# ------------------------------------------

def _sequence_len_literal_eval(x) -> int:
    """
    Sequences come as string representations of lists/tuples.
    We first literal-eval them, then measure length.

    Invalid / NaN values → length 0.
    """
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0

    # literal-eval if string
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return 0

    if isinstance(x, (list, tuple)):
        return len(x)

    return 0



def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (low, high). If n == 0 -> (nan, nan).
    """
    if n <= 0:
        return (np.nan, np.nan)

    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))




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

    d = df[
        [
            Con.TRIAL_ID,
            Con.PARTICIPANT_ID,
            seq_col,
            correct_col,
        ]
    ].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq_len"] = d[seq_col].apply(_sequence_len_literal_eval)
    trial_df = (
        d
        .groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(
            seq_len=("_seq_len", "first"),
            is_correct=(correct_col, "first"),
        )
    )

    trial_df["_bin"] = np.where(
        trial_df["seq_len"] > threshold,
        f"> {threshold}",
        f"≤ {threshold}",
    )

    rows = []
    order = [f"≤ {threshold}", f"> {threshold}"]
    for bin_label in order:
        dd = trial_df[trial_df["_bin"] == bin_label]
        n = len(dd)
        k = dd["is_correct"].sum()
        acc = (k / n) if n else np.nan
        lo, hi = _wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append(
            dict(
                group=bin_label,
                n=n,
                k_correct=k,
                accuracy=acc,
                ci_low=lo,
                ci_high=hi,
            )
        )

    summary_df = pd.DataFrame(rows)

    test_res = None
    if add_significance:
        test_res = correctness_by_seq_len_threshold_test(
            df=df,
            threshold=threshold,
            seq_col=seq_col,
            correct_col=correct_col,
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=summary_df,
        x="group",
        y="accuracy",
        order=order,
        ax=ax,
    )

    for i, r in summary_df.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) and np.isfinite(r["accuracy"]):
            ax.errorbar(
                i,
                r["accuracy"],
                yerr=[[r["accuracy"] - r["ci_low"]], [r["ci_high"] - r["accuracy"]]],
                fmt="none",
                capsize=4,
                ecolor="black",
                elinewidth=1.5,
            )
        if np.isfinite(r["accuracy"]):
            ax.text(
                i,
                min(0.98, r["accuracy"] + 0.03),
                f"n={int(r['n'])}",
                ha="center",
                va="bottom",
            )

    default_title = f"{h_or_g}: Correctness by sequence length (threshold={threshold})"
    ax.set_title(title or default_title)
    ax.set_xlabel("Sequence length bin")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if add_significance and test_res is not None:
        p = test_res["p_value"]
        stars = _p_to_stars(p)

        y_max = float(np.nanmax(summary_df["accuracy"].values))
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin

        y = y_max + 0.06 * yr
        h = 0.03 * yr

        # expand ylim if needed so stars always fit
        if y + h + 0.05 * yr > ymax:
            ax.set_ylim(ymin, y + h + 0.08 * yr)

        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, c="black", clip_on=False)
        ax.text(
            0.5,
            y + h + 0.01 * yr,
            stars,
            ha="center",
            va="bottom",
            color="black",
            clip_on=False,
        )

        x1, x2 = 0, 1
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        ax.text((x1 + x2) / 2, y + h + 0.01, stars, ha="center", va="bottom", color="black")

    if save:
        out_dir = os.path.join(output_root, "correctness_by_seq_len_threshold")
        os.makedirs(out_dir, exist_ok=True)

        fig_path = os.path.join(out_dir, f"{h_or_g}__thresh_{threshold}.png")
        fig.savefig(fig_path, dpi=300)

        data_dir = os.path.join(
            report_root,
            "correctness_by_seq_len_threshold",
        )
        os.makedirs(data_dir, exist_ok=True)

        summary_df.to_csv(
            os.path.join(data_dir, f"{h_or_g}__thresh_{threshold}__summary.csv"),
            index=False,
        )

        if test_res is not None:
            import json

            test_res_json = test_res.copy()
            if "contingency_table" in test_res_json:
                test_res_json["contingency_table"] = test_res_json["contingency_table"].tolist()

            with open(
                    os.path.join(data_dir, f"{h_or_g}__thresh_{threshold}__fisher.json"),
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(test_res_json, f, indent=2)


    return fig, summary_df, test_res



def run_all_correctness_seq_len_threshold_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    thresholds: Tuple[int, ...] = (2, 3, 4, 5),
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
) -> Dict[str, Dict[int, Dict[str, object]]]:
    """
    For each threshold, plot correctness split by sequence length bin
    (<=threshold vs >threshold) for:
      - hunters
      - gatherers
      - all_participants

    Returns
    -------
    results[group][threshold] = {"fig": fig, "summary": summary_df}
    """
    def _run_for_group(df: pd.DataFrame, group_name: str) -> Dict[int, Dict[str, object]]:
        group_results = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_sequence_len_threshold(
                df=df,
                threshold=int(t),
                h_or_g=group_name,
                save=save_plots,
                output_root=output_root,
                report_root=report_root,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)

            group_results[int(t)] = {"fig": fig, "summary": summary, "test": test_res}


        return group_results

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    results = {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }

    return results

# ------------------------------------------
# ABA / ABAB detection
# ------------------------------------------

def _parse_seq(x):
    """Parse sequence column via ast.literal_eval (expects list/tuple of strings)."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    return x if isinstance(x, (list, tuple)) else None


def _has_back_and_forth_aba(seq) -> bool:
    """
    Detect ABA in the last 3 entries:
      last == two_before AND last != one_before
    Requires length >= 3.
    """
    if seq is None or len(seq) < 3:
        return False
    a = seq[-1]
    b = seq[-2]
    c = seq[-3]
    return (a == c) and (a != b)


def _has_back_and_forth_abab(seq) -> bool:
    """
    Detect ABAB in the last 4 entries:
      x[-4], x[-3], x[-2], x[-1] form A B A B
    Requires:
      - length >= 4
      - exactly two distinct values
      - positions 0 and 2 match, positions 1 and 3 match, and they differ
    """
    if seq is None or len(seq) < 4:
        return False
    a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
    return (a == c) and (b == d) and (a != b)



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
    use_abab: bool = False,
    add_significance: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, Optional[Dict]]:
    """
    Compare correctness for trials with vs without a back-and-forth pattern
    at the end of the fixation sequence.

    Patterns:
      - ABA (default): last 3 entries form A B A
      - ABAB (use_abab=True): last 4 entries form A B A B

    Returns (fig, summary_df, test_res_or_None).
    """

    pattern_name = "ABAB" if use_abab else "ABA"
    pattern_fn = _has_back_and_forth_abab if use_abab else _has_back_and_forth_aba

    d = df[
        [
            Con.TRIAL_ID,
            Con.PARTICIPANT_ID,
            seq_col,
            correct_col,
        ]
    ].copy()
    d[correct_col] = d[correct_col].astype(int)

    d["_seq"] = d[seq_col].apply(_parse_seq)
    d["_has_pattern"] = d["_seq"].apply(
        lambda s: bool(pattern_fn(s)) if s is not None else False
    )

    trial_df = (
        d
        .groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(
            has_pattern=("_has_pattern", "first"),
            is_correct=(correct_col, "first"),
        )
    )

    trial_df["_bin"] = np.where(
        trial_df["has_pattern"],
        f"{pattern_name} present",
        f"{pattern_name} absent",
    )

    order = [f"{pattern_name} absent", f"{pattern_name} present"]

    rows = []
    for bin_label in order:
        dd = trial_df[trial_df["_bin"] == bin_label]
        n = len(dd)
        k = dd["is_correct"].sum()
        acc = (k / n) if n else np.nan
        lo, hi = _wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append(
            dict(
                group=bin_label,
                n=n,
                k_correct=k,
                accuracy=acc,
                ci_low=lo,
                ci_high=hi,
            )
        )

    summary_df = pd.DataFrame(rows)

    test_res = None
    if add_significance:
        test_res = correctness_by_sequence_pattern_test(
            df=df,
            pattern_fn=pattern_fn,
            seq_col=seq_col,
            correct_col=correct_col,
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=summary_df,
        x="group",
        y="accuracy",
        order=order,
        ax=ax,
    )

    # Black CI bars
    for i, r in summary_df.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) and np.isfinite(r["accuracy"]):
            ax.errorbar(
                i,
                r["accuracy"],
                yerr=[[r["accuracy"] - r["ci_low"]], [r["ci_high"] - r["accuracy"]]],
                fmt="none",
                capsize=4,
                ecolor="black",
                elinewidth=1.5,
            )
        if np.isfinite(r["accuracy"]):
            ax.text(
                i,
                min(0.98, r["accuracy"] + 0.03),
                f"n={int(r['n'])}",
                ha="center",
                va="bottom",
            )

    default_title = f"{h_or_g}: Correctness by end-pattern ({pattern_name})"
    ax.set_title(title or default_title)
    ax.set_xlabel("Pattern bin")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    # Significance bracket + stars
    if add_significance and test_res is not None:
        p = test_res["p_value"]
        stars = _p_to_stars(p)

        y_max = float(np.nanmax(summary_df["accuracy"].values))
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin

        y = y_max + 0.06 * yr
        h = 0.03 * yr

        # expand ylim if needed so stars always fit
        if y + h + 0.05 * yr > ymax:
            ax.set_ylim(ymin, y + h + 0.08 * yr)

        ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, c="black", clip_on=False)
        ax.text(
            0.5,
            y + h + 0.01 * yr,
            stars,
            ha="center",
            va="bottom",
            color="black",
            clip_on=False,
        )

        x1, x2 = 0, 1
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
        ax.text((x1 + x2) / 2, y + h + 0.01, stars, ha="center", va="bottom", color="black")

    if save:
        out_dir = os.path.join(output_root, "correctness_by_back_and_forth_pattern")
        os.makedirs(out_dir, exist_ok=True)

        suffix = "abab" if use_abab else "aba"
        fig_path = os.path.join(out_dir, f"{h_or_g}__{suffix}.png")
        fig.savefig(fig_path, dpi=300)

        data_dir = os.path.join(
            report_root,
            "correctness_by_back_and_forth_pattern",
        )
        os.makedirs(data_dir, exist_ok=True)

        summary_df.to_csv(
            os.path.join(data_dir, f"{h_or_g}__{suffix}__summary.csv"),
            index=False,
        )

        if test_res is not None:
            import json

            test_res_json = test_res.copy()
            if "contingency_table" in test_res_json:
                test_res_json["contingency_table"] = test_res_json["contingency_table"].tolist()

            with open(
                    os.path.join(data_dir, f"{h_or_g}__{suffix}__fisher.json"),
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(test_res_json, f, indent=2)

    return fig, summary_df, test_res



def run_all_back_and_forth_pattern_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
    use_abab: bool = False,
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
            h_or_g=group_name,
            save=save_plots,
            output_root=output_root,
            report_root=report_root,
            use_abab=use_abab,
            add_significance=True,
        )
        if print_summaries:
            print(f"\n=== {group_name.upper()} — pattern: {'ABAB' if use_abab else 'ABA'} ===")
            print(summary)
            if test_res is not None:
                print(f"Fisher p={test_res['p_value']:.4g}, OR={test_res['odds_ratio']:.4g}")
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


def _compute_trial_mean_dwell_per_word(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
) -> pd.Series:
    """
    Compute mean dwell time per word across the entire trial
    for each (TRIAL_ID, PARTICIPANT_ID).

    Returns a Series aligned with df rows.
    """
    total_dwell = (
        df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col]
        .transform("sum")
    )

    n_words = (
        df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col]
        .transform("count")
    )

    return total_dwell / n_words


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
    """

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)

    d["_trial_mean_dwell"] = _compute_trial_mean_dwell_per_word(d, dwell_col)

    trial_df = (
        d
        .groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(
            trial_mean_dwell=("_trial_mean_dwell", "first"),
            is_correct=(correct_col, "first"),
        )
    )

    trial_df["_bin"] = np.where(
        trial_df["trial_mean_dwell"] <= threshold,
        f"≤ {threshold}",
        f"> {threshold}",
    )

    order = [f"≤ {threshold}", f"> {threshold}"]

    rows = []
    for bin_label in order:
        dd = trial_df[trial_df["_bin"] == bin_label]
        n = len(dd)
        k = dd["is_correct"].sum()
        acc = (k / n) if n else np.nan
        lo, hi = _wilson_ci(k, n) if n else (np.nan, np.nan)

        rows.append(
            dict(
                group=bin_label,
                n=n,
                k_correct=k,
                accuracy=acc,
                ci_low=lo,
                ci_high=hi,
            )
        )

    summary_df = pd.DataFrame(rows)

    test_res = None
    if add_significance:
        test_res = correctness_by_trial_mean_dwell_threshold_test(
            df=df,
            threshold=threshold,
            dwell_col=dwell_col,
            correct_col=correct_col,
        )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=summary_df,
        x="group",
        y="accuracy",
        order=order,
        ax=ax,
    )

    for i, r in summary_df.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            ax.errorbar(
                i,
                r["accuracy"],
                yerr=[[r["accuracy"] - r["ci_low"]], [r["ci_high"] - r["accuracy"]]],
                fmt="none",
                capsize=4,
                ecolor="black",
                elinewidth=1.5,
            )
        ax.text(
            i,
            min(0.98, r["accuracy"] + 0.03),
            f"n={int(r['n'])}",
            ha="center",
            va="bottom",
        )

    default_title = (
        f"{h_or_g}: Correctness by whole-trial mean dwell per word "
        f"(threshold={threshold})"
    )
    ax.set_title(title or default_title)
    ax.set_xlabel("Trial mean dwell per word bin")
    ax.set_ylabel("Correctness rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    # ---- significance stars ----
    if add_significance and test_res is not None:
        p = test_res["p_value"]
        stars = _p_to_stars(p)

        y_max = float(np.nanmax(summary_df["accuracy"].values))
        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin

        y = y_max + 0.06 * yr
        h = 0.03 * yr

        # expand ylim if needed
        if y + h + 0.05 * yr > ymax:
            ax.set_ylim(ymin, y + h + 0.08 * yr)

        x1, x2 = 0, 1
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=1.5,
            c="black",
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + h + 0.01 * yr,
            stars,
            ha="center",
            va="bottom",
            color="black",
            clip_on=False,
        )


    if save:
        plot_dir = os.path.join(
            output_root,
            "correctness_by_trial_mean_dwell_threshold",
        )
        os.makedirs(plot_dir, exist_ok=True)

        fig_path = os.path.join(
            plot_dir,
            f"{h_or_g}__thresh_{threshold}.png",
        )
        fig.savefig(fig_path, dpi=300)

        data_dir = os.path.join(
            report_root,
            "correctness_by_trial_mean_dwell_threshold",
        )
        os.makedirs(data_dir, exist_ok=True)

        summary_df.to_csv(
            os.path.join(
                data_dir,
                f"{h_or_g}__thresh_{threshold}__summary.csv",
            ),
            index=False,
        )

        if test_res is not None:
            import json

            test_json = test_res.copy()
            if "contingency_table" in test_json:
                test_json["contingency_table"] = test_json["contingency_table"].tolist()

            with open(
                    os.path.join(
                        data_dir,
                        f"{h_or_g}__thresh_{threshold}__fisher.json",
                    ),
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(test_json, f, indent=2)

    return fig, summary_df, test_res


def run_all_trial_mean_dwell_threshold_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    thresholds: Tuple[float, ...] = (50.0, 75.0, 100.0),
    output_root: str = "../reports/plots/correctness_measures",
    report_root: str = "../reports/report_data",
    save_plots: bool = True,
    print_summaries: bool = False,
) -> Dict[str, Dict[float, Dict[str, object]]]:
    """
    For each threshold, compare correctness by whole-trial
    mean dwell time per word for:
      - hunters
      - gatherers
      - all_participants

    Returns
    -------
    results[group][threshold] = {
        "fig": fig,
        "summary": summary_df,
        "test": test_dict
    }
    """

    def _run_for_group(
        df: pd.DataFrame,
        group_name: str,
    ) -> Dict[float, Dict[str, object]]:
        group_results = {}

        for t in thresholds:
            fig, summary, test_res = plot_correctness_by_trial_mean_dwell_threshold(
                df=df,
                threshold=float(t),
                h_or_g=group_name,
                save=save_plots,
                output_root=output_root,
                report_root=report_root,
            )

            if print_summaries:
                print(f"\n=== {group_name.upper()} — threshold: {t} ===")
                print(summary)
                if test_res is not None:
                    print(
                        f"Fisher p={test_res['p_value']:.4g}, "
                        f"OR={test_res['odds_ratio']:.4g}"
                    )

            group_results[float(t)] = {
                "fig": fig,
                "summary": summary,
                "test": test_res,
            }

        return group_results

    all_participants = pd.concat(
        [hunters, gatherers],
        ignore_index=True,
    )

    return {
        "hunters": _run_for_group(hunters, "hunters"),
        "gatherers": _run_for_group(gatherers, "gatherers"),
        "all_participants": _run_for_group(all_participants, "all_participants"),
    }

