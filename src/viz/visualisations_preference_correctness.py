"""
visualisations_preference_correctness.py

Visualize correctness rate differences between "matching" and "not_matching"
preference groups for a given metric.

Input: trial-level dataframe (one row per TRIAL_ID x PARTICIPANT_ID) that includes:
  - C.TRIAL_ID
  - C.PARTICIPANT_ID
  - "pref_group" (matching / not_matching)
  - C.IS_CORRECT_COLUMN (0/1 or bool)

Typically produced by:
  - statistics.preference_matching.compute_trial_matching(...)

Plots:
  - Bar chart of mean correctness with 95% CI (Wilson interval).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import scipy.stats as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants as C

from src.statistics.preference_correctness_tests import correctness_by_pref_group_test
from src.derived import preference_matching as PM


def _p_to_stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


@dataclass(frozen=True)
class CorrectnessSummary:
    pref_group: str
    n_trials: int
    n_correct: int
    acc: float
    ci_low: float
    ci_high: float


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson (score) confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes (e.g. correct trials)
    n : int
        Number of trials
    alpha : float
        1 - confidence level (default: 0.05 → 95% CI)

    Returns
    -------
    (ci_low, ci_high) : tuple of floats
    """
    if n == 0:
        return (float("nan"), float("nan"))

    # observed proportion
    p_hat = k / n

    # critical value for normal distribution
    z = st.norm.ppf(1 - alpha / 2)

    # estimated variance of p̂
    var_hat = p_hat * (1 - p_hat) / n

    # Wilson-specific shrinkage factor
    omega = n / (n + z**2)

    # adjusted center and spread
    A = p_hat + (z**2) / (2 * n)
    B = z * np.sqrt(var_hat + (z**2) / (4 * n**2))

    # confidence interval
    ci_low = omega * (A - B)
    ci_high = omega * (A + B)

    return float(ci_low), float(ci_high)



def summarize_correctness_by_pref_group(
    df: pd.DataFrame,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
    group_order: Sequence[str] = ("matching", "not_matching"),
) -> pd.DataFrame:
    """
    Return a tidy summary table:
      pref_group, n_trials, n_correct, acc, ci_low, ci_high
    """

    sub = df[[pref_col, correct_col]].copy()
    sub = sub.dropna(subset=[pref_col, correct_col])

    sub[correct_col] = sub[correct_col].astype(int)

    rows: list[CorrectnessSummary] = []
    for g in group_order:
        gdf = sub[sub[pref_col] == g]
        n = int(len(gdf))
        k = int(gdf[correct_col].sum())
        acc = float(k / n) if n else np.nan
        lo, hi = wilson_ci(k, n)
        rows.append(CorrectnessSummary(g, n, k, acc, lo, hi))

    return pd.DataFrame([r.__dict__ for r in rows])


def plot_correctness_by_matching(
    df: pd.DataFrame,
    metric_name: str,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
    group_order: Sequence[str] = ("matching", "not_matching"),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_n: bool = True,
    show_test: bool = True,
) -> pd.DataFrame:
    """
    Plot correctness rate (mean) ± 95% CI for matching vs not_matching.
    Optionally annotates Fisher exact test (p-value + odds ratio).
    """
    summary = summarize_correctness_by_pref_group(
        df=df,
        pref_col=pref_col,
        correct_col=correct_col,
        group_order=group_order,
    )

    labels = summary["pref_group"].tolist()
    acc = summary["acc"].to_numpy(dtype=float)
    lo = summary["ci_low"].to_numpy(dtype=float)
    hi = summary["ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([acc - lo, hi - acc])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, acc, yerr=yerr, capsize=6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Correctness rate")
    ax.set_xlabel("Preference group")

    if title is None:
        title = f"Correctness by matching group ({metric_name})"
    ax.set_title(title)

    # n labels
    if show_n:
        for i, row in summary.iterrows():
            ax.text(
                i,
                min(1.0, row["acc"] + 0.03),
                f"n={int(row['n_trials'])}",
                ha="center",
                va="bottom",
            )

    # ---- Statistical test annotation (Fisher exact) ----
    if show_test:
        test_res = correctness_by_pref_group_test(
            df=df, pref_col=pref_col, correct_col=correct_col
        )
        p = test_res["p_value"]
        or_ = test_res["odds_ratio"]
        stars = _p_to_stars(p)

        txt = f"Fisher p={p:.3g} {stars}  (OR={or_:.2f})"
        ax.text(
            0.5, 0.98, txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

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
    C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE: "high"
}


def run_all_matching_correctness_plots(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metrics: List[str] = None,
    output_root: str = "../reports/plots/matching_correctness",
    save_plots: bool = True,
    print_summaries: bool = False,
) -> Dict:
    """
    For each metric in AREA_METRIC_COLUMNS, compute trial-level matching labels
    and plot correctness by matching group, for both:
      - extreme_mode="polarity" (direction chosen per metric)
      - extreme_mode="relative" (direction ignored)

    Runs for:
      - hunters
      - gatherers
      - all_participants

    Folder structure:
      {output_root}/{mode}/{group}/correctness_by_matching__{metric}.png

    Returns
    -------
    results[mode][group][metric] = {
        "trial_df": DataFrame,
        "summary": DataFrame,
        "save_path": str|None,
    }
    """
    if metrics is None:
        metrics = list(C.AREA_METRIC_COLUMNS)

    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    groups = {
        "hunters": hunters,
        "gatherers": gatherers,
        "all_participants": all_participants,
    }

    modes = ["polarity", "relative"]
    results: Dict = {}

    for mode in modes:
        mode_results: Dict = {}

        for group_key, df in groups.items():
            group_label = "all participants" if group_key == "all_participants" else group_key

            group_results: Dict = {}

            for metric in metrics:
                # choose direction for polarity mode
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

                save_path = None
                if save_plots:
                    save_dir = Path(output_root) / mode / group_key
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = str(save_dir / f"correctness_by_matching__{metric}.png")

                summary = plot_correctness_by_matching(
                    df=trial_df,
                    metric_name=metric,
                    title=title,
                    save_path=save_path,
                    show_test=True,
                )

                if print_summaries:
                    print(f"\n=== {mode.upper()} | {group_label.upper()} | {metric} ===")
                    if mode == "polarity":
                        print(f"direction: {direction}")
                    print(summary)

                group_results[metric] = {
                    "trial_df": trial_df,
                    "summary": summary,
                    "save_path": save_path,
                }

            mode_results[group_key] = group_results

        results[mode] = mode_results

    return results