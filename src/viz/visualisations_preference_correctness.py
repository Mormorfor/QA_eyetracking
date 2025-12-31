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
from typing import Optional, Sequence

import scipy.stats as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants as C


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
) -> pd.DataFrame:
    """
    Plot correctness rate (mean) ± 95% CI for matching vs not_matching.

    Returns the summary dataframe used for the plot (handy for logging/tests).
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

    if show_n:
        for i, row in summary.iterrows():
            ax.text(i, min(1.0, row["acc"] + 0.03), f"n={int(row['n_trials'])}", ha="center", va="bottom")

    fig.tight_layout()

    if save_path:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    return summary


def run_matching_correctness_plots(
    trial_df: pd.DataFrame,
    metric_name: str,
    output_root: str = "reports/plots/matching_correctness",
    save: bool = True,
) -> dict:
    """
    Convenience wrapper: make and optionally save one plot + return summary.
    """
    out = {}
    save_path = None
    if save:
        save_path = str(Path(output_root) / f"correctness_by_matching_{metric_name}.png")

    summary = plot_correctness_by_matching(
        df=trial_df,
        metric_name=metric_name,
        save_path=save_path,
    )

    out["summary"] = summary
    out["save_path"] = save_path
    return out
