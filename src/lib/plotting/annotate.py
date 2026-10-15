"""Small additions to accuracy bar charts: error bars, n labels, significance brackets."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.lib.stats.proportions import p_to_stars, wilson_ci


def add_wilson_errorbars_and_ns(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    y: str = "accuracy",
    n_col: str = "n",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    show_n: bool = True,
) -> None:
    """Draw confidence-interval error bars on each bar, and "n=..." above it.

    `summary_df` has one row per bar, in bar order, with the bar's value (`y`),
    its interval (`ci_low_col`, `ci_high_col`) and its count (`n_col`). Bars
    with a missing value or interval get no error bar.
    """
    for i, r in summary_df.reset_index(drop=True).iterrows():
        acc = r[y]

        if np.isfinite(r[ci_low_col]) and np.isfinite(r[ci_high_col]) and np.isfinite(acc):
            ax.errorbar(
                i,
                acc,
                yerr=[[acc - r[ci_low_col]], [r[ci_high_col] - acc]],
                fmt="none",
                capsize=4,
                ecolor="black",
                elinewidth=1.5,
            )

        if show_n and np.isfinite(acc):
            ax.text(
                i,
                min(0.98, acc + 0.03),
                f"n={int(r[n_col])}",
                ha="center",
                va="bottom",
            )


def add_significance_bracket(ax: plt.Axes, stars: str, x1: int = 0, x2: int = 1) -> None:
    """Draw a bracket joining bars x1 and x2, labelled with `stars`.

    The bracket goes just above the tallest bar, and the y-axis is extended if
    it would not fit.
    """
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    # pick a y above max bar height
    heights = [p.get_height() for p in ax.patches]  # bars
    y_max = float(np.nanmax(heights)) if heights else 0.0

    y = y_max + 0.06 * yr
    h = 0.03 * yr

    # expand ylim so bracket fits
    if y + h + 0.05 * yr > ymax:
        ax.set_ylim(ymin, y + h + 0.08 * yr)

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.01 * yr, stars, ha="center", va="bottom", color="black", clip_on=False)


def barplot_accuracy(summary_df: pd.DataFrame, order: Sequence[str], figsize=(6, 4)) -> Tuple[plt.Figure, plt.Axes]:
    """Plain bar chart of `accuracy` per `group`, in the given order, y from 0 to 1."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=summary_df, x="group", y="accuracy", order=list(order), ax=ax)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, ax
