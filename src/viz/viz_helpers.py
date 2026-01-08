# src/viz/viz_helpers.py
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def p_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_wilson_errorbars_and_ns(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    y: str = "accuracy",
    n_col: str = "n",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    show_n: bool = True,
) -> None:
    """
    Draw Wilson CI error bars and (optionally) add n=... labels above bars.

    Expects columns:
      - y (default: 'accuracy')
      - ci_low_col (default: 'ci_low')
      - ci_high_col (default: 'ci_high')
      - n_col (default: 'n') if show_n=True
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
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=summary_df, x="group", y="accuracy", order=list(order), ax=ax)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, ax


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot_and_report(
    *,
    fig: plt.Figure,
    summary_df: pd.DataFrame,
    test_res: Optional[Dict],
    plot_dir: str,
    data_dir: str,
    base_name: str,
) -> None:
    ensure_dir(plot_dir)
    ensure_dir(data_dir)

    fig.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=300)
    summary_df.to_csv(os.path.join(data_dir, f"{base_name}__summary.csv"), index=False)

    if test_res is not None:
        test_json = dict(test_res)
        if "contingency_table" in test_json and hasattr(test_json["contingency_table"], "tolist"):
            test_json["contingency_table"] = test_json["contingency_table"].tolist()

        with open(os.path.join(data_dir, f"{base_name}__fisher.json"), "w", encoding="utf-8") as f:
            json.dump(test_json, f, indent=2)
