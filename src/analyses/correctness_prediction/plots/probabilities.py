"""The model's predicted probability of a correct answer, split by whether the
answer was in fact correct.

The paper reads this probability as a stand-in for how confident the participant
was, and this is the figure that reading rests on. `run.py` draws it for the test
trials of every bundle.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, List, Sequence, Union, Dict, Any, Tuple, Mapping

import json

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.config.columns as Con
from src.lib.plotting.output import save_output

from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

from src.modeling.evaluate import (
    CorrectnessEvaluationResult,
)
from src.lib.plotting.output import save_output


def plot_predicted_probability_hist(
    y_true,
    y_prob,
    *,
    bins: int = 30,
    density: bool = False,
    show_means: bool = True,
    title: str = "Predicted probability of correctness by true outcome",
    figsize: Tuple[int, int] = (8, 5),
    correct_color: str = "#2c7fb8",
    wrong_color: str = "#de2d26",
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "predicted_probabilities",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """
    Two overlaid histograms of the predicted probability of a correct answer: one
    for the trials answered correctly, one for those answered wrongly.

    The better the model tells the two apart, the closer the correct trials sit to
    1 and the wrong ones to 0. `y_prob` is the probability per trial (a result's
    `y_prob`) and `y_true` the outcome (1 = correct). Trials with no probability
    (NaN) are left out without a message. The bins span 0 to 1. `density` scales
    each group to the same area, so groups of different sizes can be compared;
    `show_means` marks each group's mean with a dashed line.

    The figure and a two-row table (count, mean and median probability per group)
    are saved as `plot`, named by `**facets`. Returns the figure, the table and the
    saved paths.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    prob_correct = y_prob[y_true == 1]
    prob_wrong = y_prob[y_true == 0]

    fig, ax = plt.subplots(figsize=figsize)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ax.hist(
        prob_correct,
        bins=bin_edges,
        alpha=0.6,
        density=density,
        color=correct_color,
        label=f"Actually correct (true=1, n={len(prob_correct)})",
    )
    ax.hist(
        prob_wrong,
        bins=bin_edges,
        alpha=0.6,
        density=density,
        color=wrong_color,
        label=f"Actually wrong (true=0, n={len(prob_wrong)})",
    )

    if show_means:
        if len(prob_correct):
            ax.axvline(
                prob_correct.mean(), color=correct_color, linestyle="--", linewidth=1.5
            )
        if len(prob_wrong):
            ax.axvline(
                prob_wrong.mean(), color=wrong_color, linestyle="--", linewidth=1.5
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted probability of correctness")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()

    summary_df = pd.DataFrame(
        [
            {
                "group": "actually_correct",
                "true_label": 1,
                "n": int(len(prob_correct)),
                "mean_pred_prob": float(prob_correct.mean()) if len(prob_correct) else np.nan,
                "median_pred_prob": float(np.median(prob_correct)) if len(prob_correct) else np.nan,
            },
            {
                "group": "actually_wrong",
                "true_label": 0,
                "n": int(len(prob_wrong)),
                "mean_pred_prob": float(prob_wrong.mean()) if len(prob_wrong) else np.nan,
                "median_pred_prob": float(np.median(prob_wrong)) if len(prob_wrong) else np.nan,
            },
        ]
    )

    if filename is None:
        filename = "predicted_probability_hist"

    saved_paths = save_output(
        fig,
        analysis="correctness_prediction",
        plot=plot,
        tables={"data": summary_df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
        **facets,
    ).paths

    return fig, summary_df, saved_paths
