"""How well each participant's own model predicts their answers.

Each participant has a model of their own, and each of their trials is predicted
by a version trained on their other trials only (one trial left out at a time,
"LOO"; see `loo_runs`). This module gives one row per participant -- how many
trials were scored, how many of those were answered correctly and wrongly, and the
share the model got right, raw and balanced -- and a histogram of those scores.

Balanced accuracy averages how often the model is right on correct answers and how
often on wrong ones, so a model that calls every answer correct scores 0.5, however
few wrong answers the person gave. It is NaN when all of a person's scored trials
have the same outcome. That happens to someone with a single wrong answer: with
that trial left out, only correct answers remain to train on, so it is never
scored.
"""

from __future__ import annotations

from typing import Optional, Any, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import src.config.columns as Con
from src.analyses.correctness_prediction.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
)
from src.lib.plotting.output import save_output


def per_person_summary(
    results_by_pid: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per participant: trial counts, and the share their model got right.

    The counts are of the scored trials: all of a person's trials, except any whose
    removal would leave only one outcome to train on. Columns: `participant_id`,
    `n_trials`, `n_correct_trials`, `n_wrong_trials`, `accuracy`,
    `balanced_accuracy`. Best balanced accuracy first, NaN last.
    """
    rows = []
    for pid, res in results_by_pid.items():
        y_true, y_pred = res.y_true, res.y_pred
        has_both_classes = len(np.unique(y_true)) > 1
        rows.append({
            Con.PARTICIPANT_ID: pid,
            "n_trials": res.n_test,
            "n_correct_trials": res.n_positive,
            "n_wrong_trials": res.n_negative,
            "accuracy": res.accuracy,
            "balanced_accuracy": (
                balanced_accuracy_score(y_true, y_pred) if has_both_classes else np.nan
            ),
        })
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print("Mean per-person accuracy:          ", round(summary_df["accuracy"].mean(), 4))
        print("Mean per-person balanced accuracy: ",
              round(summary_df["balanced_accuracy"].mean(), 4))

    return summary_df


def plot_per_person_accuracy_hist(
    summary_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy"),
    bins: int = 15,
    save: Optional[bool] = None,
    subdir: Optional[str] = "accuracy",
    plot: str = "per_person_accuracy_hist",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Histogram of each accuracy across participants, with its mean and a line at 0.5.

    The 0.5 line is labelled chance, which holds for balanced accuracy only: for raw
    accuracy, calling every answer correct already scores the share of correct
    answers (about 0.84 on Study 1). Saves the summary table with the figure.
    Returns `(fig, saved)`, `saved` being what `save_output` wrote.
    """
    colors = [POS_COLOR, NEG_COLOR]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    axes = np.atleast_1d(axes)

    for ax, col, color in zip(axes, metrics, colors * len(metrics)):
        vals = summary_df[col].dropna()
        ax.hist(vals, bins=bins, color=color, edgecolor="white")
        ax.axvline(vals.mean(), color="black", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
        ax.axvline(0.5, color="grey", ls=":", lw=1, label="chance")
        ax.set_title(f"Per-person {col.replace('_', ' ')}")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("participants")
        ax.legend()

    plt.tight_layout()
    saved = save_output(
        fig,
        analysis="correctness_prediction/person_variance",
        save=save,
        subdir=subdir, plot=plot,
        tables={"summary": summary_df},
        to_paper=to_paper, dpi=dpi, close=close,
    )
    return fig, saved
