"""The four confusion cases, named once for the whole project.

Every scored trial carries two yes/no facts: whether the participant answered
correctly, and whether the model said they would. Crossing them gives:

                   | model says correct | model says wrong
    human correct  | TP                 | FN
    human wrong    | FP                 | TN

**"Positive" means the participant got it right**, not that the model fired. So a
false negative is a trial the person answered correctly and the model called
wrong. Both analyses below turn on reading it that way.

Two places label trials like this, and they have to mean the same thing by it:

    analyses/error_analysis            one model trained on everyone, scored on
                                       held-out cross-validation folds
    .../person_variance/mistake_types  one model per participant, scored
                                       leave-one-trial-out

They build their tables differently, so the naming lives here instead of in
either of them. It used to be two copies of the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The order used by every table, bar chart and legend. Hits first, so the two
# kinds of error sit together at the end.
QUAD_ORDER = ["TP", "TN", "FP", "FN"]

QUAD_LABELS = {
    "TP": "TP - true positive",    # human right, model says right
    "TN": "TN - true negative",    # human wrong, model says wrong
    "FP": "FP - false positive",   # human wrong, model says right
    "FN": "FN - false negative",   # human right, model says wrong
}

# Hits in cool colours, errors in warm ones, so the two are told apart without
# reading the labels.
QUAD_COLORS = {
    "TP": "#4C72B0",
    "TN": "#55A868",
    "FP": "#DD8452",
    "FN": "#C44E52",
}


def label_quadrants(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Add `quadrant` and `model_correct` to a table of scored trials.

    Takes one row per trial with a true and a predicted label, both 0 or 1, and
    returns a copy.

    Stops if either column holds anything else. A missing value, or a probability
    passed in by mistake, would otherwise land in a fifth category of its own and
    be counted as a result.
    """
    out = df.copy()
    yt = out[y_true_col].to_numpy()
    yp = out[y_pred_col].to_numpy()

    for name, values in ((y_true_col, yt), (y_pred_col, yp)):
        bad = ~np.isin(values, [0, 1])
        if bad.any():
            raise ValueError(
                f"{name} must be 0 or 1; got {np.unique(values[bad])[:5]} "
                f"on {int(bad.sum())} of {len(values)} rows."
            )

    # The four conditions cover every 0/1 pair, which the check above makes sure
    # of, so the default never fires. It is a string because `np.select` needs
    # one of the same type as the choices.
    out["quadrant"] = np.select(
        [(yt == 1) & (yp == 1), (yt == 0) & (yp == 0),
         (yt == 0) & (yp == 1), (yt == 1) & (yp == 0)],
        QUAD_ORDER,
        default="",
    )
    out["model_correct"] = yt == yp
    return out


def confusion_counts(trials: pd.DataFrame, *, quadrant_col: str = "quadrant") -> pd.Series:
    """Count the trials in each quadrant, in `QUAD_ORDER`.

    A quadrant with no trials gives 0 rather than being left out: the model made
    no error of that kind, which is a number worth keeping.
    """
    return (
        trials[quadrant_col]
        .value_counts()
        .reindex(QUAD_ORDER)
        .fillna(0)
        .astype(int)
    )


def counts_to_frame(counts: pd.Series) -> pd.DataFrame:
    """The counts as a table to save: one row per quadrant, with its share."""
    n = int(counts.sum())
    return pd.DataFrame(
        {
            "quadrant": QUAD_ORDER,
            "label": [QUAD_LABELS[q] for q in QUAD_ORDER],
            "n_trials": [int(counts[q]) for q in QUAD_ORDER],
            "share": [counts[q] / n if n else np.nan for q in QUAD_ORDER],
        }
    )
