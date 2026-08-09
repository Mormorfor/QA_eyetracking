"""Mistake types -- decomposing the feature/accuracy association.

:mod:`accuracy_characterization` relates features to a single balanced accuracy
per person. Here that is broken down by *what kind of mistake* the model makes,
using the confusion matrix (``is_correct = 1`` means the human answered
correctly):

                   | model says correct    | model says wrong
    human correct  | TP - correct on right | FN - miss
    human wrong    | FP - false alarm      | TN - correct on wrong

Two complementary views:

1. **Trial-level** (:func:`trial_feature_profile_by_quadrant`) -- pool all
   held-out trials, label each quadrant, and compare feature profiles between
   mistakes and hits *of the same true label*. Each contrast holds the human's
   answer fixed and varies only whether the model got it right:

   * ``FN - TP | human correct``: among trials the human answered **correctly**,
     how a **missed** trial's features differ from a **caught** one.
   * ``FP - TN | human wrong``: among trials the human answered **wrong**, how a
     **false alarm** differs from a correctly-flagged wrong answer.

2. **Per-person** (:func:`characterize_accuracy_by_error_type`) -- split balanced
   accuracy into sensitivity ("correct on right") and specificity ("correct on
   wrong") and rerun the per-feature correlation for each half. A feature whose
   bars point the same way for both helps everywhere; one that splits only
   tracks the model's skill at one kind of trial.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness.person_variance.accuracy_characterization import (
    MIN_WRONG_TRIALS,
    per_person_feature_means,
)
from predictive_modeling.answer_correctness.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
    QUAD_COLORS,
    signed_bar_colors,
)
from predictive_modeling.common.viz_utils import maybe_save_plot

QUAD_ORDER = ["TP", "TN", "FP", "FN"]

QUAD_LABELS = {
    "TP": "TP - correct on right",   # human right, model says right
    "TN": "TN - correct on wrong",   # human wrong, model says wrong
    "FP": "FP - false alarm",        # human wrong, model says right
    "FN": "FN - miss",               # human right, model says wrong
}


def build_trial_prediction_table(results_by_pid: Mapping[str, Any]) -> pd.DataFrame:
    """Pool every held-out trial across participants with its confusion quadrant.

    Each result's ``test_df`` is row-aligned with ``y_true``/``y_pred``/``y_prob``
    and already carries the feature columns + participant id, so no re-merge with
    the feature table is needed (which also guarantees exact row alignment).
    """
    frames = []
    for pid, res in results_by_pid.items():
        td = res.test_df.reset_index(drop=True).copy()
        assert len(td) == len(res.y_true), f"{pid}: test_df / y_true length mismatch"
        prob = np.asarray(res.y_prob, dtype=float)
        if prob.ndim > 1:                       # (n, 2) -> positive-class column
            prob = prob[:, -1]
        td[Con.PARTICIPANT_ID] = pid
        td["y_true"] = np.asarray(res.y_true).astype(int)
        td["y_pred"] = np.asarray(res.y_pred).astype(int)
        td["y_prob"] = prob
        frames.append(td)
    trials = pd.concat(frames, ignore_index=True)

    yt, yp = trials["y_true"].to_numpy(), trials["y_pred"].to_numpy()
    trials["quadrant"] = np.select(
        [(yt == 1) & (yp == 1), (yt == 0) & (yp == 0),
         (yt == 0) & (yp == 1), (yt == 1) & (yp == 0)],
        QUAD_ORDER, default="?",
    )
    trials["model_correct"] = yt == yp
    return trials


def summarize_confusion_quadrants(
    trials: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.Series:
    """Count the pooled held-out trials per confusion quadrant, and (optionally)
    print the shares plus the false-alarm / miss split of the errors."""
    counts = (
        trials["quadrant"].value_counts().reindex(QUAD_ORDER).fillna(0).astype(int)
    )
    if verbose:
        n_trials = len(trials)
        print(f"{n_trials} pooled held-out trials from "
              f"{trials[Con.PARTICIPANT_ID].nunique()} participants")
        print("\nConfusion quadrants (pooled over all trials):")
        for q in QUAD_ORDER:
            print(f"  {QUAD_LABELS[q]:22s} {counts[q]:6d}  ({counts[q] / n_trials:6.1%})")
        n_err = int(counts["FP"] + counts["FN"])
        print(f"\nErrors: {n_err} total  |  FP false alarms {counts['FP'] / n_err:.1%}"
              f"  |  FN misses {counts['FN'] / n_err:.1%}")
    return counts


def plot_confusion_quadrants(
    counts: pd.Series,
    *,
    title: str = "Confusion quadrants (pooled held-out trials)",
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/mistake_types",
    filename: str = "confusion_quadrants",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Bar chart of the quadrant counts from :func:`summarize_confusion_quadrants`."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar([QUAD_LABELS[q] for q in QUAD_ORDER],
           [counts[q] for q in QUAD_ORDER],
           color=[QUAD_COLORS[q] for q in QUAD_ORDER])
    ax.set_ylabel("trials")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved


# ---------------------------------------------------------------------------
# View 1 - trial level
# ---------------------------------------------------------------------------


def trial_feature_profile_by_quadrant(
    trials: pd.DataFrame,
    characterize_cols: Sequence[str],
    *,
    top_n_bars: int = 25,
    label: str = "",
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/mistake_types",
    filename_prefix: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Standardized mean feature value per confusion quadrant, plus the two
    label-fixed error contrasts.

    Standardization is over *all pooled trials*, so a quadrant's mean-z reads as
    "how far this group sits from the typical trial", and each contrast bar as
    "what a mistake trial looks like versus a hit of the same kind".

    Returns a dict with ``mean_z`` (quadrant columns + the two contrast columns),
    ``figs`` and ``saved_paths``.
    """
    cols = list(characterize_cols)
    X = trials[cols].apply(pd.to_numeric, errors="coerce")
    z = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    z["quadrant"] = trials["quadrant"].to_numpy()

    mean_z = z.groupby("quadrant")[cols].mean().T.reindex(columns=QUAD_ORDER)
    contrasts = {
        "FN - TP  | human correct  (+) = elevated on misses": mean_z["FN"] - mean_z["TP"],
        "FP - TN  | human wrong    (+) = elevated on false alarms": mean_z["FP"] - mean_z["TN"],
    }

    prefix = filename_prefix or (label.replace(" ", "_") or "quadrant_profile")
    figs: Dict[str, Any] = {}
    saved_paths: Dict[str, List[str]] = {}

    for key, (name, series) in zip(["fn_minus_tp", "fp_minus_tn"], contrasts.items()):
        mean_z[name] = series
        d = series.dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(top_n_bars)
        d = d.sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(d) + 1)))
        ax.barh(d.index, d.values, color=signed_bar_colors(d.values))
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("standardized mean difference (z)")
        ax.set_title(f"[{label}] {name}")
        plt.tight_layout()

        figs[key] = fig
        saved_paths[key] = maybe_save_plot(
            fig=fig, save=save, rel_dir=rel_dir, filename=f"{prefix}_{key}",
            paper_dirs=paper_dirs, dpi=dpi, close=close,
        )

    return {"mean_z": mean_z, "figs": figs, "saved_paths": saved_paths}


# ---------------------------------------------------------------------------
# View 2 - per person
# ---------------------------------------------------------------------------


def per_person_confusion(results_by_pid: Mapping[str, Any]) -> pd.DataFrame:
    """Per-person confusion counts + the two halves of balanced accuracy.

    ``sensitivity`` ("correct on right")  = TP / (TP + FN)  over human-correct trials
    ``specificity`` ("correct on wrong")  = TN / (TN + FP)  over human-wrong trials
    ``balanced_accuracy`` = (sensitivity + specificity) / 2
    """
    rows = []
    for pid, res in results_by_pid.items():
        yt, yp = np.asarray(res.y_true), np.asarray(res.y_pred)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        rows.append({
            Con.PARTICIPANT_ID: pid,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "n_correct_trials": tp + fn,
            "n_wrong_trials": tn + fp,
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        })
    df = pd.DataFrame(rows)
    df["balanced_accuracy"] = df[["sensitivity", "specificity"]].mean(axis=1)
    return df


def characterize_accuracy_by_error_type(
    conf_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    characterize_cols: Sequence[str],
    *,
    min_class_trials: int = MIN_WRONG_TRIALS,
    top_n_bars: int = 25,
    label: str = "",
    verbose: bool = True,
    save: bool = False,
    rel_dir: str = "answer_correctness/per_person_loo/mistake_types",
    filename: Optional[str] = None,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Spearman corr of each per-person mean feature with sensitivity vs
    specificity, side by side (plus balanced accuracy for reference).

    Each metric gets its OWN estimability filter: sensitivity needs enough
    human-correct trials, specificity enough human-wrong trials -- the same
    ``MIN_WRONG_TRIALS`` guard :mod:`accuracy_characterization` uses, applied per
    class. ``n`` therefore differs between the halves.

    Returns a dict with ``merged``, ``corr``, ``fig`` and ``saved_paths``.
    """
    cols = list(characterize_cols)
    merged = conf_df.merge(
        per_person_feature_means(trial_df, cols), on=Con.PARTICIPANT_ID, how="inner"
    )

    def _corr(sub, metric):
        return [sub[[metric, f]].dropna().corr(method="spearman").iloc[0, 1] for f in cols]

    out = pd.DataFrame({"feature": cols})
    for metric, count_col in [("sensitivity", "n_correct_trials"),
                              ("specificity", "n_wrong_trials")]:
        sub = merged[merged[count_col] >= min_class_trials]
        if verbose:
            print(f"[{label}] {metric:12s}: {len(sub)} participants "
                  f"({count_col} >= {min_class_trials})")
        out[f"r_{metric}"] = _corr(sub, metric)
    out["r_balacc"] = _corr(
        merged[merged["n_wrong_trials"] >= min_class_trials], "balanced_accuracy"
    )

    out["max_abs"] = out[["r_sensitivity", "r_specificity"]].abs().max(axis=1)
    out = out.sort_values("max_abs", ascending=False).reset_index(drop=True)

    # paired horizontal bars: sensitivity vs specificity per feature
    d = out.head(top_n_bars).iloc[::-1]
    y = np.arange(len(d))
    h = 0.4
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(d) + 1)))
    ax.barh(y + h / 2, d["r_sensitivity"], height=h, color=POS_COLOR,
            label="sensitivity (TP / (TP + FN))")
    ax.barh(y - h / 2, d["r_specificity"], height=h, color=NEG_COLOR,
            label="specificity (TN / (TN + FP))")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("Spearman corr of per-person mean feature with metric")
    ax.set_title(f"[{label}] feature association split by mistake type (top {len(d)})")
    ax.legend(loc="lower right")
    plt.tight_layout()

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir,
        filename=filename or f"{label.replace(' ', '_') or 'error_type'}_split",
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return {"merged": merged, "corr": out, "fig": fig, "saved_paths": saved}
