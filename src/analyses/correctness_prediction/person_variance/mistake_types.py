"""Which kind of mistake the model makes, and what goes with each kind.

`accuracy_characterization` relates features to one balanced accuracy per person.
This module splits that by the kind of mistake, using the confusion matrix of the
per-participant models' predictions (each trial predicted by a model trained on
the same person's other trials; see `loo_runs`). `is_correct = 1` means the
participant answered correctly:

                   | model says correct    | model says wrong
    human correct  | TP - correct on right | FN - miss
    human wrong    | FP - false alarm      | TN - correct on wrong

Two views:

1. **Per trial** (`trial_feature_profile_by_quadrant`): pool every scored trial and
   compare the features of mistakes with those of hits *on the same kind of
   answer*. Each comparison holds the participant's answer fixed and varies only
   whether the model got it right:

   * `FN - TP | human correct`: among correctly answered trials, how a trial the
     model **missed** differs from one it **caught**.
   * `FP - TN | human wrong`: among wrongly answered trials, how a **false alarm**
     differs from a wrong answer the model flagged.

2. **Per person** (`characterize_accuracy_by_error_type`): split balanced accuracy
   into its two halves -- how often the model is right on correct answers
   (sensitivity) and on wrong ones (specificity) -- and correlate each feature
   with each half. A feature whose bars point the same way for both goes with
   better predictions overall; one that splits goes with the model's skill on one
   kind of answer only.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.config.columns as Con
from src.analyses.correctness_prediction.person_variance.accuracy_characterization import (
    MIN_WRONG_TRIALS,
    per_person_feature_means,
)
from src.analyses.correctness_prediction.person_variance.plot_style import (
    NEG_COLOR,
    POS_COLOR,
    QUAD_COLORS,
    signed_bar_colors,
)
from src.lib.plotting.output import save_output

QUAD_ORDER = ["TP", "TN", "FP", "FN"]

QUAD_LABELS = {
    "TP": "TP - correct on right",   # human right, model says right
    "TN": "TN - correct on wrong",   # human wrong, model says wrong
    "FP": "FP - false alarm",        # human wrong, model says right
    "FN": "FN - miss",               # human right, model says wrong
}


def build_trial_prediction_table(results_by_pid: Mapping[str, Any]) -> pd.DataFrame:
    """Every scored trial of every participant in one table, with its quadrant.

    Each result's `test_df` holds the scored trials, with all their feature
    columns, in the same order as its `y_true` / `y_pred` / `y_prob`, so no join
    with the feature table is needed. Adds `y_true`, `y_pred`, `y_prob`,
    `quadrant` and `model_correct`.
    """
    frames = []
    for pid, res in results_by_pid.items():
        td = res.test_df.reset_index(drop=True).copy()
        assert len(td) == len(res.y_true), f"{pid}: test_df / y_true length mismatch"
        prob = np.asarray(res.y_prob, dtype=float)
        if prob.ndim > 1:                       # (n, 2): keep P(correct)
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
    """Count the pooled trials in each quadrant. With `verbose`, also print the
    shares, and how the errors split between false alarms and misses."""
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "mistake_types",
    plot: str = "confusion_quadrants",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Bar chart of `summarize_confusion_quadrants`'s counts, saved with them.

    Returns `(fig, saved)`, `saved` being what `save_output` wrote.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar([QUAD_LABELS[q] for q in QUAD_ORDER],
           [counts[q] for q in QUAD_ORDER],
           color=[QUAD_COLORS[q] for q in QUAD_ORDER])
    ax.set_ylabel("trials")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    saved = save_output(
        fig,
        analysis="correctness_prediction/person_variance",
        save=save,
        subdir=subdir, plot=plot,
        tables={"counts": counts.rename("n_trials").reset_index(names="quadrant")},
        to_paper=to_paper, dpi=dpi, close=close,
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "mistake_types",
    plot: str = "mistake_types",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Each feature's average in each quadrant, and the two mistake-vs-hit contrasts.

    Features are standardised over all pooled trials (a feature with no spread gets
    0), so a quadrant's average reads as how far its trials sit from the typical
    trial, in standard deviations, and each contrast as how a mistake differs from a
    hit on the same kind of answer. One bar chart per contrast, strongest
    `top_n_bars` shown.

    Returns a dict: `mean_z` (features x quadrants, plus the two contrasts), `figs`
    and `saved_paths` (what `save_output` wrote, per figure).
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
        saved_paths[key] = save_output(
            fig,
            analysis="correctness_prediction/person_variance",
            plot=plot,
            tables={"mean_z": mean_z.reset_index(names="feature")},
            save=save,
            subdir=subdir,
            label=label or None,
            tag=key,
            to_paper=to_paper, dpi=dpi, close=close,
        )

    return {"mean_z": mean_z, "figs": figs, "saved_paths": saved_paths}


# ---------------------------------------------------------------------------
# View 2 - per person
# ---------------------------------------------------------------------------


def per_person_confusion(results_by_pid: Mapping[str, Any]) -> pd.DataFrame:
    """Each participant's confusion counts, and the two halves of balanced accuracy.

        sensitivity ("correct on right") = TP / (TP + FN), over correctly answered trials
        specificity ("correct on wrong") = TN / (TN + FP), over wrongly answered trials
        balanced_accuracy                = the average of the two

    A half with no trials behind it is NaN, and `balanced_accuracy` is then the
    other half alone, because the average skips NaN. `per_person_summary` gives NaN
    in that case instead.
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
    save: Optional[bool] = None,
    subdir: Optional[str] = "mistake_types",
    plot: str = "mistake_types",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
) -> Dict[str, Any]:
    """Correlate each feature's per-person average with sensitivity and with specificity.

    `conf_df` is `per_person_confusion`'s table. Each half has its own minimum:
    sensitivity needs at least `min_class_trials` correctly answered trials, and
    specificity as many wrongly answered ones -- the `MIN_WRONG_TRIALS` threshold of
    `accuracy_characterization`, applied to each outcome. So the two halves rest on
    different numbers of participants. `r_balacc`, the Spearman correlation with
    balanced accuracy, is there for reference and uses the wrong-answer minimum.
    Draws paired bars for the strongest `top_n_bars` features, saved with the table.

    Returns a dict: `merged`, `corr`, `fig` and `saved_paths` (what `save_output`
    wrote).
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

    saved = save_output(
        fig,
        analysis="correctness_prediction/person_variance",
        save=save,
        subdir=subdir,
        plot=plot,
        tables={"by_error_type": out},
        tag=f"{label.replace(' ', '_') or 'error_type'}_split",
        to_paper=to_paper, dpi=dpi, close=close,
    )
    return {"merged": merged, "corr": out, "fig": fig, "saved_paths": saved}
