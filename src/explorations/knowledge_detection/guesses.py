"""Among right answers: known, or a lucky guess?

Telling full knowledge from none (`run.py`), most trials sort themselves: a wrong answer is
almost always a no-knowledge trial (393 of 409), and a right answer given with full
knowledge is rarely missed. The hard trials are the right answers given without knowledge:
232 lucky guesses, against 624 right answers given with full knowledge. Here the question is
asked of right answers alone -- which were known, and which guessed.

Two changes to `run.py`'s setup are tried, alone and together, on three feature sets
(`FEATURE_SETS`):

* **Training on right answers only.** `run.py` trains on every answer, so much of what its
  model learns separates wrong answers from right ones. Trained on right answers alone, the
  model can learn only what separates a guess from a known answer. Both are scored on the
  same right answers of the same held-out people.
* **Centring each feature within the person.** People differ in pace, and response time is
  the main cue, so a quick guesser looks like a knower. Each feature becomes its distance
  from the person's own mean, over all their trials in the modelled regimes, right and
  wrong. That uses no labels, but it does use the held-out person's own trials: in use it
  needs a batch of trials from the person, in a known mix of conditions. Here the design
  sets the mix, about half each. Features with missing values -- pupil size, already
  z-scored per person, and first-fixation duration -- are left as they are, so the model's
  fill of missing values is not changed.

**Scores.** The within-person AUC (`run.score_tables`): within each held-out person, how
well the probability puts their known answers above their guesses, averaged over people.
And at the model's own cut-off: the share of guesses called guesses, the share of known
answers called known, and their mean, the balanced accuracy. Calling more answers guesses
always raises the first share and lowers the second, so variants are compared on the AUC and
the balanced accuracy; the two shares show where the errors fall. Intervals and p resample
participants, with the same draws for every variant, as in `run.py`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.lib.plotting.output import save_output
from src.explorations.knowledge_detection.run import (
    ANALYSIS,
    BOTH,
    MODELLED,
    PLOT,
    REGIME_COL,
    REGIME_COLOURS,
    SET_LABELS,
    WITH_KNOWLEDGE,
    accuracy_table,
    confusion_table,
    feature_sets,
    histogram_table,
    load_trials,
    lopo_predictions,
    modelled_trials,
    score_tables,
)

FEATURE_SETS = ("response_time", "paper_model", "all_features")
TRAINING = {"all": "trained on every answer", "right": "trained on right answers"}
CENTRING = {"raw": "as measured", "centred": "centred per person"}
SEP = "|"


def variant(features: str, training: str, centring: str) -> str:
    """The name of one variant: feature set, training answers, centring."""
    return SEP.join((features, training, centring))


# The order the figures show the variants in: a feature set per row of panels.
VARIANT_ORDER = [variant(f, t, c) for f in FEATURE_SETS for t in TRAINING for c in CENTRING]


# The variant differences tested: (variant, variant it is compared with, what it asks).
GUESS_PAIRS: List[Tuple[str, str, str]] = [
    (variant("all_features", "right", "raw"), variant("all_features", "all", "raw"),
     "training on right answers only, all features"),
    (variant("response_time", "right", "raw"), variant("response_time", "all", "raw"),
     "training on right answers only, response time"),
    (variant("all_features", "all", "centred"), variant("all_features", "all", "raw"),
     "centring per person, all features"),
    (variant("response_time", "all", "centred"), variant("response_time", "all", "raw"),
     "centring per person, response time"),
    (variant("all_features", "right", "centred"), variant("all_features", "all", "raw"),
     "both changes, all features"),
    (variant("paper_model", "right", "centred"), variant("paper_model", "all", "raw"),
     "both changes, the paper's features"),
    (variant("all_features", "right", "centred"), variant("response_time", "right", "centred"),
     "what gaze adds to response time, with both changes"),
]


def centre_within_person(trials: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """`trials` with each column of `cols` that has no missing value replaced by its
    distance from the participant's mean over `trials`; the other columns unchanged.
    Returns the frame and the columns centred."""
    complete = [c for c in cols if trials[c].notna().all()]
    values = trials[complete].apply(pd.to_numeric, errors="raise").astype(float)
    out = trials.copy()
    out[complete] = values - values.groupby(trials[Con.PARTICIPANT_ID]).transform("mean")
    return out, complete


def guess_predictions(trials: pd.DataFrame, sets: Dict[str, List[str]], verbose: bool = True) -> pd.DataFrame:
    """Every variant's predictions on the right answers of held-out participants: each
    feature set of `FEATURE_SETS`, trained on every answer or on right answers only, on the
    features as measured or centred per person. One row per variant and right answer, with
    `features`, `training` and `centring` saying which."""
    cols = list(dict.fromkeys(c for s in FEATURE_SETS for c in sets[s]))
    centred, centred_cols = centre_within_person(trials, cols)
    if verbose:
        print(f"centred per person: {len(centred_cols)} of {len(cols)} columns; "
              f"left as measured (have missing values): {sorted(set(cols) - set(centred_cols))}")
    parts = []
    for centring, frame in (("raw", trials), ("centred", centred)):
        for training in TRAINING:
            fit_on = frame if training == "all" else frame[frame[Con.IS_CORRECT_COLUMN] == 1]
            preds = lopo_predictions(fit_on, {variant(s, training, centring): sets[s] for s in FEATURE_SETS})
            parts.append(preds[preds[Con.IS_CORRECT_COLUMN] == 1])
    preds = pd.concat(parts, ignore_index=True)
    preds[["features", "training", "centring"]] = preds["feature_set"].str.split(SEP, expand=True)
    # Every variant scored on the same right answers.
    per_variant = preds.groupby("feature_set").size()
    assert per_variant.nunique() == 1, f"variants scored on different trials: {per_variant.to_dict()}"
    return preds


def cutoff_tables(
    preds: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[str, str, str]] = GUESS_PAIRS,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """`cutoff`: per variant, at the model's own cut-off, the share of guesses called guesses,
    of known answers called known, and their mean (balanced accuracy), each pooled over
    people with its interval. `cutoff_pairs`: for each pair of `pairs`, the difference in
    each, with its interval and p. The same participant draws as `run.score_tables`."""
    people = np.sort(preds[Con.PARTICIPANT_ID].unique())
    draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
    weight = np.stack([np.bincount(d, minlength=len(people)) for d in draws])
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    def share(sub, label):
        """Pooled share of `sub`'s trials with true label `label` predicted as `label`."""
        part = sub[sub["y_true"] == label]
        who = pd.Index(people).get_indexer(part[Con.PARTICIPANT_ID])
        hits = np.bincount(who, weights=(part["y_pred"] == label).to_numpy(float), minlength=len(people))
        n = np.bincount(who, minlength=len(people)).astype(float)
        return hits.sum() / n.sum(), (weight @ hits) / (weight @ n)

    rows, boots, points = [], {}, {}
    for name, sub in preds.groupby("feature_set", sort=False):
        guess, guess_boot = share(sub, 0)
        known, known_boot = share(sub, 1)
        points[name] = {"guesses_called_guesses": guess, "known_called_known": known,
                        "balanced_accuracy": (guess + known) / 2}
        boots[name] = {"guesses_called_guesses": guess_boot, "known_called_known": known_boot,
                       "balanced_accuracy": (guess_boot + known_boot) / 2}
        row = {"feature_set": name, **dict(zip(("features", "training", "centring"), name.split(SEP))),
               "n_guesses": int((sub["y_true"] == 0).sum()), "n_known": int((sub["y_true"] == 1).sum())}
        for metric in points[name]:
            row |= {metric: points[name][metric],
                    f"{metric}_ci_low": np.percentile(boots[name][metric], lo_q),
                    f"{metric}_ci_high": np.percentile(boots[name][metric], hi_q)}
        rows.append(row)

    pair_rows = []
    for a, b, question in pairs:
        row = {"feature_set": a, "compared_with": b, "question": question}
        for metric in points[a]:
            diff = boots[a][metric] - boots[b][metric]
            row |= {f"{metric}_difference": points[a][metric] - points[b][metric],
                    f"{metric}_ci_low": np.percentile(diff, lo_q), f"{metric}_ci_high": np.percentile(diff, hi_q),
                    f"{metric}_p": float(np.clip(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / n_boot, 1))}
        pair_rows.append(row)

    out = {"cutoff": pd.DataFrame(rows), "cutoff_pairs": pd.DataFrame(pair_rows)}
    for t in out.values():
        t[["n_boot", "ci", "seed"]] = n_boot, ci, seed
        t[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def _label(name: str) -> str:
    features, training, centring = name.split(SEP)
    return f"{SET_LABELS[features]}\n{TRAINING[training]}, {CENTRING[centring]}"


def plot_guesses(
    scores: pd.DataFrame,
    cutoff: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Left: each variant's within-person AUC, known answers against lucky guesses, with 95%
    intervals. Right: at the model's cut-off, the share of guesses called guesses (red), of
    known answers called known (blue) and their mean (black). A variant per row, grouped by
    feature set. Saved with `tables`. Returns the figure."""
    order = VARIANT_ORDER
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.62 * len(order) + 1.8), sharey=True,
                                  layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    by_name = scores.set_index("feature_set")
    for i, name in enumerate(order):
        r = by_name.loc[name]
        ax.errorbar(r["person_auc"], i, xerr=[[r["person_auc"] - r["person_ci_low"]], [r["person_ci_high"] - r["person_auc"]]],
                    fmt="o", color="#333333", ms=6, elinewidth=1.4, capsize=0)
        ax.annotate(f"{r['person_auc']:.3f}", (r["person_ci_high"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.axvline(0.5, color="#555555", lw=1, ls="--")
    ax.set_xlabel("AUC within each held-out participant, averaged (0.5 = chance)")
    ax.set_title("Known answers against lucky guesses", fontsize=10)

    style = [("guesses_called_guesses", "#c44e52", "o", -0.2, "guesses called guesses"),
             ("known_called_known", "#4c72b0", "s", 0.0, "known answers called known"),
             ("balanced_accuracy", "#333333", "D", 0.2, "balanced accuracy (their mean)")]
    by_name = cutoff.set_index("feature_set")
    for metric, colour, marker, dy, label in style:
        for i, name in enumerate(order):
            r = by_name.loc[name]
            ax2.errorbar(r[metric], i + dy, xerr=[[r[metric] - r[f"{metric}_ci_low"]], [r[f"{metric}_ci_high"] - r[metric]]],
                         fmt=marker, color=colour, ms=5.5, elinewidth=1.3, capsize=0, label=label if i == 0 else None)
    ax2.axvline(0.5, color="#555555", lw=1, ls="--")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Share at the model's cut-off")
    ax2.set_title("At the model's cut-off", fontsize=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=9)

    for a in (ax, ax2):
        for k in range(1, len(FEATURE_SETS)):
            a.axhline(k * len(TRAINING) * len(CENTRING) - 0.5, color="#BBBBBB", lw=1)
        a.grid(axis="x", color="#E5E5E5", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(n) for n in order], fontsize=8.5)
    ax.invert_yaxis()
    n_guess, n_known = int(cutoff["n_guesses"].iloc[0]), int(cutoff["n_known"].iloc[0])
    fig.suptitle(f"Study 2, right answers only: known ({n_known}) or a lucky guess ({n_guess})? "
                 "Held-out participants", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=f"{PLOT}_guesses", tables=tables,
                save=save, close=close, subdir=model_folder(scores))
    return fig


def _variant_grid(height: float):
    """A panel per variant: a row per feature set, a column per training and centring."""
    ncol = len(TRAINING) * len(CENTRING)
    return plt.subplots(len(FEATURE_SETS), ncol, figsize=(3.4 * ncol + 1.4, height * len(FEATURE_SETS) + 1.0),
                        layout="constrained", squeeze=False)


def plot_guess_confusion(
    confusion: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """One heatmap per variant, drawn as `run.plot_confusion`: a row per regime (right
    answers only), the two columns what the model said, each cell the share of the row's
    trials with the count. Guesses belong in the left column, known answers in the right.
    Each title gives the accuracy and balanced accuracy on these right answers
    (`run.accuracy_table`). Saved with `confusion` and the accuracies. Returns the figure."""
    fig, axes = _variant_grid(2.2)
    knowledge = "knowledge" if BOTH else WITH_KNOWLEDGE[0]
    accuracy = accuracy_table(confusion)
    by_variant = accuracy.set_index("feature_set")
    for ax, name in zip(axes.flat, VARIANT_ORDER):
        rows = confusion[confusion["feature_set"] == name].set_index(REGIME_COL).loc[list(MODELLED)]
        share = np.column_stack([1 - rows["share_predicted_knowledge"], rows["share_predicted_knowledge"]])
        counts = rows[["predicted_none", "predicted_knowledge"]].to_numpy()
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(share.shape[0]):
            for j in range(2):
                ax.text(j, i, f"{100 * share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center",
                        fontsize=9, color="white" if share[i, j] > 0.6 else "#222222")
        a = by_variant.loc[name]
        ax.set_title(f"{_label(name)}\naccuracy {a['accuracy']:.3f}, balanced {a['balanced_accuracy']:.3f}",
                     fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["said guess", "said known"], fontsize=8.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{'guess' if r == MODELLED[0] else r}\n({int(rows.loc[r, 'n'])})" for r in rows.index],
                           fontsize=8.5)
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, shrink=0.5, label="share of the row's trials")
    fig.suptitle(f"Study 2, right answers, held-out participants: lucky guesses against {knowledge} "
                 "(guesses belong left, known answers right)", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=f"{PLOT}_guesses_confusion",
                tables={"confusion": confusion, "accuracy": accuracy},
                save=save, close=close, subdir=model_folder(confusion))
    return fig


def plot_guess_histograms(
    hist: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """One panel per variant, drawn as `run.plot_histograms`: the predicted probability of
    knowledge on right answers, a histogram per regime, each as shares of its own trials;
    the y scales differ between panels. The dotted line is the model's cut-off. Saved with
    `hist`. Returns the figure."""
    fig, axes = _variant_grid(2.2)
    knowledge = "knowledge" if BOTH else WITH_KNOWLEDGE[0]
    for ax, name in zip(axes.flat, VARIANT_ORDER):
        for regime in MODELLED:
            sub = hist[(hist["feature_set"] == name) & (hist[REGIME_COL] == regime)]
            edges = np.append(sub["bin_low"].to_numpy(), sub["bin_high"].iloc[-1])
            label = f"{'lucky guesses' if regime == MODELLED[0] else regime + ', right'} ({int(sub['n'].iloc[0])})"
            ax.stairs(sub["share"].to_numpy(), edges, color=REGIME_COLOURS[regime], lw=1.8, label=label)
        ax.axvline(0.5, color="#555555", lw=1, ls=":")
        ax.set_title(_label(name), fontsize=9)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"Predicted P({knowledge})")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of the group's trials")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(f"Study 2, right answers, held-out participants: predicted probability of {knowledge} "
                 "(dotted: the model's cut-off)", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=f"{PLOT}_guesses_histograms", tables={"histogram": hist},
                save=save, close=close, subdir=model_folder(hist))
    return fig


def main(save: bool = True, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Every variant's predictions, scores and tests, saved as a figure with its tables.
    Returns the tables and the predictions."""
    trials = modelled_trials(load_trials(verbose=verbose), verbose=verbose)
    sets = feature_sets(trials)
    preds = guess_predictions(trials, sets, verbose=verbose)
    out = score_tables(preds, pairs=GUESS_PAIRS)
    out |= cutoff_tables(preds)
    plot_guesses(out["scores"], out["cutoff"], dict(out), save=save, close=True)
    out["confusion"] = confusion_table(preds)
    plot_guess_confusion(out["confusion"], save=save, close=True)
    out["histogram"] = histogram_table(preds)
    plot_guess_histograms(out["histogram"], save=save, close=True)
    if verbose:
        auc = out["scores"].set_index("feature_set")[["person_auc", "person_ci_low", "person_ci_high", "pooled_auc"]]
        cut = out["cutoff"].set_index("feature_set")[["guesses_called_guesses", "known_called_known", "balanced_accuracy"]]
        print(auc.join(cut).round(3).to_string())
        show = ["question", "person_auc_difference", "person_p", "pooled_auc_difference", "pooled_p"]
        print(out["pairs"][show].round(4).to_string(index=False))
        show = ["question", "guesses_called_guesses_difference", "balanced_accuracy_difference", "balanced_accuracy_p"]
        print(out["cutoff_pairs"][show].round(4).to_string(index=False))
    out["predictions"] = preds
    return out


if __name__ == "__main__":
    main()
