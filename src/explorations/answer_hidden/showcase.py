"""Two models side by side, for showing: the headline model, and the chosen answer-hidden one.

* **answer revealed** -- the headline twelve-feature model (`select_1_plus_last_confirm`) on
  the main trial table, which tells it which answer is correct;
* **answer hidden** -- the chosen set of the *selected* naming
  (`deliberation.CHOSEN_NAME`) on the answer-hidden table.

Both go through the same machinery as the rest of this exploration: the paper's comparison
for the scores, `run.predictions_for` for trial-by-trial predictions, and
`discrimination.py` for the tests. Every result is saved and read back (`cache.py`).

The figures are kept simple, for slides. Study 1 is shown on its strictest held-out kind
throughout -- new participants reading new paragraphs (`both`) -- so every figure shows
the same trials; the other kinds give the same picture. Saved under
`reports/explorations/answer_hidden/{figures,tables}/<model>/showcase/`.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
import src.modeling.feature_sets as FS
from src.lib.plotting.output import save_output
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    CONFIDENCE_COL,
    PROB_COL,
    REGIME_COL,
    REGIME_ORDER,
)
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.deliberation import CHOSEN_NAME, chosen_cols
from src.explorations.answer_hidden.discrimination import (
    binned_probability,
    confidence_comparison,
    discrimination_table,
    knowqa_with_confidence,
)
from src.explorations.answer_hidden.run import (
    ANALYSIS,
    TEST_COLOURS,
    as_prediction_table,
    compare_in_batches,
    confusion_table,
    model_key,
    predictions_for,
)

KNOWN_NAME = "select_1_plus_last_confirm"
LABELS = {KNOWN_NAME: "answer revealed", CHOSEN_NAME: "answer hidden"}
COLOURS = {KNOWN_NAME: "#333333", CHOSEN_NAME: "#DD8452"}
SELF_REPORT_COLOUR = "#8172B3"

# Study 1's held-out kind shown, and the tests in the order the figures show them.
L1_TEST = "both"
TESTS = [("l1", L1_TEST), *(("knowqa", r) for r in REGIME_ORDER)]
TEST_LABELS = {
    L1_TEST: "Study 1\nnew people & paragraphs",
    "no knowledge": "Study 2\nno knowledge",
    "partial knowledge": "Study 2\npartial knowledge",
    "full knowledge": "Study 2\nfull knowledge",
}
SHORT_LABELS = {L1_TEST: "Study 1", "no knowledge": "no knowledge",
                "partial knowledge": "partial", "full knowledge": "full"}
# The paper's regime colours, and green for Study 1.
BOX_COLOURS = {**TEST_COLOURS, L1_TEST: "#55A868"}


def showcase_sets() -> Dict[str, list]:
    return {KNOWN_NAME: list(FS.SELECT_1_COLS) + list(FS.LAST_CONFIRM_COMPACT), CHOSEN_NAME: chosen_cols()}


def _table_for(name: str, tables) -> tuple:
    return (tables["l1"], tables["knowqa"]) if name == KNOWN_NAME else (tables["l1_hidden"], tables["knowqa_hidden"])


def showcase_results(tables, *, verbose: bool = True, rerun: bool = False) -> Dict[str, pd.DataFrame]:
    """The scores, predictions, discrimination tests and the comparison with the confidence
    rating, for both models. `tables` is `run.load_tables()`."""
    sets = showcase_sets()
    inputs = {
        "sets": sets, "model": model_key(TrialLevelLogRegModel),
        "tables": {k: table_digest(t) for k, t in tables.items()},
    }

    def scores():
        parts = [compare_in_batches({n: c}, *_table_for(n, tables), verbose=False) for n, c in sets.items()]
        return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in ("scores", "folds")}

    def predictions():
        preds = pd.concat(
            [predictions_for({n: c}, *_table_for(n, tables)) for n, c in sets.items()], ignore_index=True
        )
        preds["paper_model"] = "showcase"
        preds["answers"] = preds["model"].map({KNOWN_NAME: "known", CHOSEN_NAME: "selected"})
        return as_prediction_table(preds)

    comparison = cached("showcase_scores", inputs, scores, rerun=rerun, verbose=verbose)
    preds = cached("showcase_predictions", inputs, predictions, rerun=rerun, verbose=verbose)
    return {
        "scores": comparison["scores"],
        "folds": comparison["folds"],
        "predictions": preds,
        "discrimination": discrimination_table(preds, verbose=verbose, rerun=rerun),
        "vs_confidence": confidence_comparison(preds, tables["knowqa"], list(sets), verbose=verbose, rerun=rerun),
    }


def _save(fig, plot: str, tables: Dict[str, pd.DataFrame], save, close):
    folder = model_folder(next(iter(tables.values())))
    save_output(fig, analysis=ANALYSIS, plot=plot, tables=tables, save=save, close=close,
                subdir=f"{folder}/showcase")
    return fig


def _rows(df: pd.DataFrame, dataset: str, test: str) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["test"] == test)]


def plot_accuracy(results, *, save: Optional[bool] = None, close: Optional[bool] = None):
    """Balanced accuracy of both models on each test; Study 1 with its interval across the
    ten folds. The dashed line is chance."""
    scores = results["scores"]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(len(TESTS))
    shown = []
    for k, model in enumerate(LABELS):
        rows = [_rows(scores[scores["model"] == model], ds, t).iloc[0] for ds, t in TESTS]
        vals = np.array([r["balanced_accuracy"] for r in rows])
        lo = np.array([r["ci_low"] for r in rows], dtype=float)
        hi = np.array([r["ci_high"] for r in rows], dtype=float)
        pos = x + (k - 0.5) * 0.38
        ax.bar(pos, vals, width=0.36, color=COLOURS[model], label=LABELS[model], zorder=2)
        ok = np.isfinite(lo)
        ax.errorbar(pos[ok], vals[ok], yerr=[vals[ok] - lo[ok], hi[ok] - vals[ok]], fmt="none",
                    ecolor="black", capsize=3, zorder=3)
        for p, v, top in zip(pos, vals, np.where(np.isfinite(hi), hi, vals)):
            ax.text(p, top + 0.012, f"{v:.2f}", ha="center", fontsize=10)
        shown += [dict(r) for r in rows]
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TEST_LABELS[t] for _, t in TESTS], fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "showcase_accuracy", {"scores": pd.DataFrame(shown)}, save, close)


def plot_confusion(results, *, save: Optional[bool] = None, close: Optional[bool] = None):
    """The confusion matrix of both models on each test: a row per model, a column per test.
    Each cell gives the count and its share of the row's actual outcome, and is shaded by
    that share."""
    preds = results["predictions"]
    fig, axes = plt.subplots(2, len(TESTS), figsize=(13, 6.6), layout="constrained")
    saved = []
    for j, (ds, test) in enumerate(TESTS):
        table = confusion_table(preds, dataset=ds, test=test)
        for i, model in enumerate(LABELS):
            r = table[table["model"] == model].iloc[0]
            saved.append(dict(r))
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
            shown = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{int(cm[a, b])}\n{shown[a, b]:.2f}", ha="center", va="center",
                            fontsize=11, color="white" if shown[a, b] > 0.55 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred wrong", "pred right"] if i == 1 else [], fontsize=10)
            ax.set_yticklabels(["was wrong", "was right"] if j == 0 else [], fontsize=10)
            if j == 0:
                ax.set_ylabel(LABELS[model], fontsize=11)
            title = (TEST_LABELS[test].replace("\n", ": ") + "\n") if i == 0 else ""
            ax.set_title(f"{title}balanced acc {r['balanced_accuracy']:.2f}", fontsize=10.5)
    return _save(fig, "showcase_confusion", {"counts": pd.DataFrame(saved)}, save, close)


def plot_probabilities(results, *, save: Optional[bool] = None, close: Optional[bool] = None):
    """Predicted P(correct) as box plots, for answers that were right (coloured) and wrong
    (grey), on each test: a panel per model. Whiskers at 1.5 x the interquartile range, no
    outliers drawn; the dotted line at 0.5."""
    from matplotlib.cbook import boxplot_stats

    preds = results["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    saved = []
    for ax, model in zip(axes, LABELS):
        stats, positions, colours, labels = [], [], [], []
        for i, (ds, test) in enumerate(TESTS):
            for j, (outcome, tag) in enumerate([(1, "right"), (0, "wrong")]):
                sub = _rows(preds[preds["model"] == model], ds, test)
                sub = sub[sub["y_true"] == outcome]
                s = boxplot_stats(sub["y_prob"].to_numpy(float))[0]
                stats.append(s)
                positions.append(i * 2.6 + j * 0.9)
                colours.append(BOX_COLOURS[test] if outcome else "#cccccc")
                labels.append(f"{tag}\nn={len(sub)}")
                saved.append({"model": model, "dataset": ds, "test": test, "outcome": tag, "n": len(sub),
                              "median": s["med"], "q1": s["q1"], "q3": s["q3"], "mean": s["mean"],
                              "whisker_low": s["whislo"], "whisker_high": s["whishi"],
                              MODEL_FAMILY_COL: model_folder(sub)})
        bp = ax.bxp(stats, positions=positions, widths=0.75, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8.5)
        for i, (_, test) in enumerate(TESTS):
            ax.text(i * 2.6 + 0.45, 1.04, SHORT_LABELS[test], ha="center", fontsize=9.5)
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.set_ylim(-0.02, 1.1)
        ax.set_title(LABELS[model], fontsize=12, pad=18)
    axes[0].set_ylabel("predicted P(correct)")
    return _save(fig, "showcase_probabilities", {"boxes": pd.DataFrame(saved)}, save, close)


def plot_histograms(results, *, bins: int = 20, save: Optional[bool] = None, close: Optional[bool] = None):
    """Study 2: histograms of predicted P(correct) per knowledge regime, for answers that were
    right and wrong, drawn as the Study 2 overview draws them
    (`knowledge_regimes.descriptives.plot_probability_histograms`): a row per model, a column
    per regime, the dashed line the regime's mean. Drawn from the counts that are saved."""
    preds = results["predictions"]
    edges = np.linspace(0, 1, bins + 1)
    fig, axes = plt.subplots(2, len(REGIME_ORDER), figsize=(13, 6.6), sharex=True, sharey="row",
                             layout="constrained")
    saved = []
    for i, model in enumerate(LABELS):
        for j, regime in enumerate(REGIME_ORDER):
            ax = axes[i, j]
            sub = _rows(preds[preds["model"] == model], "knowqa", regime)
            for outcome, colour, tag in ((0, "#c44e52", "was wrong"), (1, "#4c72b0", "was correct")):
                probs = sub.loc[sub["y_true"] == outcome, "y_prob"].to_numpy(float)
                counts, _ = np.histogram(probs, bins=edges)
                ax.hist(edges[:-1], bins=edges, weights=counts, alpha=0.65, color=colour,
                        label=f"{tag} (n={len(probs)})")
                saved += [{"model": model, "regime": regime, "outcome": tag, "bin_low": lo, "bin_high": hi,
                           "n": int(n), MODEL_FAMILY_COL: model_folder(sub)}
                          for lo, hi, n in zip(edges[:-1], edges[1:], counts)]
            mean = sub["y_prob"].astype(float).mean()
            ax.axvline(mean, ls="--", c="k", lw=1.2)
            ax.set_title((f"{regime}\n" if i == 0 else "") + f"mean P = {mean:.3f}", fontsize=11)
            ax.legend(fontsize=8, frameon=False)
            if j == 0:
                ax.set_ylabel(f"{LABELS[model]}\ntrials", fontsize=11)
            if i == 1:
                ax.set_xlabel("predicted P(correct)")
    return _save(fig, "showcase_histograms", {"bins": pd.DataFrame(saved)}, save, close)


def plot_discrimination(results, *, save: Optional[bool] = None, close: Optional[bool] = None):
    """How much higher P(correct) is on right answers than on wrong ones, for both models
    on each test, with the 95% interval from resampling participants; beside each, the AUC
    and p. Zero is no discrimination at all."""
    disc = results["discrimination"]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    y = np.arange(len(TESTS))
    shown = []
    for k, model in enumerate(LABELS):
        rows = [_rows(disc[disc["model"] == model], ds, t).iloc[0] for ds, t in TESTS]
        pos = y + (k - 0.5) * 0.34
        d = np.array([r["diff"] for r in rows])
        lo = np.array([r["diff_ci_low"] for r in rows])
        hi = np.array([r["diff_ci_high"] for r in rows])
        ax.errorbar(d, pos, xerr=[d - lo, hi - d], fmt="o", color=COLOURS[model], ms=8, capsize=4,
                    label=LABELS[model])
        for p, r in zip(pos, rows):
            pval = "p < 0.001" if r["p_diff"] < 0.001 else f"p = {r['p_diff']:.3f}"
            ax.text(r["diff_ci_high"] + 0.02, p, f"AUC {r['auc']:.2f}, {pval}", va="center", fontsize=9.5)
        shown += [dict(r) for r in rows]
    ax.axvline(0, ls="--", c="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([TEST_LABELS[t].replace("\n", ": ") for _, t in TESTS], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("mean P(correct) on right answers minus on wrong answers")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return _save(fig, "showcase_discrimination", {"discrimination": pd.DataFrame(shown)}, save, close)


def plot_vs_confidence(results, knowqa: pd.DataFrame, *, save: Optional[bool] = None,
                       close: Optional[bool] = None):
    """Study 2: ROC curves of both models' P(correct), in five bins as in the paper, and of
    the participants' confidence rating, per regime. The legend gives each AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    preds = results["predictions"]
    frames = {m: knowqa_with_confidence(preds, knowqa, m) for m in LABELS}
    fig, axes = plt.subplots(1, len(REGIME_ORDER), figsize=(13, 4.6), sharey=True, layout="constrained")
    saved = []
    for ax, regime in zip(axes, REGIME_ORDER):
        curves = []
        for model, frame in frames.items():
            sub = frame[frame[REGIME_COL] == regime]
            curves.append((LABELS[model], COLOURS[model], binned_probability(sub[PROB_COL], 5),
                           sub[Con.IS_CORRECT_COLUMN]))
        curves.append(("self-report", SELF_REPORT_COLOUR, sub[CONFIDENCE_COL], sub[Con.IS_CORRECT_COLUMN]))
        for name, colour, score, yy in curves:
            fpr, tpr, _ = roc_curve(yy, score)
            area = roc_auc_score(yy, score)
            ax.plot(fpr, tpr, color=colour, lw=2.4, label=f"{name} ({area:.2f})")
            saved.append({"regime": regime, "curve": name, "auc": area, "n": len(yy),
                          MODEL_FAMILY_COL: frames[KNOWN_NAME][MODEL_FAMILY_COL].iloc[0]})
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
        ax.set_title(f"{regime}\n{int(sub[Con.IS_CORRECT_COLUMN].sum())} right, "
                     f"{int((sub[Con.IS_CORRECT_COLUMN] == 0).sum())} wrong", fontsize=11)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=9, frameon=False, loc="lower right", title="AUC", title_fontsize=9)
    axes[0].set_ylabel("true positive rate")
    return _save(fig, "showcase_vs_confidence", {"auc": pd.DataFrame(saved)}, save, close)


def main(save: bool = True, verbose: bool = True):
    """Everything above, saved. Returns the results."""
    from src.explorations.answer_hidden.run import load_tables

    tables = load_tables(verbose=verbose)
    results = showcase_results(tables, verbose=verbose)
    plot_accuracy(results, save=save, close=True)
    plot_confusion(results, save=save, close=True)
    plot_probabilities(results, save=save, close=True)
    plot_histograms(results, save=save, close=True)
    plot_discrimination(results, save=save, close=True)
    plot_vs_confidence(results, tables["knowqa"], save=save, close=True)
    return results


if __name__ == "__main__":
    main()
