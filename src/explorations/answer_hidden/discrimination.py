"""Does the predicted probability differ between answers that were right and wrong?

For every model version and test, from the predictions of `run.collect_predictions`:

    mean_p_correct, mean_p_wrong  the average P(correct) on answers that were right, and
                                  on those that were wrong
    diff                          the first minus the second
    auc                           the chance that a random right answer gets a higher
                                  P(correct) than a random wrong one; 0.5 is no
                                  discrimination at all

Each comes with an interval from resampling participants, as the paper's Study 2 test does
(`knowledge_regimes.descriptives.discrimination_bootstrap`). A test that treated trials as
independent would be too sure of itself: a person's answers are not independent, and wrong
answers cluster within people. Each of `n_boot` resamples draws as many participants as
the test has, with replacement, and takes all of their predictions. The intervals are the
middle `ci` of the resampled values; `p_diff` is two-sided, 2 x the smaller share of
resampled differences at or below 0 and at or above 0, floored at 1 / `n_boot_ok`. Within
a test, every model is resampled with the same draws, so the versions can be compared
with each other.

On Study 1 the predictions are those of the ten cross-validation folds, pooled; a trial
held out in more than one fold counts each time, and stays with its participant in the
resampling. As in the paper's test, the items are not resampled, though they recur across
participants; and Study 2 has only 13 participants, few units for a bootstrap, so its
intervals are likely too narrow.

**Against the confidence rating** (Study 2 only): `confidence_comparison` runs the paper's
own test (`discrimination_bootstrap`) on every version, and `plot_confidence_roc` draws the
paper's ROC figure with a curve per version. As there, P(correct) is cut into five
equal-width bins so that it has as many levels as the five-point rating; the answer-hidden
versions' P covers a narrower range, so the bins cost them 0.01-0.03 of AUC, about what
they cost the paper's model.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.lib.plotting.output import save_output
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.analyses.correctness_prediction.comparison import POOLED
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    CONFIDENCE_COL,
    PROB_COL,
    REGIME_COL,
    REGIME_ORDER,
    discrimination_bootstrap,
)
from src.analyses.correctness_prediction.plots.model_comparison import TEST_TITLES
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.run import ANALYSIS, CONFUSION_TESTS, VERSION_ORDER

# One colour per version, the same in every panel.
VERSION_COLOURS = {
    "known": "#333333",
    "position": "#4C72B0",
    "selected": "#DD8452",
    "spread": "#55A868",
}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """The ROC AUC of `p` for the 0/1 outcome `y`, from ranks (ties share their average
    rank): the same number as `sklearn.metrics.roc_auc_score`, faster for the thousands of
    resamples here."""
    ranks = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _test_rows(preds: pd.DataFrame, dataset: str, test: str) -> pd.DataFrame:
    rows = preds[preds["dataset"] == dataset]
    return rows if test == POOLED else rows[rows["test"] == test]


def discrimination_table(
    preds: pd.DataFrame,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """One row per dataset, test and model version: the mean P(correct) on right and on
    wrong answers, their difference, the AUC, and their intervals from resampling
    participants (see the module docstring). The tests are those of the confusion grids
    (`run.CONFUSION_TESTS`). Saved and read back while the predictions and the settings are
    the same (`cache.py`)."""

    def compute():
        lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        out = []
        for dataset, tests in CONFUSION_TESTS.items():
            for test in tests:
                rows = _test_rows(preds, dataset, test)
                people = np.sort(rows[Con.PARTICIPANT_ID].astype(str).unique())
                # The same draws of participants for every model of this test.
                draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
                for model, g in rows.groupby("model", sort=False, observed=True):
                    y = g["y_true"].to_numpy(int)
                    p = g["y_prob"].to_numpy(float)
                    by_person = g.groupby(g[Con.PARTICIPANT_ID].astype(str)).indices
                    assert set(by_person) == set(people), f"{model}: participants differ within {test}"
                    blocks = [by_person[person] for person in people]

                    boots = []
                    for draw in draws:
                        idx = np.concatenate([blocks[i] for i in draw])
                        yy, pp = y[idx], p[idx]
                        if yy.min() == yy.max():
                            continue
                        boots.append((pp[yy == 1].mean() - pp[yy == 0].mean(), auc(yy, pp)))
                    boots = np.array(boots)
                    diffs = boots[:, 0]

                    out.append({
                        "dataset": dataset, "test": test, "model": model,
                        "paper_model": g["paper_model"].iloc[0], "answers": g["answers"].iloc[0],
                        "n": len(g), "n_correct": int(y.sum()), "n_wrong": int((y == 0).sum()),
                        "n_participants": len(people),
                        "mean_p_correct": p[y == 1].mean(),
                        "mean_p_wrong": p[y == 0].mean(),
                        "diff": p[y == 1].mean() - p[y == 0].mean(),
                        "diff_ci_low": np.percentile(diffs, lo_q),
                        "diff_ci_high": np.percentile(diffs, hi_q),
                        "p_diff": float(np.clip(
                            2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / len(diffs), 1
                        )),
                        "auc": auc(y, p),
                        "auc_ci_low": np.percentile(boots[:, 1], lo_q),
                        "auc_ci_high": np.percentile(boots[:, 1], hi_q),
                        "n_boot_ok": len(diffs),
                        MODEL_FAMILY_COL: model_folder(g),
                    })
                if verbose:
                    print(f"  {dataset} / {test}: done")
        return pd.DataFrame(out).assign(n_boot=n_boot, ci=ci, seed=seed)

    inputs = {"predictions": table_digest(preds), "n_boot": n_boot, "ci": ci, "seed": seed}
    return cached("discrimination", inputs, compute, rerun=rerun, verbose=verbose)


def _p_label(p: float, n_boot_ok: int) -> str:
    """The p-value as text; at the floor of the resampling, as "< 1/n"."""
    floor = 1 / n_boot_ok
    return f"p < {floor:.4f}".rstrip("0") if p <= floor else f"p = {p:.3f}"


def plot_discrimination(
    table: pd.DataFrame,
    *,
    dataset: str,
    models: Sequence[str],
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """The difference in mean P(correct) between right and wrong answers, for every version
    on every test of `dataset`, with its interval from resampling participants: a panel
    per test, a row per version (`models`, top to bottom), coloured by version. Zero is no
    difference at all. Beside each bar: the AUC and the p-value. Saved with the rows it
    shows. Returns the figure and those rows."""
    tests = CONFUSION_TESTS[dataset]
    rows = table[(table["dataset"] == dataset) & table["model"].isin(models)]
    y = np.arange(len(models))

    fig, axes = plt.subplots(
        1, len(tests), sharey=True, sharex=True, layout="constrained",
        figsize=(3.6 * len(tests) + 2.6, 0.42 * len(models) + 1.8),
    )
    axes = np.atleast_1d(axes)
    hi = rows["diff_ci_high"].max()
    lo = min(0.0, rows["diff_ci_low"].min())
    for ax, test in zip(axes, tests):
        panel = rows[rows["test"] == test].set_index("model").reindex(models)
        colours = [VERSION_COLOURS[a] for a in panel["answers"]]
        ax.barh(y, panel["diff"], color=colours, height=0.62, zorder=2)
        ax.errorbar(
            panel["diff"], y,
            xerr=np.vstack([panel["diff"] - panel["diff_ci_low"], panel["diff_ci_high"] - panel["diff"]]),
            fmt="none", ecolor="#333333", elinewidth=1, capsize=2.5, zorder=3,
        )
        for yi, (_, r) in zip(y, panel.iterrows()):
            ax.text(max(r["diff_ci_high"], 0) + 0.04 * hi, yi,
                    f"AUC {r['auc']:.2f} · {_p_label(r['p_diff'], r['n_boot_ok'])}",
                    va="center", fontsize=7)
        ax.axvline(0, color="#888888", linestyle="--", linewidth=1, zorder=1)
        n_people = int(panel["n_participants"].iloc[0])
        ax.set_title(f"{TEST_TITLES.get(test, test)}\n{n_people} participants", fontsize=9.5)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        # Room on the right for the labels.
        ax.set_xlim(lo - 0.03 * hi, hi * 1.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models, fontsize=8)
    axes[0].invert_yaxis()

    where = ("Study 1 (L1): held-out trials of the ten folds" if dataset == "l1"
             else "Study 2 (KnowQA): trained on every Study 1 trial")
    fig.suptitle(f"{where}\nmean P(correct) on right answers minus on wrong answers "
                 "(95% interval from resampling participants)", fontsize=10.5)
    fig.supxlabel("difference in mean predicted P(correct)  (dashed line: none)", fontsize=9.5)
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="discrimination",
        tables={"discrimination": rows},
        save=save,
        close=close,
        subdir=model_folder(rows),
        dataset=dataset,
    )
    return fig, rows


# ---------------------------------------------------------------------------
# Against the confidence rating (Study 2)
# ---------------------------------------------------------------------------

SELF_REPORT_COLOUR = "#8172B3"


def knowqa_with_confidence(preds: pd.DataFrame, knowqa: pd.DataFrame, model: str) -> pd.DataFrame:
    """One version's Study 2 predictions, with each trial's confidence rating from
    `knowqa` (the paper's Study 2 table, `run.load_tables()["knowqa"]`), in the columns the
    paper's Study 2 functions take: participant, trial, `regime`, `is_correct`,
    `pred_prob`, `confidence_rating` and `model_family`. Stops if a trial has no rating."""
    rows = preds[(preds["dataset"] == "knowqa") & (preds["model"] == model)]
    out = pd.DataFrame({
        Con.PARTICIPANT_ID: rows[Con.PARTICIPANT_ID].astype(str).to_numpy(),
        Con.TRIAL_ID: rows[Con.TRIAL_ID].astype(str).to_numpy(),
        REGIME_COL: rows["test"].astype(str).to_numpy(),
        Con.IS_CORRECT_COLUMN: rows["y_true"].to_numpy(int),
        PROB_COL: rows["y_prob"].to_numpy(float),
        MODEL_FAMILY_COL: rows[MODEL_FAMILY_COL].astype(str).to_numpy(),
    })
    ratings = knowqa[[*TRIAL_ID_COLS, CONFIDENCE_COL]].astype({Con.TRIAL_ID: str})
    assert_full_coverage(out, ratings, TRIAL_ID_COLS, "confidence ratings")
    out = out.merge(ratings, on=list(TRIAL_ID_COLS), how="left")
    missing = int(out[CONFIDENCE_COL].isna().sum())
    assert not missing, f"{missing} Study 2 trial(s) have no confidence rating"
    return out


def binned_probability(prob: pd.Series, bins: int) -> pd.Series:
    """P(correct) cut into `bins` equal-width bins on [0, 1], numbered from 1 like the
    rating, by the paper's rule (`descriptives.plot_discrimination_roc`): right-closed, so
    0.2 falls in bin 1."""
    return pd.cut(prob, np.linspace(0, 1, bins + 1), labels=False, include_lowest=True) + 1


def confidence_comparison(
    preds: pd.DataFrame,
    knowqa: pd.DataFrame,
    models: Sequence[str],
    *,
    model_bins: Optional[int] = 5,
    n_boot: int = 2000,
    seed: int = 0,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """The paper's test of the model's AUC against the confidence rating's
    (`descriptives.discrimination_bootstrap`), run on each of `models`: one row per model
    and regime, with both AUCs, their difference (model minus rating), the
    participant-bootstrap intervals and p. `model_bins` as there: 5 bins by default,
    `None` for the raw probability. Every model is resampled with the same draws. Saved
    and read back as the other results are (`cache.py`)."""

    def compute():
        parts = []
        for model in models:
            frame = knowqa_with_confidence(preds, knowqa, model)
            answers = preds.loc[preds["model"] == model, "answers"].iloc[0]
            parts.append(
                discrimination_bootstrap(frame, model_bins=model_bins, n_boot=n_boot, seed=seed)
                .assign(model=model, answers=answers)
            )
        return pd.concat(parts, ignore_index=True)

    inputs = {
        "predictions": table_digest(preds), "knowqa": table_digest(knowqa), "models": list(models),
        "model_bins": model_bins, "n_boot": n_boot, "seed": seed,
    }
    return cached("confidence_comparison", inputs, compute, rerun=rerun, verbose=verbose)


def plot_confidence_roc(
    preds: pd.DataFrame,
    knowqa: pd.DataFrame,
    *,
    paper_model: str,
    model_bins: Optional[int] = 5,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """The paper's ROC figure (`descriptives.plot_discrimination_roc`) with a curve for each
    version of `paper_model` beside the confidence rating's: a panel per regime, P(correct)
    in `model_bins` bins (see the module docstring). The legend gives each curve's AUC.
    Saved with the curves' points and AUCs. Returns the figure."""
    from sklearn.metrics import roc_auc_score, roc_curve

    versions = {
        answers: preds.loc[(preds["paper_model"] == paper_model) & (preds["answers"] == answers), "model"].iloc[0]
        for answers in VERSION_ORDER
    }
    frames = {answers: knowqa_with_confidence(preds, knowqa, model) for answers, model in versions.items()}

    fig, axes = plt.subplots(1, len(REGIME_ORDER), figsize=(13, 4.4), sharey=True, layout="constrained")
    points, aucs = [], []
    for ax, regime in zip(axes, REGIME_ORDER):
        curves = []
        for answers, frame in frames.items():
            sub = frame[frame[REGIME_COL] == regime]
            score = sub[PROB_COL] if model_bins is None else binned_probability(sub[PROB_COL], model_bins)
            curves.append((answers, VERSION_COLOURS[answers], score, sub[Con.IS_CORRECT_COLUMN], versions[answers]))
        # The rating is the same in every frame; taken from the first.
        sub = next(iter(frames.values()))
        sub = sub[sub[REGIME_COL] == regime]
        curves.append(("self-report", SELF_REPORT_COLOUR, sub[CONFIDENCE_COL], sub[Con.IS_CORRECT_COLUMN], "self-report"))

        for name, colour, score, y, model in curves:
            fpr, tpr, thresholds = roc_curve(y, score)
            area = roc_auc_score(y, score)
            ax.plot(fpr, tpr, color=colour, lw=2.4 if name == "self-report" else 1.8,
                    label=f"{name} (AUC {area:.3f})")
            points += [{"regime": regime, "curve": name, "model": model, "fpr": f, "tpr": t, "threshold": th}
                       for f, t, th in zip(fpr, tpr, thresholds)]
            aucs.append({"regime": regime, "curve": name, "model": model, "auc": area,
                         "n": len(y), "n_correct": int(y.sum())})
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
        ax.set_title(f"{regime}\nn={len(sub)}, {int(sub[Con.IS_CORRECT_COLUMN].sum())} correct", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    bins_note = "raw P(correct)" if model_bins is None else f"P(correct) in {model_bins} bins"
    fig.suptitle(f"{paper_model}: discriminating correct from wrong, every version vs self-report ({bins_note})",
                 fontsize=11)

    family = next(iter(frames.values()))[MODEL_FAMILY_COL].iloc[0]
    tables = {
        "auc": pd.DataFrame(aucs).assign(**{MODEL_FAMILY_COL: family}),
        "curves": pd.DataFrame(points).assign(**{MODEL_FAMILY_COL: family}),
    }
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="confidence_roc",
        tables=tables,
        save=save,
        close=close,
        subdir=model_folder(tables["auc"]),
        model=paper_model,
        bins=model_bins if model_bins is not None else "raw",
    )
    return fig
