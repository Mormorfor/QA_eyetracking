"""The paper's model's predicted probability on correct answers, by knowledge regime.

**The model** is the paper's Study 2 model: the twelve features of
`descriptives.SHOWCASE_FEATURE_COLS`, fitted on every Study 1 trial (`fit_l1_model`) and
applied to every Study 2 trial (`attach_predictions`). It is told which answer is correct.

**The comparison** is between regimes, on correct answers only: a correct answer under no
knowledge is a guess, under partial or full knowledge it mostly is not. Each regime's mean
is over its correct trials.

**The tests.** Every participant has trials in all three regimes, so the regimes are
compared within the same people:

* resampling participants (2,000 draws, the same draws for every regime), for each
  difference a 95% interval and a two-sided p (twice the smaller share of resampled
  differences on either side of zero; it cannot go below 1/2,000);
* each participant's own mean in each regime, how many people have the higher mean in the
  regime with more knowledge, and a Wilcoxon signed-rank test on those 13 paired
  differences.

Study 2 has 13 participants; some have as few as 12 correct answers under no knowledge.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    PROB_COL,
    REGIME_COL,
    REGIME_ORDER,
    attach_predictions,
    fit_l1_model,
    load_showcase_frame,
)

ANALYSIS = "explorations/regime_probability"

# (name, the regime with more knowledge, the regime with less).
CONTRASTS = [
    ("partial - no", "partial knowledge", "no knowledge"),
    ("full - no", "full knowledge", "no knowledge"),
    ("full - partial", "full knowledge", "partial knowledge"),
]

# The paper's regime colours (`descriptives`).
REGIME_COLOURS = {
    "no knowledge": "#c44e52",
    "partial knowledge": "#dd8452",
    "full knowledge": "#4c72b0",
}


def paper_predictions(verbose: bool = True) -> pd.DataFrame:
    """Every Study 2 trial with the paper's model's predicted probability (`PROB_COL`)."""
    knowqa = load_showcase_frame(verbose=verbose)
    model = fit_l1_model(verbose=verbose)
    return attach_predictions(knowqa, model, verbose=verbose)


def regime_means(preds: pd.DataFrame) -> pd.DataFrame:
    """Per regime and outcome: the number of trials and the mean and median probability."""
    out = (
        preds.groupby([REGIME_COL, Con.IS_CORRECT_COLUMN])[PROB_COL]
        .agg(n="count", mean_prob="mean", median_prob="median")
        .reindex(pd.MultiIndex.from_product([REGIME_ORDER, [1, 0]], names=[REGIME_COL, Con.IS_CORRECT_COLUMN]))
        .reset_index()
    )
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def person_means(preds: pd.DataFrame, outcome: int = 1) -> pd.DataFrame:
    """Each participant's mean probability and number of trials in each regime, on the
    trials with `outcome` (1 correct, 0 wrong). One row per participant and regime."""
    sub = preds[preds[Con.IS_CORRECT_COLUMN] == outcome]
    out = (
        sub.groupby([Con.PARTICIPANT_ID, REGIME_COL])[PROB_COL]
        .agg(n="count", mean_prob="mean")
        .reset_index()
    )
    out[Con.IS_CORRECT_COLUMN] = outcome
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def _bootstrap_means(sub: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, np.ndarray]:
    """Each regime's mean probability over `sub`'s trials, on `n_boot` resamples of the
    participants. The same resamples for every regime, so differences are paired. A
    resample with no trials in a regime has no mean (NaN)."""
    people = np.sort(sub[Con.PARTICIPANT_ID].unique())
    person = pd.Index(people).get_indexer(sub[Con.PARTICIPANT_ID])
    draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
    # How many times each participant is drawn, per resample.
    weight = np.stack([np.bincount(d, minlength=len(people)) for d in draws])
    prob = sub[PROB_COL].to_numpy(float)
    out = {}
    for r in REGIME_ORDER:
        m = (sub[REGIME_COL] == r).to_numpy()
        sums = np.bincount(person[m], weights=prob[m], minlength=len(people))
        counts = np.bincount(person[m], minlength=len(people)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = (weight @ sums) / (weight @ counts)
    return out


def _p_two_sided(boot: np.ndarray) -> float:
    return float(np.clip(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / len(boot), 1))


def contrast_tests(
    preds: pd.DataFrame,
    *,
    outcome: int = 1,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per contrast of `CONTRASTS`, on the trials with `outcome`: the difference in
    mean probability with its resampled interval and p, and the per-participant test (see
    the module docstring). Also each regime's mean with its interval, as rows named after
    the regime."""
    sub = preds[preds[Con.IS_CORRECT_COLUMN] == outcome]
    boot = _bootstrap_means(sub, n_boot, seed)
    point = sub.groupby(REGIME_COL)[PROB_COL].mean()
    per = person_means(preds, outcome).pivot(index=Con.PARTICIPANT_ID, columns=REGIME_COL, values="mean_prob")
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    rows = []
    for r in REGIME_ORDER:
        b = boot[r][~np.isnan(boot[r])]
        rows.append({
            "test": r, "estimate": point[r],
            "ci_low": np.percentile(b, lo_q), "ci_high": np.percentile(b, hi_q),
            "n_trials": int((sub[REGIME_COL] == r).sum()), "n_boot_ok": len(b),
        })
    for name, more, less in CONTRASTS:
        b = boot[more] - boot[less]
        b = b[~np.isnan(b)]
        # Only participants with trials of this outcome in both regimes are paired.
        d = (per[more] - per[less]).dropna()
        rows.append({
            "test": name, "estimate": point[more] - point[less],
            "ci_low": np.percentile(b, lo_q), "ci_high": np.percentile(b, hi_q),
            "p": _p_two_sided(b), "n_boot_ok": len(b),
            "n_people": len(d), "people_higher": int((d > 0).sum()),
            "median_person_diff": d.median(), "wilcoxon_p": wilcoxon(d).pvalue,
        })
    out = pd.DataFrame(rows)
    out[Con.IS_CORRECT_COLUMN] = outcome
    out[["n_boot", "ci", "seed"]] = n_boot, ci, seed
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def _p_label(p: float, n_boot: int) -> str:
    return f"p < {1 / n_boot:.4f}".rstrip("0") if p <= 1 / n_boot else f"p = {p:.3f}"


def plot_correct_by_regime(
    tests: pd.DataFrame,
    people: pd.DataFrame,
    means: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Each participant's mean probability on correct answers in each regime (grey lines),
    and the regime means with 95% intervals; brackets give each contrast and its p. Saved
    with the tests, the per-participant means and the regime means. Returns the figure."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
    x = {r: i for i, r in enumerate(REGIME_ORDER)}
    for _, person in people.groupby(Con.PARTICIPANT_ID):
        person = person.set_index(REGIME_COL).reindex(REGIME_ORDER)
        ax.plot(range(len(REGIME_ORDER)), person["mean_prob"], color="#BBBBBB", lw=1, marker="o", ms=3, zorder=1)
    by_test = tests.set_index("test")
    for r in REGIME_ORDER:
        row = by_test.loc[r]
        ax.errorbar(
            x[r], row["estimate"],
            yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o", ms=9, color=REGIME_COLOURS[r], elinewidth=2, capsize=0, zorder=3,
        )
        ax.annotate(f"{row['estimate']:.2f}", (x[r], row["estimate"]), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9)

    n_boot = int(tests["n_boot"].iloc[0])
    top = people["mean_prob"].max()
    for level, (name, more, less) in enumerate(sorted(CONTRASTS, key=lambda c: x[c[1]] - x[c[2]])):
        row = by_test.loc[name]
        y = top + 0.05 + 0.07 * level
        ax.plot([x[less], x[less], x[more], x[more]], [y - 0.015, y, y, y - 0.015], color="#555555", lw=1)
        ax.text((x[less] + x[more]) / 2, y + 0.008,
                f"{row['estimate']:+.3f}, {_p_label(row['p'], n_boot)}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(REGIME_ORDER)))
    ax.set_xticklabels([r.replace(" knowledge", "\nknowledge") for r in REGIME_ORDER])
    ax.set_xlim(-0.5, len(REGIME_ORDER) - 0.5)
    ax.set_ylabel("Mean predicted P(correct) on correct answers")
    ax.grid(axis="y", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Study 2, the paper's model: correct answers by regime\n"
                 "grey: each participant; colour: regime mean, 95% interval", fontsize=10)
    save_output(
        fig, analysis=ANALYSIS, plot="prob_on_correct_by_regime",
        tables={"tests": tests, "per_person": people, "means": means},
        save=save, close=close, subdir=model_folder(tests),
    )
    return fig


def main(save: bool = True, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """The tests on correct answers, saved as a figure with its tables. Returns the
    predictions, the regime means, the per-participant means and the tests."""
    preds = paper_predictions(verbose=verbose)
    means = regime_means(preds)
    people = person_means(preds, outcome=1)
    tests = contrast_tests(preds, outcome=1)
    plot_correct_by_regime(tests, people, means, save=save, close=True)
    if verbose:
        print(tests.drop(columns=["n_boot", "ci", "seed", MODEL_FAMILY_COL]).round(4).to_string(index=False))
    return {"predictions": preds, "means": means, "per_person": people, "tests": tests}


if __name__ == "__main__":
    main()
