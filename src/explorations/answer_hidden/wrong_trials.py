"""On wrong answers: did people look at the correct answer more than at the wrong answers they
did not pick?

On a trial answered wrongly, the participant chose one of the three wrong answers, and passed
over the correct one and the other two wrong ones. How much those three were looked at does
not come from the choice, since none of them was chosen; and since they are compared within
the trial, it does not come from how long the trial took. If the correct answer gets more
looking than the passed-over wrong answers, the eyes register something about it that the
choice did not.

**The comparison**, per wrong trial and per measure: the correct answer's value minus the
mean of the two passed-over wrong answers (`other_wrong`). The measures are the paper's ten
per-answer measures (`columns.AREA_METRIC_COLUMNS_MODELING`). Pupil size and first-fixation
duration exist only for answers that were fixated; a trial where the correct answer, or both
passed-over wrong answers, has no value is left out of that measure's comparison, and the
count of trials kept is reported.

Two things other than knowledge can make the correct answer stand out, and both are checked:

* **Length.** A longer answer gets more looking in total. Dwell time, fixation count and skip
  rate are per word; dwell share and visits are not. The answers' lengths in words are
  compared the same way (`n_words`), so a difference in length is visible next to the rest.
* **Answer type.** OneStop builds each wrong answer differently (`glossary.md`): B misreads
  the part of the text that holds the answer, C uses another part, D nothing in the text.
  Which two are passed over depends on which one was chosen, and that mix changes with
  knowledge. So the correct answer is also compared with each wrong type on its own, on the
  trials where that type was not chosen (`minus_answer_B`, ...), which holds the type fixed.

**Study 2's no-knowledge regime is the control.** There the text was not read, so extra
looking at the correct answer comes from the answers themselves -- some correct answers
look plausible on their own. What matters is whether the difference is larger with
knowledge: `partial - no` and `full - no`.

**The tests** resample participants (2,000 draws), with the same draws for every measure
and regime of a study, and give each mean difference a 95% interval and a two-sided p (twice
the smaller share of resampled means on either side of zero; it cannot go below 1/2,000).
Every Study 2 participant has trials in all three regimes, so a regime contrast is paired
within the same resampled people. Study 2 has 13 participants and only 16 wrong answers
under full knowledge.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import dataset
from src.lib.plotting.output import save_output
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import REGIME_COL, REGIME_ORDER
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.run import ANALYSIS, load_tables
from src.explorations.answer_hidden.table import SELECTED_COL, SEP, selected_letters

MEASURES: List[str] = list(Con.AREA_METRIC_COLUMNS_MODELING)
LENGTH = "n_words"
CORRECT = Con.CORRECT_ANSWER_AREA
WRONG: List[str] = list(Con.WRONG_ANSWER_AREAS)

# The comparisons: with both passed-over wrong answers, and with each wrong type alone.
COMPARISONS: List[str] = ["other_wrong", *(f"minus_{w}" for w in WRONG)]

# The regime contrasts of Study 2: (name, regime with knowledge, the control).
CONTRASTS = [
    ("partial - no", "partial knowledge", "no knowledge"),
    ("full - no", "full knowledge", "no knowledge"),
]


def answer_lengths(dataset_key: str) -> pd.DataFrame:
    """One row per trial: each answer's length in words (`n_words__answer_A` ..), counted
    from the word table, where every word on screen has a row."""
    words = pd.read_csv(
        dataset(dataset_key).interest_areas,
        usecols=[*TRIAL_ID_COLS, Con.AREA_LABEL_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )
    counts = (
        words[words[Con.AREA_LABEL_COLUMN].isin(Con.ANSWER_AREAS)]
        .groupby([*TRIAL_ID_COLS, Con.AREA_LABEL_COLUMN]).size()
        .unstack(Con.AREA_LABEL_COLUMN)[Con.ANSWER_AREAS]
    )
    assert counts.notna().all().all(), f"{dataset_key}: a trial has an answer with no words"
    counts.columns = [f"{LENGTH}{SEP}{a}" for a in counts.columns]
    return counts.reset_index()


def wrong_trials(known: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """One row per wrongly answered trial of `known`, the paper's trial table: the ids, the
    regime on Study 2, the wrong answer chosen, and for each measure and the length the
    comparisons of `COMPARISONS` (`<measure>__other_wrong`, `<measure>__minus_answer_B`, ..),
    each the correct answer's value minus the other side's. A comparison with a wrong type
    that was chosen is left empty, as is one where either side has no value."""
    df = known.merge(selected_letters(dataset_key), on=list(TRIAL_ID_COLS), how="left", validate="one_to_one")
    assert df[SELECTED_COL].notna().all(), f"{dataset_key}: trials with no selected answer"
    lengths = answer_lengths(dataset_key)
    assert_full_coverage(df, lengths, TRIAL_ID_COLS, "answer lengths")
    df = df.merge(lengths, on=list(TRIAL_ID_COLS), how="left", validate="one_to_one")

    correct_trial = df[Con.IS_CORRECT_COLUMN] == 1
    assert ((df[SELECTED_COL] == CORRECT) == correct_trial).all(), (
        f"{dataset_key}: the selected answer and is_correct disagree"
    )
    wrong = df[~correct_trial].reset_index(drop=True)

    keep = [*TRIAL_ID_COLS, *([REGIME_COL] if REGIME_COL in wrong.columns else []), SELECTED_COL]
    out = {c: wrong[c] for c in keep}
    rows = np.arange(len(wrong))
    chosen = wrong[SELECTED_COL].map({w: j for j, w in enumerate(WRONG)}).to_numpy(int)
    for m in [*MEASURES, LENGTH]:
        correct = wrong[f"{m}{SEP}{CORRECT}"].to_numpy(float)
        others = wrong[[f"{m}{SEP}{w}" for w in WRONG]].to_numpy(float)
        passed = others.copy()
        passed[rows, chosen] = np.nan
        # The mean of the two passed-over wrong answers, over those with a value.
        has = ~np.isnan(passed)
        with np.errstate(invalid="ignore", divide="ignore"):
            other_mean = np.where(has, passed, 0.0).sum(axis=1) / has.sum(axis=1)
        out[f"{m}{SEP}other_wrong"] = correct - other_mean
        for j, w in enumerate(WRONG):
            d = correct - others[:, j]
            d[chosen == j] = np.nan
            out[f"{m}{SEP}minus_{w}"] = d
    return pd.DataFrame(out)


def _tests(trials: pd.DataFrame, dataset_key: str) -> Dict[str, pd.Series]:
    """The trials of each test, as a mask: Study 1 as a whole, Study 2 by regime."""
    if dataset_key == "l1":
        return {"all": pd.Series(True, index=trials.index)}
    return {r: trials[REGIME_COL] == r for r in REGIME_ORDER}


def difference_table(
    trials: Dict[str, pd.DataFrame],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """One row per dataset, test, measure and comparison: the mean difference over the
    trials with a value, its interval and p (see the module docstring), and the numbers of
    trials and participants behind it. Study 2 has a row per regime and per contrast of
    `CONTRASTS`. `trials` maps "l1" and "knowqa" to `wrong_trials`. Saved and read back while
    the trials and the settings are the same."""

    def compute():
        lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        out = []
        for ds, t in trials.items():
            people = np.sort(t[Con.PARTICIPANT_ID].unique())
            person_idx = pd.Index(people).get_indexer(t[Con.PARTICIPANT_ID])
            draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
            # How many times each participant is drawn, per resample.
            weight = np.stack([np.bincount(d, minlength=len(people)) for d in draws])
            tests = _tests(t, ds)
            for m in [*MEASURES, LENGTH]:
                for comp in COMPARISONS:
                    values = t[f"{m}{SEP}{comp}"].to_numpy(float)
                    boot_means = {}
                    for test, mask in tests.items():
                        ok = mask.to_numpy() & ~np.isnan(values)
                        sums = np.bincount(person_idx[ok], weights=values[ok], minlength=len(people))
                        counts = np.bincount(person_idx[ok], minlength=len(people)).astype(float)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            boot = (weight @ sums) / (weight @ counts)
                        boot_means[test] = boot
                        out.append(_row(ds, test, m, comp, values[ok].mean() if ok.any() else np.nan,
                                        boot, int(ok.sum()), int((counts > 0).sum()), lo_q, hi_q))
                    if ds == "knowqa":
                        for name, with_k, control in CONTRASTS:
                            point = _out_mean(out, ds, with_k, m, comp) - _out_mean(out, ds, control, m, comp)
                            boot = boot_means[with_k] - boot_means[control]
                            out.append(_row(ds, name, m, comp, point, boot, np.nan, np.nan, lo_q, hi_q))
            if verbose:
                print(f"  {ds}: done")
        return pd.DataFrame(out).assign(n_boot=n_boot, ci=ci, seed=seed)

    inputs = {"trials": {k: table_digest(v) for k, v in trials.items()}, "n_boot": n_boot, "ci": ci, "seed": seed}
    return cached("wrong_trials_differences", inputs, compute, rerun=rerun, verbose=verbose)


def _out_mean(rows: List[dict], ds: str, test: str, measure: str, comp: str) -> float:
    """The mean difference already computed for this dataset, test, measure and comparison."""
    (r,) = [r for r in rows if (r["dataset"], r["test"], r["measure"], r["comparison"]) == (ds, test, measure, comp)]
    return r["mean_diff"]


def _row(ds, test, measure, comp, point, boot, n, n_people, lo_q, hi_q) -> dict:
    # A resample with no trials for a test has no mean; it is left out and counted.
    boot = boot[~np.isnan(boot)]
    return {
        "dataset": ds, "test": test, "measure": measure, "comparison": comp,
        "mean_diff": point,
        "ci_low": np.percentile(boot, lo_q) if len(boot) else np.nan,
        "ci_high": np.percentile(boot, hi_q) if len(boot) else np.nan,
        "p": float(np.clip(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / len(boot), 1)) if len(boot) else np.nan,
        "n_trials": n, "n_participants": n_people, "n_boot_ok": len(boot),
    }


def chosen_counts(trials: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which wrong answer was chosen, as shares, on Study 1 and in each Study 2 regime, with
    the number of wrong trials."""
    parts = {}
    for ds, t in trials.items():
        for test, mask in _tests(t, ds).items():
            sel = t.loc[mask, SELECTED_COL]
            parts[(ds, test)] = sel.value_counts(normalize=True).reindex(WRONG, fill_value=0).to_dict() | {"n": len(sel)}
    return pd.DataFrame(parts).T


# The measures in the figure: the attention measures and the length. The pupil and
# first-fixation measures are in the saved table.
FIGURE_MEASURES = [
    (Con.MEAN_DWELL_TIME, "Dwell time per word (ms)"),
    (Con.MEAN_FIXATIONS_COUNT, "Fixations per word"),
    (Con.SKIP_RATE, "Share of words skipped"),
    (Con.AREA_DWELL_PROPORTION, "Share of looking time"),
    (Con.NUM_LABEL_VISITS, "Visits"),
    (LENGTH, "Length (words)"),
]

COMPARISON_STYLE = {
    "other_wrong": ("#333333", "o", "vs the two passed-over wrong answers"),
    f"minus_{WRONG[0]}": ("#4C72B0", "s", "vs B (misreads the answer's part of the text)"),
    f"minus_{WRONG[1]}": ("#DD8452", "D", "vs C (another part of the text)"),
    f"minus_{WRONG[2]}": ("#55A868", "^", "vs D (nothing in the text)"),
}

TEST_LABELS = {
    ("l1", "all"): "Study 1",
    ("knowqa", "no knowledge"): "Study 2: no knowledge",
    ("knowqa", "partial knowledge"): "Study 2: partial",
    ("knowqa", "full knowledge"): "Study 2: full",
    ("knowqa", "partial - no"): "partial minus no",
    ("knowqa", "full - no"): "full minus no",
}


def plot_differences(
    table: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """One panel per measure of `FIGURE_MEASURES`: the correct answer minus the passed-over
    wrong answers on wrong trials, with 95% intervals, a row per test and a mark per
    comparison. Right of the dashed line, the correct answer has more of the measure. The
    regime contrasts are below the line. Saved with the full table. Returns the figure."""
    tests = list(TEST_LABELS)
    offsets = dict(zip(COMPARISON_STYLE, np.linspace(-0.27, 0.27, len(COMPARISON_STYLE))))
    ncol = 3
    nrow = int(np.ceil(len(FIGURE_MEASURES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.2 * nrow + 0.8), sharey=True, layout="constrained")
    for ax, (measure, title) in zip(axes.flat, FIGURE_MEASURES):
        sub = table[table["measure"] == measure]
        for comp, (colour, marker, _) in COMPARISON_STYLE.items():
            for _, r in sub[sub["comparison"] == comp].iterrows():
                key = (r["dataset"], r["test"])
                if key not in TEST_LABELS:
                    continue
                y = tests.index(key) + offsets[comp]
                ax.errorbar(
                    r["mean_diff"], y,
                    xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                    fmt=marker, color=colour, ms=5, elinewidth=1.3, capsize=0,
                )
        ax.axvline(0, color="#555555", lw=1, ls="--")
        ax.axhline(0.5, color="#BBBBBB", lw=1)
        ax.axhline(3.5, color="#BBBBBB", lw=1)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.grid(axis="x", color="#E5E5E5", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(FIGURE_MEASURES):]:
        ax.axis("off")
    axes[0, 0].set_yticks(range(len(tests)))
    axes[0, 0].set_yticklabels([TEST_LABELS[t] for t in tests])
    axes[0, 0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=6, label=lab) for c, mk, lab in COMPARISON_STYLE.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Wrong answers: the correct answer minus the wrong answers not chosen, within the trial "
        "(right of the dashed line: the correct answer got more)",
        fontsize=11,
    )
    save_output(fig, analysis=ANALYSIS, plot="wrong_trials_correct_vs_passed_over",
                tables={"differences": table}, save=save, close=close)
    return fig


def main(save: bool = True, verbose: bool = True, rerun: bool = False) -> Dict[str, pd.DataFrame]:
    """The comparison on both studies, saved as a figure with its table. Returns the wrong
    trials, the table and which wrong answers were chosen."""
    tables = load_tables(verbose=verbose)
    trials = {ds: wrong_trials(tables[ds], ds) for ds in ("l1", "knowqa")}
    table = difference_table(trials, verbose=verbose, rerun=rerun)
    plot_differences(table, save=save, close=True)
    return {"trials": trials, "differences": table, "chosen": chosen_counts(trials)}


if __name__ == "__main__":
    main()
