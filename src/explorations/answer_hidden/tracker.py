"""How far the answer-hidden model gets on what only an eye tracker can measure.

Some of the features need no eye tracker at all. Without one, a study still logs the
presses and when they happened, and knows the text on screen, which gives:

* `total_answering_RT` and `total_answering_RT_normalized` -- from the answers appearing
  to the confirming press, and that per word on screen;
* `ANSWER_PRESS_NUMBER` -- how many times an answer was selected;
* `select_to_confirm_ms` -- from the last selection press to the confirming press.

Of the features that do need one, some measure **how long** the eyes stayed (`GAZE_TIME`):
the reading times (`RT_*`, `TFD_*`, `TimeSinceOffset_*`), dwell time and dwell share, the
first-fixation duration, and the time from first looking at the chosen answer to selecting
it. The rest measure **where and in what order** the eyes went: visits, back-and-forth,
fixation counts, skip rate, pupil size, the last look before a press.

`tracker_feature_sets` splits the chosen set (`deliberation.CHOSEN_GROUPS`) and a wider
eye-tracker set along those lines, so the scores show what each kind adds.

`added_value_test` asks whether a difference between two of the sets is real: both models'
held-out predictions on the same trials, participants resampled with the same draws for
both, as in `discrimination.discrimination_table`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.analyses.correctness_prediction.comparison import POOLED
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.deliberation import (
    DELIBERATION_GROUPS,
    base_cols,
    chosen_cols,
    plot_search_heatmap,
    run_deliberation_search,
)
from src.explorations.answer_hidden.discrimination import auc
from src.explorations.answer_hidden.run import ANALYSIS, CONFUSION_TESTS, load_tables, model_key, predictions_for
from src.explorations.answer_hidden.selected_features import SEQUENCE_FEATURES

# Measurable without an eye tracker (see the module docstring).
NO_TRACKER_COLS: List[str] = [
    Con.TOTAL_ANSWERING_RT,
    Con.TOTAL_ANSWERING_RT_NORMALIZED,
    Con.NUM_OF_SELECTS,
    "select_to_confirm_ms",
]

# Column name starts that measure how long the eyes stayed somewhere.
GAZE_TIME_PREFIXES = (
    *Con.RT_TFD_METRICS,
    Con.MEAN_DWELL_TIME,
    Con.AREA_DWELL_PROPORTION,
    Con.MEAN_FIRST_FIXATION_DURATION,
    "first_look_to_select_ms",
)


def is_gaze_time(col: str) -> bool:
    """True if `col` measures how long the eyes stayed somewhere."""
    return col.startswith(GAZE_TIME_PREFIXES)


def needs_tracker(col: str) -> bool:
    """True if `col` cannot be measured without an eye tracker."""
    return col not in NO_TRACKER_COLS


def _gaze_wide() -> List[str]:
    """The base, back-and-forth, visits and the order features of the chosen answer: a
    wider set, before anything is taken out of it."""
    groups = [base_cols(), DELIBERATION_GROUPS["back_and_forth"], DELIBERATION_GROUPS["visits"], SEQUENCE_FEATURES]
    return list(dict.fromkeys(c for g in groups for c in g))


def _without(cols: Sequence[str], *, tracker_free: bool, gaze_time: bool) -> List[str]:
    """`cols` with the features that need no eye tracker and / or the gaze times taken out."""
    return [
        c for c in cols
        if not (tracker_free and not needs_tracker(c)) and not (gaze_time and is_gaze_time(c))
    ]


def tracker_feature_sets() -> Dict[str, List[str]]:
    """The sets compared, in the order shown:

    * `no_tracker_time` -- the answering time per word alone;
    * `no_tracker` -- everything measurable without an eye tracker (`NO_TRACKER_COLS`);
    * `chosen` -- the chosen set, as on the slides;
    * `chosen_tracker_only` -- the chosen set without what needs no eye tracker;
    * `chosen_no_time` -- the chosen set with neither that nor the gaze times;
    * `gaze_wide_tracker_only`, `gaze_wide_no_time` -- the same two cuts of a wider set
      (the base, back-and-forth, visits, and the order features of the chosen answer);
    * `gaze_wide_no_time+no_tracker` -- the wider no-time set with the no-tracker
      features added back, to show what the eye tracker adds to them.
    """
    chosen, wide = chosen_cols(), _gaze_wide()
    gaze_no_time = _without(wide, tracker_free=True, gaze_time=True)
    return {
        "no_tracker_time": [Con.TOTAL_ANSWERING_RT_NORMALIZED],
        "no_tracker": list(NO_TRACKER_COLS),
        "chosen": chosen,
        "chosen_tracker_only": _without(chosen, tracker_free=True, gaze_time=False),
        "chosen_no_time": _without(chosen, tracker_free=True, gaze_time=True),
        "gaze_wide_tracker_only": _without(wide, tracker_free=True, gaze_time=False),
        "gaze_wide_no_time": gaze_no_time,
        "gaze_wide_no_time+no_tracker": [*gaze_no_time, *NO_TRACKER_COLS],
    }


# The differences tested, each (the smaller set, the larger one), with what it asks.
ADDED_VALUE_PAIRS: Dict[Tuple[str, str], str] = {
    ("no_tracker", "gaze_wide_no_time+no_tracker"): "gaze (no time) added to the no-tracker features",
    ("no_tracker", "chosen"): "the chosen set against the no-tracker features",
    ("gaze_wide_no_time", "gaze_wide_no_time+no_tracker"): "the no-tracker features added to gaze (no time)",
}


def tracker_predictions(
    tables: Dict[str, pd.DataFrame],
    pairs: Sequence[Tuple[str, str]] = tuple(ADDED_VALUE_PAIRS),
    *,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """The trial-by-trial predictions of every set in `pairs` (`run.predictions_for`):
    Study 1's held-out ones from the comparison's cross-validation, Study 2's from one fit
    on all of Study 1. Saved and read back while the sets, model and tables are the same."""
    all_sets = tracker_feature_sets()
    sets = {m: all_sets[m] for m in dict.fromkeys(m for pair in pairs for m in pair)}
    hidden = ("l1_hidden", "knowqa_hidden")
    inputs = {
        "sets": sets,
        "model": model_key(TrialLevelLogRegModel),
        "tables": {k: table_digest(tables[k]) for k in hidden},
    }
    return cached(
        "tracker_predictions", inputs,
        lambda: predictions_for(sets, *(tables[k] for k in hidden)),
        rerun=rerun, verbose=verbose,
    )


def _balanced_accuracy(y: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * ((y_pred[y == 1] == 1).mean() + (y_pred[y == 0] == 0).mean())


def _paired(rows: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """One row per prediction, a column per model and measure: the models side by side on
    the same trials. Stops unless every model predicted exactly the same rows with the same
    outcome, since the test pairs them."""
    keys = ["fold", Con.PARTICIPANT_ID, Con.TRIAL_ID]
    rows = rows[rows["model"].isin(models)].astype({"model": str, Con.PARTICIPANT_ID: str, Con.TRIAL_ID: str})
    # Study 2 has no folds; one placeholder keeps the key whole.
    rows = rows.assign(fold=rows["fold"].fillna(-1))
    assert not rows.duplicated([*keys, "model"]).any(), "a model predicted a row twice"
    wide = rows.pivot(index=keys, columns="model", values=["y_true", "y_pred", "y_prob"])
    assert not wide.isna().any().any(), "the models did not all predict the same rows"
    y = wide["y_true"][models[0]]
    for m in models[1:]:
        assert (wide["y_true"][m] == y).all(), f"{m}: outcomes differ from {models[0]}'s"
    return wide


def added_value_test(
    preds: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]] = tuple(ADDED_VALUE_PAIRS),
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """For each pair (smaller set, larger set) and each test of `run.CONFUSION_TESTS`: the
    AUC and the balanced accuracy of both models on the same held-out predictions, the
    larger minus the smaller, and its interval and two-sided p from resampling
    participants. Every model of a test is resampled with the same draws, so each
    difference is paired. The p-value is twice the smaller share of resampled differences
    on either side of zero, and cannot go below 1 / the number of resamples.

    Study 1's held-out kinds pool each fold's validation and test halves, so a trial held
    out in several folds counts once per fold, as in the comparison. The resampling covers
    which people were tested, not which were trained on. Saved and read back while the
    predictions and the settings are the same.
    """
    models = list(dict.fromkeys(m for pair in pairs for m in pair))

    def compute():
        lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
        out = []
        for dataset, tests in CONFUSION_TESTS.items():
            for test in tests:
                rows = preds[preds["dataset"] == dataset]
                if test != POOLED:
                    rows = rows[rows["test"] == test]
                wide = _paired(rows, models)
                y = wide["y_true"][models[0]].to_numpy(int)
                prob = {m: wide["y_prob"][m].to_numpy(float) for m in models}
                pred = {m: wide["y_pred"][m].to_numpy(int) for m in models}
                person = wide.index.get_level_values(Con.PARTICIPANT_ID)
                blocks = list(pd.Series(np.arange(len(wide))).groupby(np.asarray(person)).indices.values())
                draws = np.random.default_rng(seed).integers(0, len(blocks), size=(n_boot, len(blocks)))

                boots = {m: [] for m in models}
                for draw in draws:
                    idx = np.concatenate([blocks[i] for i in draw])
                    yy = y[idx]
                    if yy.min() == yy.max():
                        continue
                    for m in models:
                        boots[m].append((auc(yy, prob[m][idx]), _balanced_accuracy(yy, pred[m][idx])))
                boots = {m: np.array(b) for m, b in boots.items()}

                for small, large in pairs:
                    row = {
                        "dataset": dataset, "test": test, "smaller": small, "larger": large,
                        "question": ADDED_VALUE_PAIRS.get((small, large), ""),
                        "n": len(y), "n_wrong": int((y == 0).sum()), "n_participants": len(blocks),
                    }
                    for j, (measure, score) in enumerate((("auc", auc), ("bacc", _balanced_accuracy))):
                        values = prob if measure == "auc" else pred
                        a, b = score(y, values[small]), score(y, values[large])
                        d = boots[large][:, j] - boots[small][:, j]
                        row |= {
                            f"{measure}_smaller": a, f"{measure}_larger": b, f"{measure}_diff": b - a,
                            f"{measure}_diff_ci_low": np.percentile(d, lo_q),
                            f"{measure}_diff_ci_high": np.percentile(d, hi_q),
                            f"{measure}_p": float(np.clip(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d), 1)),
                        }
                    row["n_boot_ok"] = len(boots[small])
                    row[MODEL_FAMILY_COL] = model_folder(rows[rows["model"].isin(models)])
                    out.append(row)
                if verbose:
                    print(f"  {dataset} / {test}: done")
        return pd.DataFrame(out).assign(n_boot=n_boot, ci=ci, seed=seed)

    inputs = {"predictions": table_digest(preds), "pairs": [list(p) for p in pairs],
              "n_boot": n_boot, "ci": ci, "seed": seed}
    return cached("tracker_added_value", inputs, compute, rerun=rerun, verbose=verbose)


# One colour and one marker per pair, so a pair is never told apart by colour alone.
PAIR_STYLE = {
    ("no_tracker", "gaze_wide_no_time+no_tracker"): ("#4C72B0", "o"),
    ("no_tracker", "chosen"): ("#DD8452", "s"),
    ("gaze_wide_no_time", "gaze_wide_no_time+no_tracker"): ("#55A868", "D"),
}

TEST_LABELS = {
    ("l1", "new_item"): "Study 1: new paragraphs",
    ("l1", "new_subject"): "Study 1: new people",
    ("l1", "both"): "Study 1: both new",
    ("knowqa", "no knowledge"): "Study 2: no knowledge",
    ("knowqa", "partial knowledge"): "Study 2: partial",
    ("knowqa", "full knowledge"): "Study 2: full",
    ("knowqa", POOLED): "Study 2: all",
}


def plot_added_value(
    table: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """The differences of `added_value_test`: the AUC difference and the balanced-accuracy
    difference, with their 95% intervals, a row per test and a mark per pair; right of the
    dashed line, the larger set does better. Saved with the table. Returns the figure."""
    tests = list(dict.fromkeys(zip(table["dataset"], table["test"])))
    pairs = list(dict.fromkeys(zip(table["smaller"], table["larger"])))
    offsets = np.linspace(-0.25, 0.25, len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.62 * len(tests) + 1.6), sharey=True, layout="constrained")
    for ax, (measure, title) in zip(axes, (("auc", "AUC"), ("bacc", "Balanced accuracy"))):
        for (small, large), off in zip(pairs, offsets):
            colour, marker = PAIR_STYLE[(small, large)]
            sub = table[(table["smaller"] == small) & (table["larger"] == large)]
            for _, r in sub.iterrows():
                y = tests.index((r["dataset"], r["test"])) + off
                ax.errorbar(
                    r[f"{measure}_diff"], y,
                    xerr=[[r[f"{measure}_diff"] - r[f"{measure}_diff_ci_low"]],
                          [r[f"{measure}_diff_ci_high"] - r[f"{measure}_diff"]]],
                    fmt=marker, color=colour, ms=6, elinewidth=1.5, capsize=0,
                )
        ax.axvline(0, color="#555555", lw=1, ls="--")
        # Study 1 and Study 2 apart.
        n_l1 = sum(ds == "l1" for ds, _ in tests)
        ax.axhline(n_l1 - 0.5, color="#BBBBBB", lw=1)
        ax.set_xlabel(f"{title}: larger set minus smaller")
        ax.set_title(f"{title} difference, 95% interval")
        ax.grid(axis="x", color="#E5E5E5", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(tests)))
    axes[0].set_yticklabels([TEST_LABELS.get(t, f"{t[0]}: {t[1]}") for t in tests])
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=PAIR_STYLE[p][0], marker=PAIR_STYLE[p][1], ls="", ms=6,
                   label=ADDED_VALUE_PAIRS.get(p, f"{p[1]} vs {p[0]}"))
        for p in pairs
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle("Does the eye tracker add to the press log? Answer-hidden model, paired by trial", fontsize=11)
    save_output(
        fig, analysis=ANALYSIS, plot="tracker_added_value", tables={"differences": table},
        save=save, close=close, subdir=model_folder(table),
    )
    return fig


def main(save: bool = True, verbose: bool = True, rerun: bool = False) -> Dict[str, pd.DataFrame]:
    """The sets of `tracker_feature_sets`, scored and saved as a heatmap with its scores,
    and the paired test of `ADDED_VALUE_PAIRS`, saved as a figure with its table. Returns
    the comparison and the test."""
    tables = load_tables(verbose=verbose)
    sets = tracker_feature_sets()
    search = run_deliberation_search(tables, sets, verbose=verbose, rerun=rerun)
    plot_search_heatmap(search["scores"], search="tracker", order=list(sets), save=save, close=True)
    preds = tracker_predictions(tables, verbose=verbose, rerun=rerun)
    test = added_value_test(preds, verbose=verbose, rerun=rerun)
    plot_added_value(test, save=save, close=True)
    return {**search, "added_value": test}


if __name__ == "__main__":
    main()
