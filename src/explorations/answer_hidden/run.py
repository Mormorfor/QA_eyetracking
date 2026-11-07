"""The paper's correctness model, scored with and without knowing which answer is correct.

Every model here comes in four versions:

* **known** -- a feature set from the paper, on the paper's trial table;
* **position**, **selected**, **spread** -- the same set passed through
  `table.hide_answers`, on the table from `table.build_answer_hidden_table`, with the
  answers named by screen position, by whether they were selected, or summarised over all
  four. Named `<paper model>_<naming>`, e.g. `select_1_selected`.

All are scored by the paper's own comparison (`analyses/correctness_prediction/
comparison.py::run_model_comparison`): Study 1 by the ten-fold cross-validation, held-out
participants and paragraphs; Study 2 (KnowQA) by one fit on every Study 1 trial, scored on
each knowledge regime and on all of them together. Every version uses the same folds and
the same trials, so the gap to the known version is what knowing the correct answer is
worth.

The comparison and figure functions are the paper's, imported rather than copied, so the
versions cannot drift apart in how they are scored. Those functions save under the paper's
analysis, so they are called with `save=False` here and the results are saved under
`reports/explorations/answer_hidden/`.

The confusion matrices (`collect_predictions`, `plot_confusion_grid`) come from the same
predictions as the comparison's scores: on Study 1, the held-out trials of all ten folds,
which the comparison scores and then drops; on Study 2, the one fit on Study 1.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import GATHERERS_REFOLDED_DIR, HUNTERS_REFOLDED_DIR, dataset
from src.features.build import PARAGRAPH_MODEL_COLS, load_model_ready
import src.modeling.feature_sets as FS
from src.modeling.confusion import confusion_counts, label_quadrants
from src.modeling.crossval import run_cross_validation_on_combined_folds
from src.modeling.evaluate import AVERAGE_NOVELTY, collect_logreg_coef_summaries
from src.modeling.folds import parse_regime
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.lib.plotting.output import save_output
from src.analyses.correctness_prediction.comparison import (
    PERSON_LEVEL_COLS,
    POOLED,
    TESTS,
    run_model_comparison,
)
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    PRED_COL,
    PROB_COL,
    REGIME_COL,
    attach_predictions,
    fit_l1_model,
    load_showcase_frame,
)
from src.analyses.correctness_prediction.plots.coefficients import plot_coef_summary_barh
from src.analyses.correctness_prediction.plots.model_comparison import (
    TEST_TITLES,
    plot_model_comparison,
)
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.table import NAMINGS, hide_answers, load_hidden_table

ANALYSIS = "explorations/answer_hidden"

# The paper's two headline models: the ones whose coefficients `main` saves.
PAPER_MODELS = ("select_1", "select_1_plus_last_confirm")

# The columns of the confusion grids, left to right.
VERSION_ORDER = ("known", *NAMINGS)

# The tests that get a confusion grid: the comparison's, less Study 1's average of the
# three held-out kinds, which is a mean of scores rather than a set of trials.
CONFUSION_TESTS = {
    "l1": [t for t in TESTS["l1"] if t != AVERAGE_NOVELTY],
    "knowqa": list(TESTS["knowqa"]),
}

# The columns of `collect_predictions`, one row per prediction. `fold` is Study 1's
# cross-validation fold, empty on Study 2.
PREDICTION_COLUMNS = [
    "dataset", "test", "model", "paper_model", "answers", "fold",
    Con.PARTICIPANT_ID, Con.TRIAL_ID, "y_true", "y_pred", "y_prob", MODEL_FAMILY_COL,
]


def known_feature_sets(l1_trial_df: pd.DataFrame) -> Dict[str, List[str]]:
    """The paper's models that get answer-free versions.

    * `select_1` and `select_1_plus_last_confirm`: the paper's ten- and twelve-feature
      models, from `configs/feature_sets/`.
    * `all_features`: every model feature in the table (`get_full_feature_cols`), less the
      opening-scan features, which summarise a person over several trials and which the
      comparison refuses, and less the paragraph features (see `table.py`).
    """
    saved = FS.load_feature_sets()
    full = [
        c for c in FS.get_full_feature_cols(l1_trial_df)
        if c not in PERSON_LEVEL_COLS and c not in PARAGRAPH_MODEL_COLS
    ]
    return {
        "select_1": saved["select_1"],
        "select_1_plus_last_confirm": saved["select_1_plus_last_confirm"],
        "all_features": full,
    }


def answer_free_feature_sets(known: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Each known set under each naming of `table.NAMINGS`, named `<name>_<naming>`."""
    return {
        f"{name}_{by}": hide_answers(cols, by=by)
        for name, cols in known.items()
        for by in NAMINGS
    }


def _versions(known_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """One row per model version: `model`, the `paper_model` it comes from, and `answers`,
    "known" or the naming."""
    return pd.DataFrame(
        [{"model": name, "paper_model": name, "answers": "known"} for name in known_sets]
        + [
            {"model": f"{name}_{by}", "paper_model": name, "answers": by}
            for name in known_sets
            for by in NAMINGS
        ]
    )


def l1_trial_ids() -> pd.DataFrame:
    """Study 1's trials with their paragraph ids, which match trials to the fold files.
    The comparison reads only these from its first argument; the trial-grain table has
    them."""
    return pd.read_csv(
        dataset("l1").trials,
        usecols=[Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.TEXT_ID_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )


def load_tables(verbose: bool = True, rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    """The four trial tables: each study with the answers known and hidden.

    Study 2's known table is the one the paper scores (`load_showcase_frame`), with the
    regime and confidence rating attached. Each hidden table is built from its known
    table, with the selected-answer features added, and saved beside the paper's table
    (`table.load_hidden_table`): read back while the known table and the source files are
    unchanged, rebuilt when they change or with `rebuild=True`.
    """
    l1 = load_model_ready()
    knowqa = load_showcase_frame(verbose=verbose)
    return {
        "l1": l1,
        "l1_hidden": load_hidden_table(l1, "l1", rebuild=rebuild, verbose=verbose),
        "knowqa": knowqa,
        "knowqa_hidden": load_hidden_table(knowqa, "knowqa", rebuild=rebuild, verbose=verbose),
    }


def model_key(model_builder: Callable[[], Any]) -> str:
    """The model in a cache key: its class and settings, as the unfitted model prints."""
    return repr(model_builder())


def compare_in_batches(
    sets: Mapping[str, Sequence[str]],
    l1: pd.DataFrame,
    knowqa: pd.DataFrame,
    *,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    batch_size: int = 8,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """The paper's `run_model_comparison` on `sets`, `batch_size` sets at a time, with the
    results stacked.

    In batches because the cross-validation keeps every fold's tables for every model it
    is given, which for dozens of sets fills the memory. Each model is scored on its own,
    so the batches give the same numbers as one call.
    """
    names = list(sets)
    trial_ids = l1_trial_ids()
    parts = []
    for start in range(0, len(names), batch_size):
        batch = {name: list(sets[name]) for name in names[start:start + batch_size]}
        parts.append(run_model_comparison(
            trial_ids, batch, trial_df=l1, knowqa_df=knowqa,
            model_builder=model_builder, verbose=verbose,
        ))
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in ("scores", "folds")}


def run_answer_hidden_comparison(
    tables: Mapping[str, pd.DataFrame],
    known_sets: Mapping[str, Sequence[str]],
    *,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    verbose: bool = True,
    rerun: bool = False,
) -> Dict[str, pd.DataFrame]:
    """`run_model_comparison` on the known tables and on the hidden ones, with the
    results stacked.

    `tables` is what `load_tables` returns, `known_sets` the paper's sets
    (`known_feature_sets`); the answer-free ones are made from them. Returns the
    comparison's `scores` and `folds`, with an `answers` column: "known", or the naming.

    Saved after the first run and read back while the sets, the model and the tables are
    the same (`cache.py`); `rerun=True` runs it again.
    """
    free_sets = answer_free_feature_sets(known_sets)
    answers = _versions(known_sets).set_index("model")["answers"]

    def compute():
        parts = []
        for sets, l1, knowqa in (
            (known_sets, tables["l1"], tables["knowqa"]),
            (free_sets, tables["l1_hidden"], tables["knowqa_hidden"]),
        ):
            out = compare_in_batches(sets, l1, knowqa, model_builder=model_builder, verbose=verbose)
            parts.append({k: v.assign(answers=v["model"].map(answers)) for k, v in out.items()})
        return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in ("scores", "folds")}

    inputs = {
        "known_sets": {k: list(v) for k, v in known_sets.items()},
        "free_sets": free_sets,
        "model": model_key(model_builder),
        "tables": {k: table_digest(t) for k, t in tables.items()},
    }
    return cached("comparison", inputs, compute, rerun=rerun, verbose=verbose)


def model_order(known_sets: Mapping[str, Sequence[str]]) -> List[str]:
    """Each known model followed by its answer-free versions: the order the figures use."""
    return [n for name in known_sets for n in (name, *(f"{name}_{by}" for by in NAMINGS))]


def plot_answer_hidden_comparison(
    comparison: Mapping[str, pd.DataFrame],
    *,
    dataset: str,
    models: Sequence[str],
    save: Optional[bool] = None,
    close: Optional[bool] = None,
    **facets,
):
    """The paper's model-comparison figure for `dataset` ("l1" or "knowqa"), saved here
    with its scores and, for Study 1, the fold rows behind them. `**facets` go into the
    file name, so figures of different model sets do not overwrite each other. Returns
    the figure and the scores it shows."""
    fig, plotted = plot_model_comparison(
        comparison, dataset=dataset, models=models, save=False, close=False,
    )
    tables = {"scores": plotted}
    if dataset == "l1":
        folds = comparison["folds"]
        tables["folds"] = folds[folds["model"].isin(plotted["model"])]
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="model_comparison",
        tables=tables,
        save=save,
        close=close,
        subdir=model_folder(plotted),
        dataset=dataset,
        **facets,
    )
    return fig, plotted


def predictions_for(
    sets: Mapping[str, Sequence[str]],
    l1: pd.DataFrame,
    knowqa: pd.DataFrame,
    *,
    n_folds: int = 10,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    batch_size: int = 4,
) -> pd.DataFrame:
    """The predictions of every set in `sets`, trial by trial: Study 1's held-out
    predictions of the comparison's cross-validation on `l1`, and Study 2's from one fit
    on all of `l1`, scored on `knowqa` (see `collect_predictions`). The columns of
    `PREDICTION_COLUMNS` except `paper_model` and `answers`, which say what the set is."""
    trial_ids = l1_trial_ids()
    names = list(sets)
    parts = []
    for start in range(0, len(names), batch_size):
        batch = {name: list(sets[name]) for name in names[start:start + batch_size]}
        cv = run_cross_validation_on_combined_folds(
            trial_ids,
            fold_dirs=[HUNTERS_REFOLDED_DIR, GATHERERS_REFOLDED_DIR],
            model_builders={name: model_builder for name in batch},
            feature_cols_by_model=batch,
            target_col=Con.IS_CORRECT_COLUMN,
            n_folds=n_folds,
            trial_df=l1,
        )
        for name in batch:
            for fold, regimes in cv.per_fold_results[name].items():
                for regime, res in regimes.items():
                    # The fold's held-out trials, in the order of its predictions.
                    assert len(res.eval_df) == len(res.y_true) == len(res.y_prob)
                    parts.append(pd.DataFrame({
                        "dataset": "l1",
                        "test": parse_regime(regime)[1],
                        "model": name,
                        "fold": fold,
                        Con.PARTICIPANT_ID: res.eval_df[Con.PARTICIPANT_ID].to_numpy(),
                        Con.TRIAL_ID: res.eval_df[Con.TRIAL_ID].to_numpy(),
                        "y_true": np.asarray(res.y_true, dtype=int),
                        "y_pred": np.asarray(res.y_pred, dtype=int),
                        "y_prob": np.asarray(res.y_prob, dtype=float),
                        MODEL_FAMILY_COL: res.model_family,
                    }))

    for name, cols in sets.items():
        model = fit_l1_model(l1, feature_cols=cols, model_builder=model_builder, verbose=False)
        scored = attach_predictions(knowqa, model, feature_cols=cols, verbose=False)
        parts.append(pd.DataFrame({
            "dataset": "knowqa",
            "test": scored[REGIME_COL],
            "model": name,
            Con.PARTICIPANT_ID: scored[Con.PARTICIPANT_ID],
            Con.TRIAL_ID: scored[Con.TRIAL_ID].astype(str),
            "y_true": scored[Con.IS_CORRECT_COLUMN].astype(int),
            "y_pred": scored[PRED_COL].astype(int),
            "y_prob": scored[PROB_COL].astype(float),
            MODEL_FAMILY_COL: scored[MODEL_FAMILY_COL],
        }))
    preds = pd.concat(parts, ignore_index=True)
    # Study 1's trial ids are numbers and Study 2's text, so the shared column is text.
    preds[Con.TRIAL_ID] = preds[Con.TRIAL_ID].astype(str)
    return preds


def as_prediction_table(preds: pd.DataFrame) -> pd.DataFrame:
    """`preds` with `PREDICTION_COLUMNS` in order, the labels stored as categories: the
    same few labels repeat on every row, and categories keep a saved table small."""
    preds = preds.copy()
    for col in ("dataset", "test", "model", "paper_model", "answers", MODEL_FAMILY_COL,
                Con.PARTICIPANT_ID, Con.TRIAL_ID):
        preds[col] = preds[col].astype("category")
    return preds[PREDICTION_COLUMNS]


def collect_predictions(
    tables: Mapping[str, pd.DataFrame],
    known_sets: Mapping[str, Sequence[str]],
    *,
    n_folds: int = 10,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    batch_size: int = 4,
    verbose: bool = True,
    rerun: bool = False,
) -> pd.DataFrame:
    """Every model version's predictions, trial by trial: what the confusion matrices
    count.

    * Study 1: the held-out predictions of the comparison's ten-fold cross-validation,
      run again here with the same folds, tables and models, because the comparison
      keeps only the scores. `test` is the held-out kind ("new_item", "new_subject",
      "both"); its validation and test halves are pooled, as in the comparison.
    * Study 2: the predictions of one fit on every Study 1 trial, as in the comparison.
      `test` is the knowledge regime.

    One row per prediction, with `PREDICTION_COLUMNS`; a Study 1 trial held out in more
    than one fold has a row for each. The cross-validation runs
    `batch_size` models at a time, to keep the memory down (`compare_in_batches`). Saved
    and read back as `run_answer_hidden_comparison` is.
    """
    free_sets = answer_free_feature_sets(known_sets)

    def compute():
        preds = pd.concat([
            predictions_for(sets, l1, knowqa, n_folds=n_folds, model_builder=model_builder, batch_size=batch_size)
            for sets, l1, knowqa in (
                (known_sets, tables["l1"], tables["knowqa"]),
                (free_sets, tables["l1_hidden"], tables["knowqa_hidden"]),
            )
        ], ignore_index=True)
        preds = preds.merge(_versions(known_sets), on="model", how="left", validate="many_to_one")
        return as_prediction_table(preds)

    inputs = {
        "known_sets": {k: list(v) for k, v in known_sets.items()},
        "free_sets": free_sets,
        "model": model_key(model_builder),
        "n_folds": n_folds,
        "tables": {k: table_digest(t) for k, t in tables.items()},
        # What the result holds: a change here is a change in the result.
        "output": PREDICTION_COLUMNS,
    }
    return cached("predictions", inputs, compute, rerun=rerun, verbose=verbose)


def confusion_table(preds: pd.DataFrame, *, dataset: str, test: str) -> pd.DataFrame:
    """The confusion counts of every model version on one test, from `collect_predictions`.

    One row per version: `paper_model`, `answers`, the four counts (TP, TN, FP, FN, with
    "positive" meaning the participant was right; `modeling/confusion.py`), `n`, and the
    sensitivity, specificity and balanced accuracy worked out from them. Balanced
    accuracy needs both outcomes, so it is NaN when one is absent. Study 2's `all` pools
    the regimes.

    On Study 1 these are counted over the pooled held-out trials of all ten folds, so the
    balanced accuracy can differ slightly from the comparison's, which is the mean of the
    ten folds' own scores.
    """
    rows = preds[preds["dataset"] == dataset]
    if test != POOLED:
        rows = rows[rows["test"] == test]
    if rows.empty:
        raise ValueError(f"no {dataset} predictions for test {test!r}")

    out = []
    keys = ["model", "paper_model", "answers", MODEL_FAMILY_COL]
    for (model, paper_model, answers, family), g in rows.groupby(keys, sort=False, observed=True):
        c = confusion_counts(label_quadrants(g))
        sens = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else np.nan
        spec = c["TN"] / (c["TN"] + c["FP"]) if c["TN"] + c["FP"] else np.nan
        out.append({
            "dataset": dataset, "test": test, "model": model,
            "paper_model": paper_model, "answers": answers,
            **{q: int(c[q]) for q in ("TP", "TN", "FP", "FN")},
            "n": len(g),
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2,
            MODEL_FAMILY_COL: family,
        })
    return pd.DataFrame(out)


def plot_confusion_grid(
    preds: pd.DataFrame,
    *,
    dataset: str,
    test: str,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """The confusion matrix of every model version on one test, as a grid: a row per
    paper model, a column per version (known, then the three answer-free namings).

    Each cell shows the count and, below it, its share of the row's actual outcome; the
    shading is that share, so the panels compare across versions whatever the accuracy.
    The two diagonal shares are the specificity (top left) and the sensitivity (bottom
    right), and each title gives their mean, the balanced accuracy. Saved with the
    counts (`confusion_table`). Returns the figure and the counts.
    """
    table = confusion_table(preds, dataset=dataset, test=test)
    paper_models = list(dict.fromkeys(table["paper_model"]))

    fig, axes = plt.subplots(
        len(paper_models), len(VERSION_ORDER),
        figsize=(2.7 * len(VERSION_ORDER) + 1.6, 2.5 * len(paper_models) + 1.2),
        squeeze=False, layout="constrained",
    )
    for i, paper_model in enumerate(paper_models):
        for j, answers in enumerate(VERSION_ORDER):
            ax = axes[i, j]
            cell = table[(table["paper_model"] == paper_model) & (table["answers"] == answers)]
            assert len(cell) == 1, f"{paper_model}/{answers}: {len(cell)} rows"
            r = cell.iloc[0]
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
            with np.errstate(invalid="ignore"):
                shown = cm / cm.sum(axis=1, keepdims=True)

            ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(
                        b, a, f"{int(cm[a, b])}\n{shown[a, b]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if shown[a, b] > 0.55 else "black",
                    )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred wrong", "pred correct"] if i == len(paper_models) - 1 else [], fontsize=8)
            ax.set_yticklabels(["was wrong", "was correct"] if j == 0 else [], fontsize=8)
            if j == 0:
                ax.set_ylabel(paper_model, fontsize=9)
            ax.set_title(f"{answers}\nbalanced acc {r['balanced_accuracy']:.3f}", fontsize=9)

    where = (
        "Study 1 (L1): held-out trials of the ten folds"
        if dataset == "l1" else "Study 2 (KnowQA): trained on every Study 1 trial"
    )
    n = table["n"].unique()
    assert len(n) == 1, f"the versions scored different numbers of trials: {n}"
    fig.suptitle(
        f"{where}\n{TEST_TITLES.get(test, test).replace(chr(10), ': ')}, {int(n[0])} predictions"
        " (shares are of each row)",
        fontsize=11,
    )
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="confusion",
        tables={"counts": table},
        save=save,
        close=close,
        subdir=model_folder(table),
        dataset=dataset,
        test=test,
    )
    return fig, table


# The paper's colours for the two outcomes (`knowledge_regimes.descriptives`).
OUTCOME_STYLE = ((0, "#c44e52", "was wrong"), (1, "#4c72b0", "was correct"))


def plot_probability_grid(
    preds: pd.DataFrame,
    *,
    dataset: str,
    paper_model: str,
    bins: int = 20,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Predicted P(correct) for every version of one paper model, split into answers that
    were right and wrong, drawn as the paper's Study 2 histograms are
    (`knowledge_regimes.descriptives.plot_probability_histograms`): a row per version, a
    column per test (Study 1's held-out kinds, or Study 2's regimes), the dashed line the
    panel's mean.

    Each panel asks whether the two humps sit in different places: whether the version
    tells right from wrong answers apart within that test, rather than only following its
    base rate. On Study 1 the predictions are pooled over the ten folds, so a trial held
    out in more than one fold counts each time. Saved with the binned counts and the
    means. Returns the figure.
    """
    tests = [t for t in CONFUSION_TESTS[dataset] if t != POOLED]
    rows = preds[(preds["dataset"] == dataset) & (preds["paper_model"] == paper_model)]
    edges = np.linspace(0, 1, bins + 1)

    fig, axes = plt.subplots(
        len(VERSION_ORDER), len(tests),
        figsize=(4.3 * len(tests), 2.5 * len(VERSION_ORDER) + 0.9),
        sharex=True, sharey="row", squeeze=False, layout="constrained",
    )
    binned, means = [], []
    for i, answers in enumerate(VERSION_ORDER):
        for j, test in enumerate(tests):
            ax = axes[i, j]
            sub = rows[(rows["answers"] == answers) & (rows["test"] == test)]
            assert len(sub), f"no {dataset} predictions for {paper_model}/{answers}/{test}"
            key = {"model": sub["model"].iloc[0], "answers": answers, "test": test,
                   MODEL_FAMILY_COL: model_folder(sub)}
            for outcome, colour, tag in OUTCOME_STYLE:
                probs = sub.loc[sub["y_true"] == outcome, "y_prob"].astype(float)
                counts, _ = np.histogram(probs, bins=edges)
                # Drawn from the counts that are saved, so the two cannot disagree.
                ax.hist(edges[:-1], bins=edges, weights=counts, alpha=0.65, color=colour,
                        label=f"{tag} (n={len(probs)})")
                binned += [
                    {**key, "outcome": tag, "bin_low": lo, "bin_high": hi, "n": int(n)}
                    for lo, hi, n in zip(edges[:-1], edges[1:], counts)
                ]
                means.append({**key, "outcome": tag, "n": len(probs), "mean_prob": probs.mean()})
            mean = sub["y_prob"].astype(float).mean()
            means.append({**key, "outcome": "all", "n": len(sub), "mean_prob": mean})
            ax.axvline(mean, ls="--", c="k", lw=1.2)
            ax.set_title((f"{test}\n" if i == 0 else "") + f"mean P = {mean:.3f}", fontsize=9.5)
            ax.legend(fontsize=7, frameon=False)
            if j == 0:
                ax.set_ylabel(f"{answers}\n{'predictions' if dataset == 'l1' else 'trials'}", fontsize=9)
            if i == len(VERSION_ORDER) - 1:
                ax.set_xlabel("predicted P(correct)")

    where = (
        "Study 1 (L1), held-out trials of the ten folds"
        if dataset == "l1" else "Study 2 (KnowQA), trained on every Study 1 trial"
    )
    fig.suptitle(f"{paper_model}: predicted P(correct) by outcome -- {where}", fontsize=11)
    tables = {"bins": pd.DataFrame(binned), "means": pd.DataFrame(means)}
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="probability_histograms",
        tables=tables,
        save=save,
        close=close,
        subdir=model_folder(tables["means"]),
        dataset=dataset,
        model=paper_model,
    )
    return fig


# The paper's regime colours (`knowledge_regimes.descriptives`); Study 1's held-out kinds
# share one.
TEST_COLOURS = {
    "no knowledge": "#c44e52",
    "partial knowledge": "#dd8452",
    "full knowledge": "#4c72b0",
    "new_item": "#4c72b0",
    "new_subject": "#4c72b0",
    "both": "#4c72b0",
}


def plot_probability_boxes(
    preds: pd.DataFrame,
    *,
    dataset: str,
    paper_model: str,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Predicted P(correct) as box plots, drawn as the paper's Study 2 figure is
    (`knowledge_regimes.descriptives.plot_probability_by_regime`): for each test, a box for
    the answers that were right (in the test's colour) and one for those that were wrong
    (grey); whiskers at 1.5 x the interquartile range, no outliers drawn; the dotted line
    at 0.5. A panel per version of `paper_model`, side by side on one y-axis.

    The boxes are drawn from `matplotlib.cbook.boxplot_stats`, and the same statistics are
    saved, so the table is exactly what the figure shows. On Study 1 the predictions are
    pooled over the ten folds. Returns the figure.
    """
    from matplotlib.cbook import boxplot_stats

    tests = [t for t in CONFUSION_TESTS[dataset] if t != POOLED]
    rows = preds[(preds["dataset"] == dataset) & (preds["paper_model"] == paper_model)]
    fig, axes = plt.subplots(
        1, len(VERSION_ORDER), sharey=True, layout="constrained",
        figsize=(3.4 * len(VERSION_ORDER) + 0.8, 4.6),
    )
    saved = []
    for ax, answers in zip(axes, VERSION_ORDER):
        stats, positions, colours, labels = [], [], [], []
        for i, test in enumerate(tests):
            for j, (outcome, tag) in enumerate([(1, "correct"), (0, "wrong")]):
                sub = rows[(rows["answers"] == answers) & (rows["test"] == test) & (rows["y_true"] == outcome)]
                assert len(sub), f"no {dataset} predictions for {paper_model}/{answers}/{test}/{tag}"
                s = boxplot_stats(sub["y_prob"].to_numpy(float))[0]
                stats.append(s)
                positions.append(i * 2.6 + j * 0.9)
                colours.append(TEST_COLOURS[test] if outcome else "#cccccc")
                labels.append(f"{test.split()[0]}\n{tag}\nn={len(sub)}")
                saved.append({
                    "model": sub["model"].iloc[0], "answers": answers, "test": test, "outcome": tag,
                    "n": len(sub), "mean": s["mean"], "whisker_low": s["whislo"], "q1": s["q1"],
                    "median": s["med"], "q3": s["q3"], "whisker_high": s["whishi"],
                    MODEL_FAMILY_COL: model_folder(sub),
                })
        bp = ax.bxp(stats, positions=positions, widths=0.75, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0.5, ls=":", c="grey", lw=1)
        ax.set_title(answers, fontsize=10)
    axes[0].set_ylabel("predicted P(correct)")

    where = ("Study 1 (L1), held-out trials of the ten folds" if dataset == "l1"
             else "Study 2 (KnowQA), trained on every Study 1 trial")
    fig.suptitle(f"{paper_model}: predicted P(correct) by outcome -- {where}", fontsize=11)
    table = pd.DataFrame(saved)
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="probability_boxes",
        tables={"boxes": table},
        save=save,
        close=close,
        subdir=model_folder(table),
        dataset=dataset,
        model=paper_model,
    )
    return fig


def collect_answer_hidden_coefficients(
    trial_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    ci_method: str = "bootstrap",
    ci_cluster: str = "cluster",
    verbose: bool = True,
    rerun: bool = False,
) -> Dict[str, pd.DataFrame]:
    """One fit on all of `trial_df` per set, as the paper's coefficients are made: by
    default with bootstrap intervals that resample whole participants (about a minute
    per set on Study 1). Each table carries `model_family`. Each set is saved and read
    back on its own, as `run_answer_hidden_comparison` is, so adding a set runs only that
    one."""
    family = TrialLevelLogRegModel().family

    def one(name, cols):
        coefs = collect_logreg_coef_summaries(
            trial_df, {name: cols}, ci_method=ci_method, ci_cluster=ci_cluster,
        )
        return coefs[name].assign(**{MODEL_FAMILY_COL: family})

    digest = table_digest(trial_df)
    return {
        name: cached(
            "coefficients",
            {"features": list(cols), "ci_method": ci_method, "ci_cluster": ci_cluster,
             "model": model_key(TrialLevelLogRegModel), "table": digest},
            lambda name=name, cols=list(cols): one(name, cols),
            rerun=rerun, verbose=verbose,
        )
        for name, cols in feature_sets.items()
    }


def plot_answer_hidden_coefficients(
    coef: pd.DataFrame,
    *,
    model: str,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """The paper's coefficient bars for one model, every feature shown, saved here with
    the full coefficient table. Returns the figure."""
    fig, _, _ = plot_coef_summary_barh(
        coef,
        significant_only=False,
        title=f"{model}: one fit on every Study 1 trial",
        xlabel="Coefficient (log-odds per SD of the feature)",
        ylabel="",
        figsize=(9, max(4, 0.32 * len(coef) + 1.5)),
        save=False,
    )
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="coefficients",
        tables={"coefficients": coef},
        save=save,
        close=close,
        subdir=model_folder(coef),
        model=model,
    )
    return fig


def main(save: bool = True, verbose: bool = True, rerun: bool = False) -> Dict[str, Any]:
    """Everything above, saved: both comparison figures, a confusion grid and the
    probability histograms per test, and the coefficients of every answer-free version of
    the paper's two headline models. The first run takes several minutes (six
    bootstraps); after that the results are read back (`cache.py`) unless `rerun=True`.
    Returns the comparison, the predictions and the coefficients."""
    tables = load_tables(verbose=verbose)
    known = known_feature_sets(tables["l1"])
    comparison = run_answer_hidden_comparison(tables, known, verbose=verbose, rerun=rerun)
    for ds in TESTS:
        plot_answer_hidden_comparison(
            comparison, dataset=ds, models=model_order(known), save=save, close=True,
        )

    preds = collect_predictions(tables, known, verbose=verbose, rerun=rerun)
    for ds, tests in CONFUSION_TESTS.items():
        for test in tests:
            plot_confusion_grid(preds, dataset=ds, test=test, save=save, close=True)
        for paper_model in known:
            plot_probability_grid(preds, dataset=ds, paper_model=paper_model, save=save, close=True)
            plot_probability_boxes(preds, dataset=ds, paper_model=paper_model, save=save, close=True)

    free = answer_free_feature_sets({n: known[n] for n in PAPER_MODELS})
    coefs = collect_answer_hidden_coefficients(tables["l1_hidden"], free, verbose=verbose, rerun=rerun)
    for name, coef in coefs.items():
        plot_answer_hidden_coefficients(coef, model=name, save=save, close=True)
    plt.close("all")
    return {"comparison": comparison, "predictions": preds, "coefficients": coefs}


if __name__ == "__main__":
    main()
