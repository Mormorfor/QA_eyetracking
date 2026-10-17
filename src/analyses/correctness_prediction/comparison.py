"""Several models compared on the same tests, in Study 1 and in Study 2.

`run_model_comparison` takes any set of models -- each a name and a list of feature
columns -- and scores every one twice:

* **Study 1 (L1)**, by the paper's ten-fold cross-validation. Each fold trains on one set
  of participants reading one set of paragraphs, then scores three kinds of held-out
  trials: a new paragraph (`new_item`), a new participant (`new_subject`), or both.
  `average` is the plain mean of the three. Each has an interval across the ten folds.
* **Study 2 (KnowQA)**, by one fit on every L1 trial, scored on every KnowQA trial: each
  knowledge regime on its own, and all of them together (`all`). One fit and one test
  set, so there is no interval.

Every model is the project's logistic regression; only the features differ. The score is
balanced accuracy, so chance is 0.5 whatever the share of correct answers.

**A model that uses paragraph features is not tested on KnowQA**, because KnowQA has no
paragraph screen. It keeps its KnowQA rows, with no score and a note saying why. It is not
refitted without those columns: that would put another model's score under its name. Any
other column missing from KnowQA stops the run.

**Features that summarise a person over several trials are refused.** The folds are cut
from the ready-made trial table, where the opening-scan features are worked out from all
of a person's trials, held-out ones included (`pitfalls.md` §3). The saved feature sets do
not use them.

`plots/model_comparison.py` draws the result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import GATHERERS_REFOLDED_DIR, HUNTERS_REFOLDED_DIR, dataset
from src.features.build import load_model_ready
import src.modeling.feature_sets as FS
from src.modeling.crossval import run_cross_validation_on_combined_folds
from src.modeling.evaluate import AVERAGE_NOVELTY, summarize_cv_results_by_novelty
from src.modeling.folds import NOVELTY_BY_CELL
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.analyses.correctness_prediction.knowledge_regimes import descriptives as D

# The paper's model comparison: these sets from `configs/feature_sets/`, by name. Named
# rather than "every file in the folder", because other code can save sets there too.
PAPER_MODELS: List[str] = [
    "select_1_plus_last_confirm",
    "last_confirm_compact",
    "select_1",
    "correct_mean_wrong_RT",
    "total_answering_RT_normalized",
]

# The KnowQA test that pools every regime; the same label `descriptives` uses.
POOLED = "all"

# Each dataset's tests, in panel order: the three kinds of held-out trial, then their
# average; the knowledge regimes from least to most knowledge, then all of them together.
TESTS: Dict[str, List[str]] = {
    "l1": [*NOVELTY_BY_CELL.values(), AVERAGE_NOVELTY],
    "knowqa": [*D.REGIME_ORDER, POOLED],
}

# The opening-scan features, each worked out from several of a person's trials.
PERSON_LEVEL_COLS = frozenset(
    FS.PATTERN_COLS_WITH_INTERACTIONS
    + FS.PATTERN_DISTANCE_COLS
    + [Con.N_STRATEGY_TRIALS_WITH_Q, Con.N_STRATEGY_TRIALS_NO_Q]
)

SCORE_COLS = [
    "dataset", "test", "model", "n_features", "balanced_accuracy",
    "ci_low", "ci_high", "n_folds", "n_scored", "note",
]


def _is_paragraph_column(col: str) -> bool:
    """Whether `col` was measured on the paragraph screen.

    Every such column's name ends in one of that screen's areas: `outside`,
    `distractor` or `critical`.
    """
    return any(col.endswith(f"_{area}") for area in FS.PARAGRAPH_SCREEN.regions)


def _why_not_on_knowqa(name: str, cols: Sequence[str], knowqa_df: pd.DataFrame) -> Optional[str]:
    """Why model `name` cannot be scored on KnowQA, or None if it can.

    Stops if a column is missing for any reason other than KnowQA having no
    paragraph screen.
    """
    missing = [c for c in cols if c not in knowqa_df.columns]
    if not missing:
        return None
    unexplained = [c for c in missing if not _is_paragraph_column(c)]
    if unexplained or dataset("knowqa").has_paragraph:
        raise ValueError(
            f"{name}: KnowQA's trial table has no {unexplained or missing}. Only "
            f"paragraph features are expected to be missing there, because KnowQA "
            f"has no paragraph screen."
        )
    return "not tested: uses paragraph features, and KnowQA has no paragraph screen"


def _knowqa_row(test: str, name: str, cols: Sequence[str], **values: Any) -> Dict[str, Any]:
    """One KnowQA row of `scores`. Without `values` it is a row with no score."""
    row = {
        "dataset": "knowqa", "test": test, "model": name, "n_features": len(cols),
        "balanced_accuracy": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "n_folds": np.nan, "n_scored": np.nan, "note": "",
    }
    row.update(values)
    return row


def run_model_comparison(
    word_df: pd.DataFrame,
    models: Mapping[str, Sequence[str]],
    *,
    trial_df: Optional[pd.DataFrame] = None,
    knowqa_df: Optional[pd.DataFrame] = None,
    n_folds: int = 10,
    ci: float = 0.95,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Score every model in `models` on Study 1 and on Study 2.

    `models` maps a name to its feature columns, e.g.
    `modeling/feature_sets.py::load_feature_sets()`. `word_df` is Study 1's word-level
    table (`features/build.py::load_model_input`); only its paragraph ids are used, to
    match trials to the fold files. `trial_df` is Study 1's trial table, and `knowqa_df`
    Study 2's with the regime attached (`descriptives.load_showcase_frame`). Both are
    loaded if not given.

    Returns two tables:

    * `scores` -- one row per dataset, test and model: the balanced accuracy, its
      interval (Study 1 only), and `n_scored`, how many trials were scored. For Study 1
      that is summed over the folds, so a trial held out in several folds counts each
      time. `note` says why a model has no score.
    * `folds` -- Study 1's cross-validation rows, one per model, fold and held-out half:
      the scores the Study 1 rows are built from.

    Study 1 takes about seven seconds per model, Study 2 about one.
    """
    names = list(models)
    for name in names:
        person_level = sorted(set(models[name]) & PERSON_LEVEL_COLS)
        if person_level:
            raise ValueError(
                f"{name} uses features that summarise a person over several trials: "
                f"{person_level}. Here the folds are cut from the ready-made trial "
                f"table, where those include held-out trials (`pitfalls.md` §3). "
                f"Cross-validate it with `modeling/crossval.py` and "
                f"rebuild_features=True instead."
            )

    if trial_df is None:
        trial_df = load_model_ready()

    # --- Study 1 ------------------------------------------------------------------
    # The fold files name paragraphs by `text_id`, which the trial table does not
    # carry; the word-level table does. One row per trial is all the matching needs.
    trial_ids = word_df[[Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.TEXT_ID_COLUMN]].drop_duplicates()
    assert len(trial_ids) == len(trial_df), "expected exactly one paragraph per trial"

    if verbose:
        print(f"Study 1: cross-validating {len(names)} model(s) over {n_folds} folds")
    cv_out = run_cross_validation_on_combined_folds(
        trial_ids,
        fold_dirs=[HUNTERS_REFOLDED_DIR, GATHERERS_REFOLDED_DIR],  # both groups, pooled
        model_builders={name: TrialLevelLogRegModel for name in names},
        feature_cols_by_model={name: list(models[name]) for name in names},
        target_col=Con.IS_CORRECT_COLUMN,
        n_folds=n_folds,
        trial_df=trial_df,
    )
    l1 = summarize_cv_results_by_novelty(cv_out, ci=ci)
    l1_scores = pd.DataFrame({
        "dataset": "l1",
        "test": l1["novelty"],
        "model": l1["model"],
        "n_features": l1["model"].map(lambda m: len(models[m])),
        "balanced_accuracy": l1["mean_metric"],
        "ci_low": l1["ci_low"],
        "ci_high": l1["ci_high"],
        "n_folds": l1["n_folds"],
        "n_scored": l1["total_n_eval"],
        "note": "",
    })

    # --- Study 2 ------------------------------------------------------------------
    if knowqa_df is None:
        knowqa_df = D.load_showcase_frame(verbose=verbose)

    knowqa_rows: List[Dict[str, Any]] = []
    for name in names:
        cols = list(models[name])
        reason = _why_not_on_knowqa(name, cols, knowqa_df)
        if reason is not None:
            if verbose:
                print(f"Study 2: {name} {reason}")
            knowqa_rows += [_knowqa_row(test, name, cols, note=reason) for test in TESTS["knowqa"]]
            continue

        if verbose:
            print(f"Study 2: {name}")
        model = D.fit_l1_model(trial_df, feature_cols=cols, verbose=False)
        scored = D.attach_predictions(knowqa_df, model, feature_cols=cols, verbose=verbose)
        metrics = D.prediction_metrics_by_regime(scored).set_index(D.REGIME_COL)

        # A regime with another label would be scored but never shown.
        if set(metrics.index) != set(TESTS["knowqa"]):
            raise ValueError(
                f"KnowQA's regimes are {sorted(metrics.index)}; this comparison "
                f"expects {TESTS['knowqa']}."
            )
        for test in TESTS["knowqa"]:
            knowqa_rows.append(_knowqa_row(
                test, name, cols,
                balanced_accuracy=metrics.loc[test, "balanced_accuracy"],
                n_scored=metrics.loc[test, "n"],
            ))

    scores = pd.concat([l1_scores, pd.DataFrame(knowqa_rows)], ignore_index=True)
    return {"scores": scores[SCORE_COLS], "folds": cv_out.summary_df}
