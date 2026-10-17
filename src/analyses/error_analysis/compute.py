"""Which errors the correctness model makes, and whether guessing explains them.

The model predicts whether a participant picked the correct answer. It gets two
kinds of thing wrong, in very unequal numbers: false negatives (the person was
right, the model said wrong) outnumber false positives by about eight to one.
Nothing in the training favours one kind of error over the other, so that comes
from the data.

One explanation is that the label is wrong. If what the model reads is how sure
someone was rather than whether they were right, then a person who did not know
the answer but picked it anyway is still scored "correct", and a model that
called them wrong is counted as making an error when it arguably read them
right. Some share of the false negatives are lucky guesses the model saw through.

Estimate how lucky guessing could affect the FN/FP imbalance.

Assume all human-incorrect responses (TN + FP) come from non-knowers.
Given a probability p of guessing correctly, estimate the number of
lucky guesses among human-correct responses as:

    n_lucky = (TN + FP) * p / (1 - p)

These lucky guesses are currently classified as TP or FN, but would
be considered non-knowers if evaluating underlying knowledge rather
than observed answer correctness. Reclassification therefore moves
TP -> FP and FN -> TN.

Since the actual distribution of lucky guesses is unknown, consider
two scenarios:

- proportional: Lucky guesses are distributed proportionally across
  TP and FN, assuming no association between guessing and the model's
  predictions. The fraction moved from FN is FN / (TP + FN).

- excuse_false_negatives: All lucky guesses are assumed to come from
  FN (up to the available FN count), representing perfect identification
  of lucky guessers by the model. No additional FP are created.

The adjusted FN/FP ratio depends on both p and the assumed distribution.
The second scenario always minimizes total reclassified errors, but
does not necessarily minimize the FN/FP ratio.

These are hypothetical sensitivity analyses, not observed corrections.
They assume a binary knowledge state and that all incorrect answers
reflect lack of knowledge. The scenarios bound the ratio only if lucky
guessers are at least as likely to be predicted incorrect as other
human-correct responses.
"""



from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.modeling.confusion import (
    QUAD_LABELS,
    QUAD_ORDER,
    confusion_counts,
    counts_to_frame,
    label_quadrants,
)

# The three held-out regimes the paper reports. `train_train` is the training
# set, and the three `val_*` regimes are a second test set whose trials would
# count the same reader twice here, so none of the four belong.
TEST_REGIMES: List[str] = [
    "test_seen_subject_unseen_item",
    "test_unseen_subject_seen_item",
    "test_unseen_subject_unseen_item",
]

REGIME_TITLES: Dict[str, str] = {
    "test_seen_subject_unseen_item": "Seen subject, new item",
    "test_unseen_subject_seen_item": "New subject, seen item",
    "test_unseen_subject_unseen_item": 'New subject, new item ("both")',
}

# The guessing chances to try, and how to write each one.
GUESS_RATES: List[Tuple[str, float]] = [("1/2", 0.5), ("1/3", 1 / 3), ("1/4", 0.25)]

GUESS_MODES: List[str] = ["excuse_false_negatives", "proportional"]

MODE_LABELS: Dict[str, str] = {
    "excuse_false_negatives": "model spots guessers  (FN->TN)",
    "proportional": "model can't tell  (FN->TN, TP->FP)",
}


# The ten attention features of the headline model plus the two last-fixation
# ones: the same twelve the per-person models use, so the two sets of mistakes
# can be read against each other.
DEFAULT_MODEL_NAME = "select_1_plus_last_confirm"


def default_feature_cols() -> List[str]:
    """The twelve feature columns, looked up when called rather than at import."""
    import src.modeling.feature_sets as FG

    return list(FG.SELECT_1_COLS) + list(FG.LAST_CONFIRM_COMPACT)


def run_crossval_predictions(
    all_participants: pd.DataFrame,
    *,
    trial_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    n_folds: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cross-validate the general model and return its held-out predictions.

    One logistic regression trained across everyone, on the pooled hunters and
    gatherers folds. Same folds and same features as the headline model, so these
    are the headline model's mistakes.

    Gives one row per held-out trial, fold and regime, with `y_true`, `y_pred`
    and `y_prob`.

    `trial_df` is the ready-made trial table, loaded if not given. Passing it is
    the quick path, and fine here: all twelve features are measured per trial,
    none of them summarise a person across trials, so there is nothing for the
    per-regime rebuild to guard against (`pitfalls.md` §3).
    """
    from src.config.datasets import GATHERERS_REFOLDED_DIR, HUNTERS_REFOLDED_DIR
    from src.config import columns as Con
    from src.features.build import load_model_ready
    from src.modeling.crossval import (
        crossval_predictions_to_df,
        run_cross_validation_on_combined_folds,
    )
    from src.modeling.models.logreg_model import TrialLevelLogRegModel

    cols = list(feature_cols) if feature_cols is not None else default_feature_cols()
    if trial_df is None:
        trial_df = load_model_ready()

    if verbose:
        print(f"cross-validating {model_name} on {len(cols)} features, {n_folds} folds")

    cv_out = run_cross_validation_on_combined_folds(
        df=all_participants,
        fold_dirs=[HUNTERS_REFOLDED_DIR, GATHERERS_REFOLDED_DIR],
        model_builders={model_name: (lambda: TrialLevelLogRegModel())},
        target_col=Con.IS_CORRECT_COLUMN,
        n_folds=n_folds,
        df_participant_col=Con.PARTICIPANT_ID,
        df_text_col=Con.TEXT_ID_COLUMN,
        feature_cols_by_model={model_name: cols},
        trial_df=trial_df,
        keep_cols=None,
    )
    preds = crossval_predictions_to_df(cv_out, model_name=model_name)
    if verbose:
        print(f"{len(preds)} held-out predictions across "
              f"{preds['regime'].nunique()} regimes")
    return preds


def build_trial_prediction_table(preds: pd.DataFrame, regime: str) -> pd.DataFrame:
    """One regime's held-out trials, each labelled with its quadrant.

    `preds` comes from `run_crossval_predictions`. Trials from different folds
    are pooled, which is safe: each trial is held out in exactly one fold.

    An unknown regime stops the run. Returning nothing instead would make every
    count below zero, which reads as an answer rather than a mistake.
    """
    available = sorted(preds["regime"].unique())
    if regime not in available:
        raise ValueError(
            f"No held-out predictions for regime {regime!r}. Available: {available}"
        )
    trials = preds[preds["regime"] == regime].reset_index(drop=True)
    return label_quadrants(trials)


def confusion_summary(
    trials: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.Series:
    """Count the pooled trials in each quadrant, printing the shares if asked.

    The printed line splitting the errors into false positives and false
    negatives is the thing this whole analysis is about.
    """
    counts = confusion_counts(trials)
    if verbose:
        n = len(trials)
        print(f"{n} pooled held-out trials")
        for q in QUAD_ORDER:
            print(f"  {QUAD_LABELS[q]:22s} {counts[q]:6d}  ({counts[q] / n:6.1%})")
        n_err = int(counts["FP"] + counts["FN"])
        if n_err:
            print(
                f"  Errors: {n_err} total  |  FP {counts['FP'] / n_err:.1%}"
                f"  |  FN {counts['FN'] / n_err:.1%}"
            )
        else:
            print("  Errors: none -- the model got every pooled trial right.")
    return counts


def fn_fp_ratio(fp: float, fn: float) -> float:
    """False negatives per false positive, or `inf` if there are no false positives.

    `inf` is the right answer there -- the imbalance is total, not undefined --
    but it cannot be plotted or averaged, so callers check for it.
    """
    return float("inf") if fp == 0 else fn / fp


def guess_adjust(
    counts: pd.Series,
    p: float,
    mode: str = "proportional",
) -> Dict[str, float]:
    """Recount FP and FN assuming some correct answers were lucky guesses.

    `p` is the chance a non-knower guesses correctly. The module docstring has
    the assumptions behind it and what the two modes do. Gives back the new `FP`
    and `FN`, the estimated lucky guesses `n_lucky`, how many of them there was
    room to move (`n_lucky_applied`), and whether it ran out (`clamped`).

    Careful with `clamped`. `excuse_false_negatives` moves lucky guesses out of
    `FN` only, so it cannot move more than there are; `proportional` cannot move
    more than there are correct answers. Anything left over has nowhere to go, so
    it is reported rather than quietly dropped -- running out means that at this
    `p` the guessing story needs more lucky guesses than the data holds, which
    says `p` is too high.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be a probability strictly between 0 and 1; got {p}")
    if mode not in GUESS_MODES:
        raise ValueError(f"mode must be one of {GUESS_MODES}; got {mode!r}")

    TP, TN, FP, FN = (int(counts[q]) for q in QUAD_ORDER)

    # Non-knowers who guessed right, hidden among the human-correct trials.
    n_lucky = (TN + FP) * p / (1 - p)

    if mode == "excuse_false_negatives":
        # Every lucky guess was one the model called "wrong": FN -> TN.
        applied = min(n_lucky, FN)
        out = {"FP": float(FP), "FN": float(FN - applied)}
    else:
        # The model cannot tell, so the lucky guesses split TP:FN by size.
        human_correct = TP + FN
        applied = min(n_lucky, human_correct)
        if human_correct == 0:
            out = {"FP": float(FP), "FN": float(FN)}
        else:
            out = {
                "FP": FP + applied * TP / human_correct,
                "FN": FN - applied * FN / human_correct,
            }

    out["n_lucky"] = n_lucky
    out["n_lucky_applied"] = float(applied)
    out["clamped"] = bool(n_lucky > applied + 1e-9)
    return out


def guessing_scenarios(
    counts: pd.Series,
    *,
    rates: Sequence[Tuple[str, float]] = tuple(GUESS_RATES),
    modes: Sequence[str] = tuple(GUESS_MODES),
    regime: Optional[str] = None,
) -> pd.DataFrame:
    """One row per guess rate and mode, plus a `raw` row, each with its FN/FP ratio.

    The `raw` row holds the counts as observed, so the table can be read on its
    own -- the adjusted ratios only mean something next to it.
    """
    rows = [
        {
            "scenario": "raw",
            "mode": "-",
            "p": np.nan,
            "FP": float(counts["FP"]),
            "FN": float(counts["FN"]),
            "FN/FP": fn_fp_ratio(float(counts["FP"]), float(counts["FN"])),
            "n_lucky": 0.0,
            "n_lucky_applied": 0.0,
            "clamped": False,
        }
    ]
    for name, p in rates:
        for mode in modes:
            a = guess_adjust(counts, p, mode)
            rows.append(
                {
                    "scenario": f"guess {name}",
                    "mode": mode,
                    "p": p,
                    "FP": a["FP"],
                    "FN": a["FN"],
                    "FN/FP": fn_fp_ratio(a["FP"], a["FN"]),
                    "n_lucky": a["n_lucky"],
                    "n_lucky_applied": a["n_lucky_applied"],
                    "clamped": a["clamped"],
                }
            )
    out = pd.DataFrame(rows)
    # The trial count the scenario was built from. Rides along because the
    # regimes differ several-fold in size and a ratio alone does not say so.
    out["n_trials"] = int(counts.sum())
    if regime is not None:
        out.insert(0, "regime", regime)
    return out


def quadrants_by_regime(
    preds: pd.DataFrame,
    *,
    regimes: Sequence[str] = tuple(TEST_REGIMES),
) -> pd.DataFrame:
    """Quadrant counts and shares for every regime, in one table.

    One table rather than one per regime, so comparing them is a groupby instead
    of a glob (`conventions.md`, "a sweep writes one table").
    """
    frames = []
    for regime in regimes:
        trials = build_trial_prediction_table(preds, regime)
        frame = counts_to_frame(confusion_counts(trials))
        frame.insert(0, "regime", regime)
        frame.insert(1, "regime_title", REGIME_TITLES.get(regime, regime))
        frame["n_regime_trials"] = len(trials)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def guessing_scenarios_by_regime(
    preds: pd.DataFrame,
    *,
    regimes: Sequence[str] = tuple(TEST_REGIMES),
    rates: Sequence[Tuple[str, float]] = tuple(GUESS_RATES),
    modes: Sequence[str] = tuple(GUESS_MODES),
) -> pd.DataFrame:
    """`guessing_scenarios` for every regime, stacked into one table."""
    frames = []
    for regime in regimes:
        counts = confusion_counts(build_trial_prediction_table(preds, regime))
        frames.append(
            guessing_scenarios(counts, rates=rates, modes=modes, regime=regime)
        )
    return pd.concat(frames, ignore_index=True)


def balance_point(counts: pd.Series) -> Dict[str, float]:
    """The guessing chance at which false negatives and false positives come out equal.

    Every lucky guess moves one trial out of the human-correct row. If the model
    had called it wrong it goes FN -> TN, so FN drops by one; if the model had
    called it right it goes TP -> FP, so FP rises by one. Either way the gap
    `FN - FP` shrinks by exactly one, so

        FN' - FP' = (FN - FP) - n_lucky

    and how the guesses were split does not come into it. The errors balance when
    `n_lucky` reaches the gap, which happens at

        p* = gap / (gap + n_wrong)

    That matters more than it looks: the split is the part we cannot observe, and
    it has no bearing on where the ratio crosses 1. It only changes how far the
    ratio moves.

    `p_star` is NaN when the gap is zero or negative -- there are already at
    least as many false positives as false negatives, so no amount of guessing
    brings them together.
    """
    TP, TN, FP, FN = (int(counts[q]) for q in QUAD_ORDER)
    gap = FN - FP
    n_wrong = TN + FP
    p_star = gap / (gap + n_wrong) if gap > 0 and (gap + n_wrong) > 0 else np.nan
    return {
        "gap_fn_minus_fp": float(gap),
        "n_wrong": float(n_wrong),
        "n_lucky_needed": float(gap),
        "p_star": float(p_star),
    }


def ratio_curve(
    counts: pd.Series,
    *,
    p_grid: Optional[Sequence[float]] = None,
    modes: Sequence[str] = tuple(GUESS_MODES),
) -> pd.DataFrame:
    """FN, FP and their ratio across a range of guessing chances, for each mode.

    One row per `p` and mode. Built on `guess_adjust`, so it agrees with the
    scenario tables rather than recomputing the arithmetic a second way.

    The default grid stops short of 1: as `p` approaches it the estimated lucky
    guesses run away to infinity, and well before that there are more of them
    than there are trials to move.
    """
    if p_grid is None:
        p_grid = np.linspace(0.01, 0.60, 119)
    rows = []
    for p in p_grid:
        for mode in modes:
            a = guess_adjust(counts, float(p), mode)
            rows.append(
                {
                    "p": float(p),
                    "mode": mode,
                    "FP": a["FP"],
                    "FN": a["FN"],
                    "FN/FP": fn_fp_ratio(a["FP"], a["FN"]),
                    "clamped": a["clamped"],
                }
            )
    return pd.DataFrame(rows)
