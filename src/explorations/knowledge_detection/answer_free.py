"""Does the eye tracker beat response time when it does not know the answer key?

`run.py`'s gaze features know which answer is correct: many compare it with the wrong ones
(`__correct`, `__contrast`, ...), and the per-answer columns name each answer by letter, with
`answer_A` always the correct one. So a gaze model can read off whether the answer was right,
and whether it was right separates the regimes well on its own. Here the gaze features are
rebuilt without the key (`add_answer_free_columns`) and set against response time, with and
without correctness.

**Without the key:**

* each per-answer measure becomes its largest, average and smallest value over the four
  answers (`<measure>__answer_max`, `__answer_mean`, `__answer_min`, the spread naming of
  `explorations/answer_hidden`): how attention was spread, with no answer singled out.
  Answers with no value are skipped; a trial where none of the four has one has no value,
  and the model fills it with 0 as it fills every missing pupil and first-fixation value;
* the comparisons with the correct answer are dropped, and so is which answer was looked at
  last (`last_before_*_answer_*`), which names the answer by its letter;
* the question area and the trial-wide measures (`seq_len`, `has_xyx`, ...) are kept as they
  are, since they name no answer.

The response-time measures (`run.RESPONSE_TIME_COLS`) are kept apart, so the eye tracker can
be scored on its own. Its own time measures -- dwell on the question, the mean dwell over the
answers -- stay in it: they are what an eye tracker measures.

**Two models, each regime with knowledge against none:** partial knowledge against none,
with full knowledge left out, and full against none, with partial left out. Each is trained
and scored on held-out participants as in `run.py`. Both are run here whatever
`run.WITH_KNOWLEDGE` says.

**Scores and tests** as `run.score_tables`: the within-person AUC, and differences between
feature sets on the same resampled participants (`PAIRS`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.modeling.provenance import model_folder
from src.lib.plotting.output import save_output
from src.explorations.knowledge_detection.run import (
    ANALYSIS,
    NO_KNOWLEDGE,
    REGIME_COL,
    REGIME_COLOURS,
    RESPONSE_TIME_COLS,
    TARGET,
    feature_sets,
    load_trials,
    lopo_predictions,
    score_tables,
)

KNOWLEDGE_REGIMES = ("partial knowledge", "full knowledge")

# The column endings that compare the correct answer with the wrong ones.
KEY_SUFFIXES = (
    Con.CORRECT_SUFFIX, Con.WRONG_MEAN_SUFFIX, Con.CONTRAST_SUFFIX,
    Con.DISTANCE_FURTHEST_SUFFIX, Con.DISTANCE_CLOSEST_SUFFIX,
)
SPREAD = ("answer_max", "answer_mean", "answer_min")
LAST_LOOK_PREFIX = "last_before_"

SET_LABELS = {
    "response_time": "response time alone",
    "response_time+correctness": "response time + correctness",
    "gaze_no_key": "eye tracker, no answer key",
    "gaze_no_key+time": "eye tracker, no key + response time",
    "gaze_no_key+time+correctness": "eye tracker, no key + response time + correctness",
    "all_features": "all features (eye tracker with the key + time)",
}

# The feature-set differences tested: (set, set it is compared with, what it asks).
PAIRS: List[Tuple[str, str, str]] = [
    ("gaze_no_key", "response_time", "the eye tracker alone, without the key, against time alone"),
    ("gaze_no_key+time", "response_time", "what the eye tracker without the key adds to time"),
    ("gaze_no_key+time+correctness", "response_time+correctness",
     "what the eye tracker without the key adds to time and correctness"),
    ("all_features", "gaze_no_key+time", "what knowing the answer key adds to the eye tracker and time"),
]


def add_answer_free_columns(trials: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """`trials` with the spread columns added, and a table saying what became of each column
    of `cols` (`run.feature_sets`'s `all_features`): kept, spread over the answers, dropped
    for using the key, dropped for naming an answer by letter, or response time. Stops
    unless every per-answer measure has all four answers."""
    letters = list(Con.ANSWER_AREAS)
    fate, by_answer = {}, {}
    for c in cols:
        letter = next((a for a in letters if c.endswith(a)), None)
        if c in RESPONSE_TIME_COLS:
            fate[c] = "response time"
        elif c.endswith(KEY_SUFFIXES):
            fate[c] = "dropped: uses the key"
        elif letter and c.startswith(LAST_LOOK_PREFIX):
            fate[c] = "dropped: names an answer"
        elif letter:
            fate[c] = "spread over the answers"
            by_answer.setdefault(c[: -len(letter)], []).append(c)
        else:
            fate[c] = "kept"
    spread = {}
    for base, group in by_answer.items():
        assert sorted(group) == sorted(f"{base}{a}" for a in letters), f"{base}: not one column per answer: {group}"
        values = trials[[f"{base}{a}" for a in letters]].apply(pd.to_numeric, errors="raise")
        spread[f"{base}answer_max"] = values.max(axis=1)
        spread[f"{base}answer_mean"] = values.mean(axis=1)
        spread[f"{base}answer_min"] = values.min(axis=1)
    out = pd.concat([trials, pd.DataFrame(spread, index=trials.index)], axis=1)
    table = pd.DataFrame({"column": list(fate), "fate": list(fate.values())})
    return out, table


def answer_free_sets(trials: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    """The trials with the spread columns, the feature sets of `SET_LABELS`, and the table of
    what became of each column."""
    everything = feature_sets(trials)["all_features"]
    frame, fate = add_answer_free_columns(trials, everything)
    kept = fate.loc[fate["fate"] == "kept", "column"].tolist()
    spread_cols = [c for c in frame.columns if c.endswith(SPREAD) and c not in trials.columns]
    gaze = kept + spread_cols
    time = list(RESPONSE_TIME_COLS)
    correct = [Con.IS_CORRECT_COLUMN]
    sets = {
        "response_time": time,
        "response_time+correctness": time + correct,
        "gaze_no_key": gaze,
        "gaze_no_key+time": gaze + time,
        "gaze_no_key+time+correctness": gaze + time + correct,
        "all_features": everything,
    }
    return frame, sets, fate


def _short(regime: str) -> str:
    return regime.split()[0]


def predictions(frame: pd.DataFrame, sets: Dict[str, List[str]]) -> pd.DataFrame:
    """For each regime of `KNOWLEDGE_REGIMES`, every set's held-out predictions of that
    regime against none, the other knowledge regime left out. `knowledge_regime` says which
    model a row comes from."""
    parts = []
    for regime in KNOWLEDGE_REGIMES:
        trials = frame[frame[REGIME_COL].isin([NO_KNOWLEDGE, regime])].copy()
        trials[TARGET] = (trials[REGIME_COL] == regime).astype(int)
        parts.append(lopo_predictions(trials, sets).assign(knowledge_regime=regime))
    return pd.concat(parts, ignore_index=True)


def score(preds: pd.DataFrame, pairs: Sequence[Tuple[str, str, str]] = PAIRS) -> Dict[str, pd.DataFrame]:
    """`run.score_tables` on each regime's model, stacked: `scores` and `pairs`."""
    parts = []
    for regime in KNOWLEDGE_REGIMES:
        name = f"{_short(regime)} vs none"
        parts.append(score_tables(
            preds[preds["knowledge_regime"] == regime], pairs=pairs,
            comparisons={name: ((regime,), (NO_KNOWLEDGE,))}, trained=name,
        ))
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in ("scores", "pairs")}


def plot_answer_free(
    scores: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Each feature set's within-person AUC on held-out participants, for partial knowledge
    against none and full against none, with 95% intervals. Saved with `tables`. Returns
    the figure."""
    sets = [s for s in SET_LABELS if s in set(scores["feature_set"])]
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(sets) + 1.8), layout="constrained")
    style = {f"{_short(r)} vs none": (REGIME_COLOURS[r], m, dy)
             for r, m, dy in zip(KNOWLEDGE_REGIMES, ("s", "D"), (-0.13, 0.13))}
    for comp, (colour, marker, dy) in style.items():
        sub = scores[scores["comparison"] == comp].set_index("feature_set")
        for i, s in enumerate(sets):
            r = sub.loc[s]
            ax.errorbar(r["person_auc"], i + dy,
                        xerr=[[r["person_auc"] - r["person_ci_low"]], [r["person_ci_high"] - r["person_auc"]]],
                        fmt=marker, color=colour, ms=6, elinewidth=1.4, capsize=0, label=comp if i == 0 else None)
            ax.annotate(f"{r['person_auc']:.3f}", (r["person_ci_high"], i + dy), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0.5, color="#555555", lw=1, ls="--")
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([SET_LABELS[s] for s in sets], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("AUC within each held-out participant, averaged (0.5 = chance)")
    ax.grid(axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Study 2: each knowledge regime against none, with and without the answer key", fontsize=10)
    save_output(fig, analysis=ANALYSIS, plot="answer_free_gaze", tables=tables,
                save=save, close=close, subdir=model_folder(scores))
    return fig


def main(save: bool = True, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Both models, scored and tested, saved as a figure with its tables. Returns the
    tables and the predictions."""
    frame, sets, fate = answer_free_sets(load_trials(verbose=verbose))
    if verbose:
        print(fate["fate"].value_counts().to_string())
        print({s: len(c) for s, c in sets.items()})
    preds = predictions(frame, sets)
    out = score(preds)
    out["columns"] = fate
    out["feature_sets"] = pd.DataFrame(
        [{"feature_set": s, "n_features": len(c), "features": ", ".join(c)} for s, c in sets.items()]
    )
    plot_answer_free(out["scores"], dict(out), save=save, close=True)
    if verbose:
        print(out["scores"][["feature_set", "comparison", "person_auc", "person_ci_low", "person_ci_high",
                             "pooled_auc", "balanced_accuracy"]].round(3).to_string(index=False))
        print(out["pairs"][["question", "comparison", "person_auc_difference", "person_ci_low",
                            "person_ci_high", "person_p", "pooled_p"]].round(4).to_string(index=False))
    out["predictions"] = preds
    return out


if __name__ == "__main__":
    main()
