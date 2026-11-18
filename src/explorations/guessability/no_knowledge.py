"""Study 2's no-knowledge regime without its most guessable questions.

Under no knowledge a correct answer is a guess. A question whose answer gives itself away is
answered right however the participant reads it, so its trials teach the model's test little
about reading. Here the L1-trained correctness model, the same model and predictions as the
Study 2 analysis (`knowledge_regimes/descriptives.py`), is scored on the no-knowledge trials
three ways: all of them, without the most guessable `TOP` of the questions, and on those
questions alone.

**How guessable a question is**: the share of right guesses pooled over the three sources in
`pooled.py`, every guess counting once. The pilot has by far the most guesses per question, so
it dominates the pool. By default every pilot question is pooled, however much it changed
before the experiment (`pooled.match_questions` records how much). `drop_edited_pilot` keeps
only the pilot's guesses on questions that stayed almost the same; a question left with no
guesses outside KnowQA then cannot be ranked (`unrankable`) and has to be taken out of the
test before ranking.

**The ranking is over the test's own questions**: those answered under no knowledge. `TOP` is
then the share of those questions left out, not of all questions. Ties at the cut all go in.

**The experiment's own answers are part of the pool.** That ranks questions partly on the
outcome the model is scored on. The part is small (at most two no-knowledge answers per
question, against dozens of guesses from the other two sources), and `flag_most_guessable` with
`share_col="share_external"` ranks without them, to check that it does not matter.

Imports from `analyses/correctness_prediction/knowledge_regimes/` so that the model, its
predictions and its figures are the Study 2 analysis's own; its figure functions are called
with `save=False`, and the figures saved here, under `reports/explorations/guessability/`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import KNOW_QA_IA_ANSWERS_PATH, dataset
from src.features.build import load_model_ready
from src.lib.plotting.output import save_output
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.explorations.guessability.data import CHANCE
from src.explorations.guessability.pooled import (
    ALMOST_THE_SAME, QUESTION_KEY, SOURCES, experiment_texts, guess_rows, item_table, matched_guesses,
    question_key,
)
import src.analyses.correctness_prediction.knowledge_regimes.descriptives as D

ANALYSIS = "explorations/guessability"
NO_KNOWLEDGE = "no knowledge"
TOP = 0.25
# The trial columns saved behind the run figures: who, which question, which run, the outcome
# and what the model said.
TRIAL_COLS = [Con.PARTICIPANT_ID, Con.TEXT_ID_WITH_Q_COLUMN, QUESTION_KEY, D.REGIME_COL,
              Con.IS_CORRECT_COLUMN, D.PROB_COL, D.PRED_COL, D.CONFIDENCE_COL, MODEL_FAMILY_COL]


def run_labels(top: float = TOP) -> Dict[str, str]:
    """The three runs' labels. The Study 2 figure functions order labels they do not know
    alphabetically, and these sort as all, left out, only."""
    return {"all": "all questions",
            "without": f"most guessable {top:.0%} left out",
            "only": f"most guessable {top:.0%} only"}


def knowqa_predictions(feature_cols: Sequence[str] = D.SHOWCASE_FEATURE_COLS,
                       verbose: bool = True) -> pd.DataFrame:
    """Every KnowQA trial with the L1 model's prediction, made as in the Study 2 analysis:
    fitted on every L1 trial, on `feature_cols`."""
    l1 = load_model_ready(dataset("l1").model_ready)
    df = D.load_showcase_frame(dataset("knowqa").model_ready, KNOW_QA_IA_ANSWERS_PATH, verbose=verbose)
    model = D.fit_l1_model(l1, feature_cols=feature_cols, verbose=verbose)
    df = D.attach_predictions(df, model, feature_cols=feature_cols, verbose=verbose)
    df[QUESTION_KEY] = question_key(df[Con.TEXT_ID_WITH_Q_COLUMN])
    return df


def all_guesses(knowqa: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every guess from the three sources, one row each, and the two guessing files' matching
    tables (one row per question of each file)."""
    texts = experiment_texts()
    rows, pairs = [], []
    for ds in ("guess", "pilot"):
        g, p = matched_guesses(ds, texts)
        rows.append(guess_rows(g, ds))
        pairs.append(p)
    nk = knowqa[knowqa[D.REGIME_COL] == NO_KNOWLEDGE]
    nk = nk.assign(participant=nk[Con.PARTICIPANT_ID],
                   difficulty=nk[Con.TEXT_ID_WITH_Q_COLUMN].str.split("_").str[2])
    rows.append(guess_rows(nk, "knowqa"))
    return pd.concat(rows, ignore_index=True), pd.concat(pairs, ignore_index=True)


def guessability(guesses: pd.DataFrame) -> pd.DataFrame:
    """`pooled.item_table`, plus `share_external`: the pooled share without KnowQA's own
    answers."""
    items = item_table(guesses)
    ext = guesses[guesses["dataset"] != "knowqa"].groupby(QUESTION_KEY)["is_correct"].mean()
    items["n_external"] = items["n_guesses"] - items["n_knowqa"]
    items["share_external"] = items[QUESTION_KEY].map(ext)
    return items


def drop_edited_pilot(guesses: pd.DataFrame, pairs: pd.DataFrame,
                      keep: Sequence[str] = ALMOST_THE_SAME) -> pd.DataFrame:
    """`guesses` without the pilot's guesses on questions whose pilot version differs from the
    experiment's by more than `keep` allows (`pooled.CHANGE_LEVELS`). The other sources are
    untouched: `guess.tsv` asks the experiment's own wording, and KnowQA is the experiment."""
    level = guesses[QUESTION_KEY].map(pairs.loc[pairs["dataset"] == "pilot"].set_index(QUESTION_KEY)["change"])
    pilot = guesses["dataset"].eq("pilot")
    assert level[pilot].notna().all(), "a pilot guess has no pilot match"
    return guesses[~pilot | level.isin(keep)].reset_index(drop=True)


def unrankable(items: pd.DataFrame, knowqa: pd.DataFrame) -> pd.Index:
    """The no-knowledge questions with no guesses outside KnowQA, which `flag_most_guessable`
    refuses to rank."""
    keys = knowqa.loc[knowqa[D.REGIME_COL] == NO_KNOWLEDGE, QUESTION_KEY].unique()
    sub = items[items[QUESTION_KEY].isin(keys)]
    return pd.Index(sub.loc[sub["n_external"] == 0, QUESTION_KEY])


def flag_most_guessable(items: pd.DataFrame, knowqa: pd.DataFrame, *, top: float = TOP,
                        share_col: str = "share_correct") -> Tuple[pd.DataFrame, float]:
    """The questions answered under no knowledge, with `most_guessable` on the top `top` of
    them by `share_col`, and the share at the cut.

    Stops if a question has no guesses outside KnowQA: it would be ranked on the experiment's
    own answers alone, the outcome being predicted. `unrankable` lists such questions, to be
    taken out of `knowqa` first, visibly."""
    keys = knowqa.loc[knowqa[D.REGIME_COL] == NO_KNOWLEDGE, QUESTION_KEY].unique()
    sub = items[items[QUESTION_KEY].isin(keys)].copy()
    assert len(sub) == len(keys), f"{len(keys) - len(sub)} no-knowledge question(s) have no guesses"
    only_own = int((sub["n_external"] == 0).sum())
    assert only_own == 0, f"{only_own} no-knowledge question(s) have no guesses outside KnowQA"
    cut = float(sub[share_col].quantile(1 - top))
    sub["most_guessable"] = sub[share_col] >= cut
    return sub.sort_values(share_col, ascending=False, ignore_index=True), cut


def run_frame(knowqa: pd.DataFrame, flagged: pd.DataFrame, top: float = TOP) -> pd.DataFrame:
    """The no-knowledge trials three times over, `regime` relabelled with the run they belong
    to (`run_labels`): all of them, those of the questions not flagged, those of the flagged."""
    nk = knowqa[knowqa[D.REGIME_COL] == NO_KNOWLEDGE]
    flag = nk[QUESTION_KEY].map(flagged.set_index(QUESTION_KEY)["most_guessable"])
    assert flag.notna().all(), "a no-knowledge trial's question is not in the flagged table"
    flag = flag.astype(bool)
    lab = run_labels(top)
    return pd.concat([nk.assign(**{D.REGIME_COL: lab["all"]}),
                      nk[~flag].assign(**{D.REGIME_COL: lab["without"]}),
                      nk[flag].assign(**{D.REGIME_COL: lab["only"]})], ignore_index=True)


def run_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run: trials, questions, the participants' accuracy, and the model's balanced
    accuracy, sensitivity, specificity, AUC (and the confidence rating's AUC) and mean
    probability on right and on wrong answers. The Study 2 tables' pooled `all` row is dropped:
    the runs repeat trials, so pooling them means nothing."""
    m = D.prediction_metrics_by_regime(runs)
    a = D.discrimination_comparison(runs)[[D.REGIME_COL, "auc_model", "auc_confidence"]]
    p = D.probability_by_regime(runs)[[D.REGIME_COL, "mean_prob", "mean_prob_correct", "mean_prob_wrong"]]
    out = m.merge(a, on=D.REGIME_COL).merge(p, on=D.REGIME_COL)
    out = out[out[D.REGIME_COL] != "all"].rename(columns={D.REGIME_COL: "run", "accuracy": "model_accuracy"})
    out.insert(2, "questions", out["run"].map(runs.groupby(D.REGIME_COL)[QUESTION_KEY].nunique()))
    out.insert(4, "participant_accuracy", out["n_correct"] / out["n"])
    cols = ["run", "n", "questions", "n_correct", "participant_accuracy", "model_accuracy",
            "balanced_accuracy", "sensitivity", "specificity", "auc_model", "auc_confidence",
            "mean_prob", "mean_prob_correct", "mean_prob_wrong", MODEL_FAMILY_COL]
    return out[cols].reset_index(drop=True)


def source_agreement(items: pd.DataFrame) -> pd.DataFrame:
    """For each pair of sources, over the questions both guessed: how many, and the Spearman
    correlation of their shares right. Low agreement says the pooled share is mostly one
    source's."""
    rows = []
    srcs = list(SOURCES)
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            both = items[(items[f"n_{a}"] > 0) & (items[f"n_{b}"] > 0)]
            rows.append({"source_a": a, "source_b": b, "questions": len(both),
                         "median_guesses_a": both[f"n_{a}"].median(), "median_guesses_b": both[f"n_{b}"].median(),
                         "spearman_r": both[f"share_{a}"].corr(both[f"share_{b}"], method="spearman")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

KEPT, LEFT_OUT = "#4c72b0", "#c44e52"
# What a pool other than the default (every guess from every source) leaves out, for the
# figure titles. The key is also the `pool` setting in the file names, so the two versions'
# figures sit side by side.
POOLS = {"final_wording": "the pilot's guesses on reworded and rewritten questions left out"}


def _title(text: str, pool: Optional[str]) -> str:
    return text if pool is None else f"{text}\n({POOLS[pool]})"


def plot_guessability(flagged: pd.DataFrame, cut: float, *, top: float = TOP,
                      share_col: str = "share_correct", bin_width: float = 0.05,
                      pool: Optional[str] = None,
                      save: Optional[bool] = None, close: Optional[bool] = None):
    """The no-knowledge questions by their pooled share of right guesses, the ones left out
    stacked on top in red; the dashed line is the cut, the dotted one chance."""
    edges = np.round(np.arange(0, 1 + bin_width / 2, bin_width), 10)
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    kept = flagged.loc[~flagged["most_guessable"], share_col]
    out = flagged.loc[flagged["most_guessable"], share_col]
    ax.hist([kept, out], bins=edges, stacked=True, color=[KEPT, LEFT_OUT], rwidth=0.92,
            label=[f"kept ({len(kept)})", f"left out, the most guessable {top:.0%} ({len(out)})"])
    ax.axvline(cut, color="k", ls="--", lw=1, label=f"cut: {cut:.2f}")
    ax.axvline(CHANCE, color="#7f7f7f", ls=":", lw=1, label="one in four")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the question's guesses that were right, all sources pooled")
    ax.set_ylabel("questions")
    ax.set_title(_title(f"The {len(flagged)} questions answered under no knowledge: how guessable they are\n"
                        f"median {int(flagged['n_guesses'].median())} guesses per question", pool), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    save_output(fig, analysis=ANALYSIS, plot="no_knowledge_questions_by_guessability",
                tables={"questions": flagged}, save=save, close=close, ranked_by=share_col, top=top,
                pool=pool)
    return fig


def _save_run_figure(fig, runs: pd.DataFrame, plot: str, tables: Dict[str, pd.DataFrame],
                     pool: Optional[str], save: Optional[bool], close: Optional[bool]):
    save_output(fig, analysis=ANALYSIS, plot=plot, subdir=model_folder(runs), tables=tables,
                save=save, close=close, pool=pool)
    return fig


def plot_probability_histograms(runs: pd.DataFrame, *, pool: Optional[str] = None,
                                save: Optional[bool] = None, close: Optional[bool] = None):
    """The Study 2 histograms of predicted P(correct), one panel per run, split into right
    and wrong answers."""
    fig = D.plot_probability_histograms(runs, save=False)
    fig.suptitle(_title("L1 model on KnowQA's no-knowledge trials: predicted P(correct) by run", pool),
                 fontsize=11)
    fig.tight_layout()
    return _save_run_figure(fig, runs, "no_knowledge_probability_histograms",
                            {"trials": runs[TRIAL_COLS]}, pool, save, close)


def plot_confusions(runs: pd.DataFrame, *, pool: Optional[str] = None,
                    save: Optional[bool] = None, close: Optional[bool] = None):
    """The Study 2 confusion matrices, one per run, shaded by the share of each actual outcome."""
    fig = D.plot_confusion_by_regime(runs, save=False)
    fig.suptitle(_title("L1 model on KnowQA's no-knowledge trials: confusion by run (row-normalised)", pool),
                 fontsize=11)
    fig.tight_layout()
    return _save_run_figure(fig, runs, "no_knowledge_confusion", {"metrics": run_metrics(runs)},
                            pool, save, close)


def plot_participant_heatmap(runs: pd.DataFrame, *, pool: Optional[str] = None,
                             save: Optional[bool] = None, close: Optional[bool] = None):
    """The Study 2 participant heatmap, with the three runs as its columns."""
    fig = D.plot_participant_regime_heatmap(runs, figsize=(15, 4.2), save=False)
    fig.suptitle(_title("Per participant, per run (no-knowledge trials)", pool), fontsize=11)
    fig.tight_layout()
    return _save_run_figure(fig, runs, "no_knowledge_participant_heatmap",
                            {"profile": D.participant_profile(runs)}, pool, save, close)
