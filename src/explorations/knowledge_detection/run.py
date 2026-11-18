"""Telling Study 2's knowledge regimes from no knowledge, from the answer screen alone.

**The label** is the regime: 1 for the regimes in `WITH_KNOWLEDGE`, 0 for no knowledge.
Right and wrong answers are labelled alike. `WITH_KNOWLEDGE` is full knowledge alone (the
correct answer's content, in other words, shown before the answer screen): partial
knowledge is left out for now, and its trials are dropped, and counted, before anything is
fitted. With both regimes, the outputs are saved as `knowledge_vs_none` rather than
`full_vs_none`, with the comparisons of each regime against none and of full against
partial. The model weighs the two classes equally (`class_weight="balanced"`), and the
scores are AUCs, which do not depend on the split.

**The model** is the paper's logistic regression (`TrialLevelLogRegModel`), with its
missing-value fill and standardisation, on several feature sets (`feature_sets`). Whether
the answer was right is itself a strong cue -- 37% correct without knowledge, 97.5% with
full knowledge -- so correctness alone, and response time alone, are scored as baselines
that need no eye tracker. The paper's twelve features already carry most of correctness:
one of them is whether the last answer looked at before confirming was the correct one.

**Leave one participant out.** Each of the 13 participants is predicted by a model trained
on the other 12, so the score is for new people. Every participant has every regime, and
which questions fall in which regime is counterbalanced across people, so a held-out
person's questions were seen in training, but under other regimes.

**Regimes are blocks within a session**, one block of each per session, in counterbalanced
order: `block_positions` counts how often each regime came first, second and third (12 to
15 of the 39 sessions each). So the model cannot tell the regime by session, and the
position in the session is close to balanced.

**Two AUCs.** The main one is *within person*: each held-out participant's own AUC, over
their own trials, averaged over the 13. It asks whether a person's trials with knowledge
look different from that same person's trials without. The *pooled* AUC, over every
held-out prediction at once, also asks that trials from different people be on one scale,
and each person was scored by a different fold's model. That costs the small models most:
with partial knowledge included, correctness alone had a pooled AUC of 0.73 against 0.76
within person, only because the 13 models' intercepts differ. So feature sets are compared within person; the pooled AUC is in
the table.

**The tests** resample participants (2,000 draws, the same draws for every feature set and
comparison): a 95% interval for each AUC and a two-sided p against 0.5, and the same for the
difference between two feature sets (`PAIRS`). The p cannot go below 1/2,000.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

from src.config import columns as Con
from src.features.build import PARAGRAPH_MODEL_COLS
import src.modeling.feature_sets as FS
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.lib.plotting.output import save_output
from src.analyses.correctness_prediction.comparison import PERSON_LEVEL_COLS
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    REGIME_COL,
    REGIME_ORDER,
    SESSION_COL,
    SHOWCASE_FEATURE_COLS,
    load_showcase_frame,
)

ANALYSIS = "explorations/knowledge_detection"

TARGET = "has_knowledge"
NO_KNOWLEDGE = "no knowledge"

# The regimes labelled as knowledge. Partial knowledge is left out for now; with both,
# set ("partial knowledge", "full knowledge").
WITH_KNOWLEDGE: Tuple[str, ...] = ("full knowledge",)
MODELLED = tuple(r for r in REGIME_ORDER if r == NO_KNOWLEDGE or r in WITH_KNOWLEDGE)
BOTH = len(WITH_KNOWLEDGE) == 2
PLOT = "knowledge_vs_none" if BOTH else f"{WITH_KNOWLEDGE[0].split()[0]}_vs_none"

# What can be measured without an eye tracker: the time to answer, raw and per word, and
# how many times an answer was selected.
RESPONSE_TIME_COLS = [Con.TOTAL_ANSWERING_RT, Con.TOTAL_ANSWERING_RT_NORMALIZED, Con.NUM_OF_SELECTS]

# Each comparison: (regimes scored as positive, regimes scored as negative). The first is
# the one the model is trained on. With both knowledge regimes, each is also scored against
# none, and "full vs partial" asks whether the model gives full knowledge the higher
# probability of the two.
TRAINED = "knowledge vs none" if BOTH else f"{WITH_KNOWLEDGE[0].split()[0]} vs none"
COMPARISONS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {TRAINED: (WITH_KNOWLEDGE, (NO_KNOWLEDGE,))}
if BOTH:
    COMPARISONS |= {
        "partial vs none": (("partial knowledge",), (NO_KNOWLEDGE,)),
        "full vs none": (("full knowledge",), (NO_KNOWLEDGE,)),
        "full vs partial": (("full knowledge",), ("partial knowledge",)),
    }

# The feature-set differences tested: (set, set it is compared with, what it asks).
PAIRS = [
    ("all_features+correctness", "response_time+correctness", "what gaze adds to response time and correctness"),
    ("all_features", "correctness", "all features, without correctness, against correctness alone"),
    ("paper_model", "correctness", "the paper's features against correctness alone"),
    ("all_features+correctness", "all_features", "what correctness adds to all features"),
    ("all_features_no_question", "all_features", "what is lost without the question area"),
]


def load_trials(verbose: bool = True) -> pd.DataFrame:
    """Study 2's trial table (`load_showcase_frame`), every regime, with the label `TARGET`
    (empty for a regime not modelled)."""
    # A copy, because the loaded frame is built column by column and is fragmented.
    trials = load_showcase_frame(verbose=verbose).copy()
    assert set(trials[REGIME_COL]) == set(REGIME_ORDER), f"unexpected regimes: {set(trials[REGIME_COL])}"
    assert not trials.duplicated([Con.PARTICIPANT_ID, Con.TRIAL_ID]).any(), "the table is not one row per trial"
    trials[TARGET] = trials[REGIME_COL].map({NO_KNOWLEDGE: 0, **{r: 1 for r in WITH_KNOWLEDGE}})
    return trials


def modelled_trials(trials: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """The trials of the regimes in `MODELLED`; the rest are dropped, and counted."""
    keep = trials[REGIME_COL].isin(MODELLED)
    if verbose:
        dropped = trials.loc[~keep, REGIME_COL].value_counts().to_dict()
        print(f"modelled: {list(MODELLED)}, {int(keep.sum())} trials; left out: {dropped}")
    out = trials[keep].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out


def feature_sets(trials: pd.DataFrame) -> Dict[str, List[str]]:
    """The feature sets, from fewest features to most.

    * `correctness` -- whether the answer was right, alone;
    * `response_time` -- `RESPONSE_TIME_COLS`, alone;
    * `response_time+correctness` -- both: everything that needs no eye tracker;
    * `paper_model` -- the paper's twelve features (`SHOWCASE_FEATURE_COLS`);
    * `all_features` -- every model feature of the trial table, less the opening-scan
      features (worked out over several of a person's trials) and the paragraph features
      (Study 2 has none), plus `RESPONSE_TIME_COLS`;
    * `all_features+correctness` -- that and correctness;
    * `all_features_no_question` -- `all_features` without the columns measured on the
      question area (`is_question_col`). The question is the strongest single cue
      (`feature_auc`), and whether that is knowledge or the order of screens depends on
      what each regime showed before the answer screen; this set shows how much is left
      without it. The total response time still includes the time on the question.
    """
    everything = [
        c for c in FS.get_full_feature_cols(trials)
        if c not in PERSON_LEVEL_COLS and c not in PARAGRAPH_MODEL_COLS
    ]
    gaze_and_time = list(dict.fromkeys(everything + RESPONSE_TIME_COLS))
    correct = [Con.IS_CORRECT_COLUMN]
    return {
        "correctness": correct,
        "response_time": list(RESPONSE_TIME_COLS),
        "response_time+correctness": [*RESPONSE_TIME_COLS, *correct],
        "paper_model": list(SHOWCASE_FEATURE_COLS),
        "all_features": gaze_and_time,
        "all_features+correctness": [*gaze_and_time, *correct],
        "all_features_no_question": [c for c in gaze_and_time if not is_question_col(c)],
    }


def is_question_col(col: str) -> bool:
    """True if `col` is measured on the question area: every such column's name ends in the
    area's name (`mean_dwell_time__question`, `RT_pure_question`,
    `last_before_confirm_question`)."""
    return col.endswith(Con.QUESTION_AREA)


def lopo_predictions(
    trials: pd.DataFrame,
    sets: Dict[str, Sequence[str]],
    *,
    model_builder=TrialLevelLogRegModel,
) -> pd.DataFrame:
    """Each trial's predicted probability of knowledge from every feature set, by a model
    trained on the other participants. One row per feature set and trial."""
    parts = []
    for person in sorted(trials[Con.PARTICIPANT_ID].unique()):
        held_out = trials[Con.PARTICIPANT_ID] == person
        train, test = trials[~held_out], trials[held_out]
        for name, cols in sets.items():
            model = model_builder()
            model.fit(train, target_col=TARGET, feature_cols=list(cols))
            parts.append(pd.DataFrame({
                "feature_set": name,
                Con.PARTICIPANT_ID: test[Con.PARTICIPANT_ID].to_numpy(),
                Con.TRIAL_ID: test[Con.TRIAL_ID].to_numpy(),
                REGIME_COL: test[REGIME_COL].to_numpy(),
                Con.IS_CORRECT_COLUMN: test[Con.IS_CORRECT_COLUMN].to_numpy(int),
                "y_true": test[TARGET].to_numpy(int),
                "y_pred": model.predict(test, feature_cols=list(cols)),
                "y_prob": model.predict_proba(test, feature_cols=list(cols)),
                MODEL_FAMILY_COL: model.family,
            }))
    return pd.concat(parts, ignore_index=True)


def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The AUC of scores `p` for labels `y` (bool), once per row of `w` (resamples x
    trials), each trial counted as many times as its weight; tied scores count a half."""
    values, group = np.unique(p, return_inverse=True)
    onehot = sparse.csr_matrix((np.ones(len(p)), (np.arange(len(p)), group)), shape=(len(p), len(values)))
    pos = np.asarray((onehot.T @ (w * y).T).T)
    neg = np.asarray((onehot.T @ (w * ~y).T).T)
    below = np.cumsum(neg, axis=1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def _p_two_sided(boot: np.ndarray, null: float) -> float:
    return float(np.clip(2 * min((boot <= null).mean(), (boot >= null).mean()), 1 / len(boot), 1))


def score_tables(
    preds: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[str, str, str]] = PAIRS,
    comparisons: Optional[Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]] = None,
    trained: Optional[str] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """`scores`: one row per feature set and comparison of `comparisons` (by default
    `COMPARISONS`) -- the within-person AUC (`person_auc`, the mean of each person's own)
    and the pooled AUC (`pooled_auc`), each with its interval and p against 0.5; the lowest
    and highest person's AUC and how many people are above 0.5; and for the comparison the
    model was trained on (`trained`, by default `TRAINED`) the balanced accuracy. `pairs`:
    one row per pair of `pairs` (by default `PAIRS`) and comparison -- the difference in
    each AUC, with its interval and p. See the module docstring."""
    comparisons = COMPARISONS if comparisons is None else comparisons
    trained = TRAINED if trained is None else trained
    people = np.sort(preds[Con.PARTICIPANT_ID].unique())
    draws = np.random.default_rng(seed).integers(0, len(people), size=(n_boot, len(people)))
    weight = np.stack([np.bincount(d, minlength=len(people)) for d in draws])
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    def interval(boot, null):
        return np.percentile(boot, lo_q), np.percentile(boot, hi_q), _p_two_sided(boot, null)

    rows, boots, points = [], {}, {}
    for name, sub in preds.groupby("feature_set", sort=False):
        person = pd.Index(people).get_indexer(sub[Con.PARTICIPANT_ID])
        for comp, (positive, negative) in comparisons.items():
            m = sub[REGIME_COL].isin(positive + negative).to_numpy()
            y = sub.loc[m, REGIME_COL].isin(positive).to_numpy()
            p = sub.loc[m, "y_prob"].to_numpy(float)
            who = person[m]
            # Each person's own AUC, in the order of `people`. Every person has every
            # regime, so both sides of every comparison.
            own = np.array([roc_auc_score(y[who == i], p[who == i]) for i in range(len(people))])
            point = {"person": own.mean(), "pooled": weighted_auc(y, p, np.ones((1, m.sum())))[0]}
            boot = {"person": weight @ own / len(people), "pooled": weighted_auc(y, p, weight[:, who])}
            boots[(name, comp)], points[(name, comp)] = boot, point
            row = {"feature_set": name, "comparison": comp}
            for kind in ("person", "pooled"):
                lo, hi, pv = interval(boot[kind], 0.5)
                row |= {f"{kind}_auc": point[kind], f"{kind}_ci_low": lo, f"{kind}_ci_high": hi,
                        f"{kind}_p_vs_chance": pv}
            row |= {
                "min_person_auc": own.min(), "max_person_auc": own.max(),
                "people_above_chance": int((own > 0.5).sum()), "n_people": len(people),
                "n_trials": int(m.sum()), "n_positive": int(y.sum()), "n_negative": int((~y).sum()),
            }
            if comp == trained:
                pred = sub.loc[m, "y_pred"].to_numpy(int)
                row["balanced_accuracy"] = ((pred[y] == 1).mean() + (pred[~y] == 0).mean()) / 2
            rows.append(row)
    scores = pd.DataFrame(rows)

    pair_rows = []
    for a, b, question in pairs:
        for comp in comparisons:
            row = {"feature_set": a, "compared_with": b, "question": question, "comparison": comp}
            for kind in ("person", "pooled"):
                lo, hi, pv = interval(boots[(a, comp)][kind] - boots[(b, comp)][kind], 0.0)
                row |= {f"{kind}_auc_difference": points[(a, comp)][kind] - points[(b, comp)][kind],
                        f"{kind}_ci_low": lo, f"{kind}_ci_high": hi, f"{kind}_p": pv}
            pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    for t in (scores, pairs):
        t[["n_boot", "ci", "seed"]] = n_boot, ci, seed
        t[MODEL_FAMILY_COL] = model_folder(preds)
    return {"scores": scores, "pairs": pairs}


def probability_table(preds: pd.DataFrame) -> pd.DataFrame:
    """The mean predicted probability of knowledge, per feature set, regime and outcome
    (right or wrong answer), with the number of trials."""
    out = (
        preds.groupby(["feature_set", REGIME_COL, Con.IS_CORRECT_COLUMN], sort=False)["y_prob"]
        .agg(n="count", mean_prob="mean", median_prob="median")
        .reset_index()
    )
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def confusion_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per feature set, regime and answer (right or wrong): how many trials the model labels
    as knowledge and as none (its own threshold, `y_pred`), and the share labelled
    knowledge."""
    rows = []
    for (s, regime, correct), g in preds.groupby(["feature_set", REGIME_COL, Con.IS_CORRECT_COLUMN], sort=False):
        k = int(g["y_pred"].sum())
        rows.append({
            "feature_set": s, REGIME_COL: regime, Con.IS_CORRECT_COLUMN: correct, "n": len(g),
            "predicted_none": len(g) - k, "predicted_knowledge": k, "share_predicted_knowledge": k / len(g),
        })
    out = pd.DataFrame(rows)
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def accuracy_table(confusion: pd.DataFrame) -> pd.DataFrame:
    """Per feature set, from `confusion_table`'s counts: the accuracy (the share of trials
    whose regime the model got right) and the balanced accuracy (the mean of that share among
    trials with knowledge and among trials without). The accuracy depends on the mix of the
    two; the balanced accuracy does not."""
    rows = []
    for s, g in confusion.groupby("feature_set", sort=False):
        knows = g[REGIME_COL].isin(WITH_KNOWLEDGE)
        hits_k, n_k = g.loc[knows, "predicted_knowledge"].sum(), g.loc[knows, "n"].sum()
        hits_none, n_none = g.loc[~knows, "predicted_none"].sum(), g.loc[~knows, "n"].sum()
        rows.append({
            "feature_set": s, "n_knowledge": int(n_k), "n_none": int(n_none),
            "accuracy": (hits_k + hits_none) / (n_k + n_none),
            "balanced_accuracy": (hits_k / n_k + hits_none / n_none) / 2,
        })
    out = pd.DataFrame(rows)
    out[MODEL_FAMILY_COL] = model_folder(confusion)
    return out


def histogram_table(preds: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Per feature set, regime and answer (right or wrong): how many trials fall in each of
    `bins` equal bins of the predicted probability of knowledge, and their share of the
    group. One row per bin."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for (s, regime, correct), g in preds.groupby(["feature_set", REGIME_COL, Con.IS_CORRECT_COLUMN], sort=False):
        counts, _ = np.histogram(g["y_prob"], bins=edges)
        assert counts.sum() == len(g), f"{s}: a probability outside [0, 1]"
        for lo, hi, n in zip(edges[:-1], edges[1:], counts):
            rows.append({
                "feature_set": s, REGIME_COL: regime, Con.IS_CORRECT_COLUMN: correct,
                "bin_low": lo, "bin_high": hi, "n_in_bin": int(n), "share": n / len(g), "n": len(g),
            })
    out = pd.DataFrame(rows)
    out[MODEL_FAMILY_COL] = model_folder(preds)
    return out


def feature_auc(trials: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Each feature on its own: its AUC for `TARGET`, over the trials where it
    has a value, with that number of trials. Above 0.5, higher values go with knowledge.
    Sorted by distance from 0.5."""
    rows = []
    y = trials[TARGET].to_numpy(int)
    for c in cols:
        x = pd.to_numeric(trials[c], errors="raise").to_numpy(float)
        ok = ~np.isnan(x)
        rows.append({"feature": c, "auc": roc_auc_score(y[ok], x[ok]), "n_trials": int(ok.sum())})
    out = pd.DataFrame(rows)
    return out.iloc[np.argsort(-np.abs(out["auc"].to_numpy() - 0.5), kind="stable")].reset_index(drop=True)


def block_positions(trials: pd.DataFrame) -> pd.DataFrame:
    """How many sessions had each regime as their first, second and third block. Stops
    unless every session is three blocks, one per regime."""
    ordered = trials[[SESSION_COL, Con.TRIAL_ID, REGIME_COL]].sort_values([SESSION_COL, Con.TRIAL_ID])
    new_block = ordered[REGIME_COL] != ordered.groupby(SESSION_COL)[REGIME_COL].shift()
    ordered = ordered.assign(block=new_block.groupby(ordered[SESSION_COL]).cumsum())
    blocks = ordered.drop_duplicates([SESSION_COL, "block"])
    per_session = blocks.groupby(SESSION_COL)[REGIME_COL].agg(list)
    assert per_session.map(lambda r: sorted(r) == sorted(REGIME_ORDER)).all(), (
        "a session is not one block of each regime"
    )
    out = pd.crosstab(blocks[REGIME_COL], blocks["block"]).reindex(REGIME_ORDER)
    out.columns = [f"block_{b}" for b in out.columns]
    return out.reset_index()


SET_LABELS = {
    "correctness": "correctness alone",
    "response_time": "response time alone",
    "response_time+correctness": "response time + correctness",
    "paper_model": "the paper's 12 features",
    "all_features": "all features",
    "all_features+correctness": "all features + correctness",
    "all_features_no_question": "all features, no question area",
}

COMPARISON_STYLE = {
    "knowledge vs none": ("#333333", "o", "knowledge vs none (trained)"),
    "partial vs none": ("#dd8452", "s", "partial vs none"),
    "full vs none": ("#4c72b0", "D", "full vs none"),
    "full vs partial": ("#8c8c8c", "^", "full vs partial"),
}

# The paper's regime colours (`descriptives`).
REGIME_COLOURS = {"no knowledge": "#c44e52", "partial knowledge": "#dd8452", "full knowledge": "#4c72b0"}


def plot_detection(
    scores: pd.DataFrame,
    probabilities: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Left: each feature set's within-person AUC on held-out people, for each comparison,
    with 95% intervals. Right: the mean predicted probability of knowledge in each regime, on right
    answers (filled) and wrong ones (hollow). Saved with `tables`. Returns the figure."""
    sets = [s for s in SET_LABELS if s in set(scores["feature_set"])]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.62 * len(sets) + 2.2), sharey=True,
                                  layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    shown = [c for c in COMPARISON_STYLE if c in COMPARISONS]
    offsets = dict(zip(shown, np.linspace(-0.27, 0.27, len(shown)) if len(shown) > 1 else [0.0]))
    for comp in shown:
        colour, marker, _ = COMPARISON_STYLE[comp]
        sub = scores[scores["comparison"] == comp].set_index("feature_set")
        for i, s in enumerate(sets):
            r = sub.loc[s]
            ax.errorbar(r["person_auc"], i + offsets[comp],
                        xerr=[[r["person_auc"] - r["person_ci_low"]], [r["person_ci_high"] - r["person_auc"]]],
                        fmt=marker, color=colour, ms=5.5, elinewidth=1.3, capsize=0)
    ax.axvline(0.5, color="#555555", lw=1, ls="--")
    ax.set_xlabel("AUC within each held-out participant, averaged (0.5 = chance)")
    ax.set_title("How well each feature set tells the regimes apart", fontsize=10)

    regime_offsets = dict(zip(MODELLED, np.linspace(-0.18, 0.18, len(MODELLED))))
    for i, s in enumerate(sets):
        sub = probabilities[probabilities["feature_set"] == s]
        for _, r in sub.iterrows():
            filled = r[Con.IS_CORRECT_COLUMN] == 1
            colour = REGIME_COLOURS[r[REGIME_COL]]
            ax2.plot(r["mean_prob"], i + regime_offsets[r[REGIME_COL]], "o", ms=6.5,
                     color=colour, markerfacecolor=colour if filled else "white", markeredgewidth=1.5)
    ax2.axvline(0.5, color="#555555", lw=1, ls="--")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Mean predicted P(knowledge)")
    ax2.set_title("Mean probability of knowledge, by regime and answer", fontsize=10)

    for a in (ax, ax2):
        a.grid(axis="x", color="#E5E5E5", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([SET_LABELS[s] for s in sets])
    ax.invert_yaxis()

    h1 = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=6, label=lab)
          for c, mk, lab in (COMPARISON_STYLE[k] for k in shown)]
    h2 = [plt.Line2D([], [], color=REGIME_COLOURS[r], marker="o", ls="", ms=6, label=r) for r in MODELLED]
    h2 += [plt.Line2D([], [], color="#555555", marker="o", ls="", ms=6, label="right answer"),
           plt.Line2D([], [], color="#555555", marker="o", ls="", ms=6, markerfacecolor="white", label="wrong answer")]
    ax.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    ax2.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)
    knowledge = "knowledge (partial or full)" if BOTH else WITH_KNOWLEDGE[0]
    fig.suptitle(f"Study 2: {knowledge} against no knowledge, from the answer screen", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=PLOT, tables=tables,
                save=save, close=close, subdir=model_folder(scores))
    return fig


def plot_confusion(
    confusion: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """One heatmap per feature set: each row a regime and answer (right or wrong), the two
    columns what the model said, each cell the share of the row's trials, with the count.
    No-knowledge rows belong in the left column, knowledge rows in the right; the rest are
    the errors. Each title gives the accuracy and balanced accuracy (`accuracy_table`).
    Saved with `confusion` and the accuracies. Returns the figure."""
    sets = [s for s in SET_LABELS if s in set(confusion["feature_set"])]
    accuracy = accuracy_table(confusion)
    by_set = accuracy.set_index("feature_set")
    # Right guesses next to right answers with knowledge, wrong answers at either edge.
    row_order = [(r, c) for r in MODELLED for c in ((0, 1) if r == NO_KNOWLEDGE else (1, 0))]
    ncol = 4
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol + 1.6, 2.9 * nrow + 0.9),
                             sharey=True, layout="constrained", squeeze=False)
    knowledge = "knowledge" if BOTH else WITH_KNOWLEDGE[0]
    for ax, s in zip(axes.flat, sets):
        sub = confusion[confusion["feature_set"] == s].set_index([REGIME_COL, Con.IS_CORRECT_COLUMN])
        rows = sub.loc[row_order]
        share = np.column_stack([1 - rows["share_predicted_knowledge"], rows["share_predicted_knowledge"]])
        counts = rows[["predicted_none", "predicted_knowledge"]].to_numpy()
        image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(share.shape[0]):
            for j in range(2):
                ax.text(j, i, f"{100 * share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center",
                        fontsize=9, color="white" if share[i, j] > 0.6 else "#222222")
        a = by_set.loc[s]
        ax.set_title(f"{SET_LABELS[s]}\naccuracy {a['accuracy']:.3f}, balanced {a['balanced_accuracy']:.3f}",
                     fontsize=9.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["said\nno knowledge", f"said\n{knowledge}"], fontsize=8.5)
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([f"{r}, {'right' if c else 'wrong'} ({int(rows.loc[(r, c), 'n'])})"
                            for r, c in row_order], fontsize=8.5)
        ax.tick_params(length=0)
    for ax in axes.flat[len(sets):]:
        ax.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.6, label="share of the row's trials")
    fig.suptitle(f"Study 2, held-out participants: what the model said, by regime and answer "
                 f"(no knowledge belongs left, {knowledge} right)", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=f"{PLOT}_confusion", tables={"confusion": confusion, "accuracy": accuracy},
                save=save, close=close, subdir=model_folder(confusion))
    return fig


# Below this many trials a group is drawn as one tick per trial, not as a histogram: 20 bins
# over 16 trials would be mostly empty bins and single trials.
MIN_FOR_HISTOGRAM = 30


def plot_histograms(
    hist: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """One panel per feature set: the predicted probability of knowledge on held-out
    participants, a histogram per regime and answer -- colour the regime, solid a right
    answer, dashed a wrong one -- each as shares of its own trials, so groups of different
    sizes compare; the y scales differ between panels. A group under `MIN_FOR_HISTOGRAM` trials is drawn as ticks along the
    bottom, one per trial. The dotted line is the model's cut-off. Saved with `hist`.
    Returns the figure."""
    sets = [s for s in SET_LABELS if s in set(hist["feature_set"])]
    groups = [(r, c) for r in MODELLED for c in ((0, 1) if r == NO_KNOWLEDGE else (1, 0))]
    ncol = 4
    nrow = int(np.ceil(len(sets) / ncol))
    # Each panel has its own y scale: correctness alone puts every trial of a group in one
    # bin, and on a shared scale that would flatten every other panel.
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow + 1.0),
                             sharex=True, layout="constrained", squeeze=False)
    handles = {}
    for ax, s in zip(axes.flat, sets):
        top = hist.loc[(hist["feature_set"] == s) & (hist["n"] >= MIN_FOR_HISTOGRAM), "share"].max()
        for regime, correct in groups:
            sub = hist[(hist["feature_set"] == s) & (hist[REGIME_COL] == regime)
                       & (hist[Con.IS_CORRECT_COLUMN] == correct)]
            colour, ls = REGIME_COLOURS[regime], "-" if correct else "--"
            label = f"{regime}, {'right' if correct else 'wrong'} ({int(sub['n'].iloc[0])})"
            if sub["n"].iloc[0] >= MIN_FOR_HISTOGRAM:
                edges = np.append(sub["bin_low"].to_numpy(), sub["bin_high"].iloc[-1])
                ax.stairs(sub["share"].to_numpy(), edges, color=colour, ls=ls, lw=1.8 if correct else 1.4)
                handles[label] = plt.Line2D([], [], color=colour, ls=ls, lw=1.8)
            else:
                p = preds.loc[(preds["feature_set"] == s) & (preds[REGIME_COL] == regime)
                              & (preds[Con.IS_CORRECT_COLUMN] == correct), "y_prob"]
                ax.plot(p, np.full(len(p), -0.04 * top), "|", color=colour, ms=9, mew=1.4)
                handles[label + ", one tick each"] = plt.Line2D([], [], color=colour, marker="|", ls="", ms=9, mew=1.4)
        ax.axvline(0.5, color="#555555", lw=1, ls=":")
        ax.set_title(SET_LABELS[s], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.08 * top, 1.05 * top)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(sets):]:
        ax.axis("off")
    knowledge = "knowledge" if BOTH else WITH_KNOWLEDGE[0]
    for ax in axes[-1]:
        ax.set_xlabel(f"Predicted P({knowledge})")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of the group's trials")
    fig.legend(handles=list(handles.values()), labels=list(handles), loc="outside lower center",
               ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"Study 2, held-out participants: predicted probability of {knowledge}, "
                 "by regime and answer (dotted: the model's cut-off)", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot=f"{PLOT}_histograms", tables={"histogram": hist},
                save=save, close=close, subdir=model_folder(hist))
    return fig


def main(save: bool = True, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """The predictions, scores and tests, saved as a figure with its tables. Returns the
    tables and the predictions."""
    every_regime = load_trials(verbose=verbose)
    trials = modelled_trials(every_regime, verbose=verbose)
    sets = feature_sets(trials)
    preds = lopo_predictions(trials, sets)
    out = score_tables(preds)
    out["probabilities"] = probability_table(preds)
    out["feature_auc"] = feature_auc(trials, sets["all_features"])
    # The design check is on every regime, since the blocks are.
    out["block_positions"] = block_positions(every_regime)
    out["feature_sets"] = pd.DataFrame(
        [{"feature_set": s, "n_features": len(c), "features": ", ".join(c)} for s, c in sets.items()]
    )
    plot_detection(out["scores"], out["probabilities"], dict(out), save=save, close=True)
    out["confusion"] = confusion_table(preds)
    plot_confusion(out["confusion"], save=save, close=True)
    out["histogram"] = histogram_table(preds)
    plot_histograms(out["histogram"], preds, save=save, close=True)
    if verbose:
        show = ["feature_set", "comparison", "person_auc", "person_ci_low", "person_ci_high",
                "person_p_vs_chance", "pooled_auc", "min_person_auc", "people_above_chance", "balanced_accuracy"]
        print(out["scores"][show].round(3).to_string(index=False))
        show = ["feature_set", "compared_with", "comparison", "person_auc_difference", "person_ci_low",
                "person_ci_high", "person_p", "pooled_auc_difference", "pooled_p"]
        print(out["pairs"][show].round(4).to_string(index=False))
    out["predictions"] = preds
    return out


if __name__ == "__main__":
    main()
