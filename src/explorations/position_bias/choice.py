"""Did the choice lean towards a position, or towards the answer the scan reached first?

**Randomisation tests** (`randomisation_tests`) come first. Each re-draws the shuffle of the
answers per unit (`data.UNIT_COLS`) and asks how often a random placement gives a statistic as
far from even as the one observed. If where an answer sits never changed which answer is
chosen, the chosen answer, the wrong answer chosen and the correct answer would each land at a
random position, so these tests are exact, whatever the answers are like and however people
differ. They cover where the chosen answer sat, where the errors went, whether accuracy depends
on where the correct answer sat, whether the choice went to the participant's usual first look,
and whether the first look itself goes to one kind of answer more than another -- if it does
not, the first look is not drawn by content.

**Conditional logits** (`choice_models`) then say how large the effects are, over and above the
kind of answer (A-D), which is in every model: the chosen answer against the other three of its
trial. Standard errors are clustered on participants; intervals and p resample participants.
Study 2's full-knowledge regime is left out: almost every choice there is the correct answer.

**Read the order models with knowledge with care.** Someone who knows the answer often decides
soon after reaching it, so the trials that reach all four answers lean towards those where the
known answer came late. A guess has no such pull.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import columns as Con
from src.explorations.drift_diffusion.data import CONDITION, KEYS, LABELS
from src.explorations.position_bias.data import (
    ALL_PARTICIPANTS, GUESSING, ORDER_NAMES, PID, POSITIONS, participant_mean, random_placements, subsets, with_gaze,
)

SUBSET = "subset"
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
KIND_INDEX = {k: i for i, k in enumerate(LABELS)}

MODELLED = {"knowqa": [GUESSING, "partial knowledge"], "l1": [ALL_PARTICIPANTS]}

# Each model: its terms, and the trials it is fitted on. "complete scans" are the trials that
# reached all four answers before the decision: an answer not reached by then has no place in
# the order, and can hardly be chosen, so keeping those trials would make "late in the order"
# partly mean "never seen".
MODELS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "position": (("kind", "position"), "all trials"),
    "order": (("kind", "order"), "complete scans"),
    "top_and_first": (("kind", "top", "first"), "trials with gaze"),
    "length": (("kind", "position", "length"), "all trials"),
    "content": (("kind", "position", "length", "attractiveness"), "trials with attractiveness"),
}

TERM_LABELS = {
    "kind_B": "B (misreads the right sentence)",
    "kind_C": "C (from another sentence)",
    "kind_D": "D (not in the text)",
    "position_left": "left",
    "position_right": "right",
    "position_bottom": "bottom",
    "at_top": "at the top",
    "reached_first": "reached first",
    "reached_2nd": "2nd",
    "reached_3rd": "3rd",
    "reached_4th": "4th",
    "log2_words": "twice as many words",
    "attractiveness": "chosen by everyone else (vs nobody)",
}


# --- Shares ---------------------------------------------------------------------------------------

def _shares(d: pd.DataFrame, by: Sequence[str], value: str = "is_chosen") -> pd.DataFrame:
    """Pooled count and share of `value` in each `by` cell, and the mean over participants of
    each one's own share, with its standard error."""
    by = list(by)
    pooled = d.groupby(by)[value].agg(n_yes="sum", n_rows="size").reset_index()
    pooled["share"] = pooled["n_yes"] / pooled["n_rows"]
    per = participant_mean(d, by, value).rename(columns={"mean": "participant_mean"}).drop(columns="n_rows")
    return pooled.merge(per, on=by)


def position_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: how often the chosen answer sat at each position, over all trials and
    over the wrong ones."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        parts.append(_shares(d, ["position"]).assign(**{SUBSET: name, "trials": "all trials"}))
        wrong = d[d[Con.IS_CORRECT_COLUMN] == 0]
        parts.append(_shares(wrong, ["position"]).assign(**{SUBSET: name, "trials": "wrong answers"}))
    return pd.concat(parts, ignore_index=True)


def accuracy_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: accuracy by where the correct answer sat, and by when the scan first
    reached it (trials with gaze; 0 = not before the decision)."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        correct = d[d["label"] == "A"]
        parts.append(_shares(correct, ["position"]).rename(columns={"position": "level"}).assign(
            **{SUBSET: name, "by": "position of the correct answer"}))
        gaze = with_gaze(correct)
        t = _shares(gaze, ["order"])
        parts.append(t.assign(level=t["order"].map(ORDER_NAMES)).drop(columns="order").assign(
            **{SUBSET: name, "by": "when the correct answer was first reached"}))
    return pd.concat(parts, ignore_index=True).rename(columns={"n_yes": "n_correct", "share": "accuracy"})


def order_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: how often the answer reached 1st .. 4th (or not before the decision) was
    chosen, over the trials with gaze and over the complete scans."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        g = with_gaze(d)
        parts.append(_shares(g, ["order"]).assign(**{SUBSET: name, "trials": "trials with gaze"}))
        parts.append(_shares(g[g["n_reached"] == 4], ["order"]).assign(**{SUBSET: name, "trials": "complete scans"}))
    out = pd.concat(parts, ignore_index=True)
    out["order_name"] = out["order"].map(ORDER_NAMES)
    return out


def participant_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Each participant's own share of choices at each position, and by arrival order on the
    complete scans, for each group."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        pos = d.groupby([PID, "position"])["is_chosen"].mean().rename("share").reset_index()
        g = with_gaze(d)
        full = g[g["n_reached"] == 4]
        order = full.groupby([PID, "order"])["is_chosen"].mean().rename("share").reset_index()
        parts += [
            pos.rename(columns={"position": "level"}).assign(by="position", **{SUBSET: name}),
            order.assign(level=order["order"].map(ORDER_NAMES)).drop(columns="order").assign(
                by="order, complete scans", **{SUBSET: name}),
        ]
    return pd.concat(parts, ignore_index=True)


def kind_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: how often each kind of answer was chosen."""
    return pd.concat([_shares(d, ["label"]).assign(**{SUBSET: name}) for name, d in subsets(answers, dataset_key)],
                     ignore_index=True)


def length_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: how much longer the chosen answer is than its trial's average, in log2
    words (0: no preference; 1: twice as long), as a mean over participants."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        lw = np.log2(d["n_words"])
        above = lw - lw.groupby([d[k] for k in KEYS]).transform("mean")
        chosen = d.assign(above=above)[d["is_chosen"] == 1]
        per = chosen.groupby(PID)["above"].mean()
        rows.append({SUBSET: name, "log2_words_above_trial_mean": per.mean(), "se": per.std() / np.sqrt(len(per)),
                     "n_participants": len(per), "n_trials": len(chosen)})
    return pd.DataFrame(rows)


def top_summary(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group, as means over participants: the share of choices that went to the top
    answer; accuracy with the correct answer at the top minus accuracy with it elsewhere
    (participants who had both); and, on the answers not chosen, the top answer's share of the
    gaze minus the mean share of the other three positions (trials with gaze)."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        chosen = d[d["is_chosen"] == 1]
        top = chosen.groupby(PID)["position"].apply(lambda s: (s == "top").mean())
        correct = d[d["label"] == "A"]
        acc = correct.groupby([PID, correct["position"] == "top"])["is_chosen"].mean().unstack().dropna()
        acc_gain = acc[True] - acc[False]
        g = with_gaze(d)
        g = g[g["is_chosen"] == 0].groupby([PID, g["position"] == "top"])["gaze"].mean().unstack().dropna()
        gaze_gain = g[True] - g[False]
        for measure, per in [("share of choices at the top", top),
                             ("accuracy: correct answer at the top minus elsewhere", acc_gain),
                             ("gaze on answers not chosen: top minus elsewhere", gaze_gain)]:
            rows.append({SUBSET: name, "measure": measure, "mean": per.mean(),
                         "se": per.std() / np.sqrt(len(per)), "n_participants": len(per)})
    return pd.DataFrame(rows)


# --- Randomisation tests --------------------------------------------------------------------------

TESTS = {
    "positions": "chosen answer: at each position equally often",
    "top": "chosen answer: at the top one time in four",
    "errors_positions": "wrong answer chosen: at each position equally often",
    "errors_top": "wrong answer chosen: at the top one time in four",
    "accuracy": "accuracy: the same wherever the correct answer sat",
    "accuracy_top": "accuracy: the same with the correct answer at the top as elsewhere",
    "usual": "chosen answer: at the participant's usual first look one time in four",
    "first_kind": "first look: on each kind of answer equally often",
}


def _chi2(counts: np.ndarray) -> float:
    """Chi-squared distance of counts from equal."""
    expected = counts.sum() / counts.size
    return float(((counts - expected) ** 2 / expected).sum())


def _chi2_independence(table: np.ndarray) -> float:
    """Chi-squared statistic of independence of a two-way table of counts."""
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


def _trial_arrays(answers: pd.DataFrame, usual: pd.Series) -> Dict[str, np.ndarray]:
    """One entry per trial: its unit, the kind chosen, whether it was right, where the first
    look landed (-1 without gaze) and the participant's usual first look (-1 if none); and the
    observed position of each kind in each unit."""
    trials = answers[answers["is_chosen"] == 1].sort_values(KEYS)
    units = answers.drop_duplicates(["unit", "label"])
    observed = np.full((int(answers["unit"].max()) + 1, 4), -1)
    observed[units["unit"].to_numpy(), units["label"].map(KIND_INDEX).to_numpy()] = units["position"].map(POS_INDEX).to_numpy()
    return {
        "condition": trials[CONDITION].to_numpy(),
        "unit": trials["unit"].to_numpy(),
        "chosen": trials["label"].map(KIND_INDEX).to_numpy(),
        "correct": trials[Con.IS_CORRECT_COLUMN].to_numpy().astype(bool),
        "first": trials["first_position"].map(POS_INDEX).fillna(-1).astype(int).to_numpy(),
        "usual": trials[PID].map(usual).map(POS_INDEX).fillna(-1).astype(int).to_numpy(),
        "observed": observed,
    }


def _trial_statistics(place: np.ndarray, tr: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """Every statistic of `TESTS`, for every group, under one placement: `place[u, kind]` is
    the position of each kind in each unit."""
    pos_chosen = place[tr["unit"], tr["chosen"]]
    pos_correct = place[tr["unit"], 0]
    kind_at = np.argsort(place, axis=1)  # the inverse: kind_at[u, p] is the kind at position p
    out = {}
    for name, m in masks.items():
        pc = pos_chosen[m]
        out[(name, "positions")] = _chi2(np.bincount(pc, minlength=4))
        out[(name, "top")] = abs((pc == 0).mean() - 0.25)
        w = m & ~tr["correct"]
        out[(name, "errors_positions")] = _chi2(np.bincount(pos_chosen[w], minlength=4)) if w.any() else np.nan
        out[(name, "errors_top")] = abs((pos_chosen[w] == 0).mean() - 0.25) if w.any() else np.nan
        right = tr["correct"][m].astype(int)
        table = np.bincount(right * 4 + pos_correct[m], minlength=8).reshape(2, 4)
        out[(name, "accuracy")] = _chi2_independence(table)
        top = pos_correct[m] == 0
        out[(name, "accuracy_top")] = abs(right[top].mean() - right[~top].mean())
        u = m & (tr["usual"] >= 0)
        out[(name, "usual")] = abs((pos_chosen[u] == tr["usual"][u]).mean() - 0.25)
        g = m & (tr["first"] >= 0)
        out[(name, "first_kind")] = _chi2(np.bincount(kind_at[tr["unit"][g], tr["first"][g]], minlength=4))
    return out


def randomisation_tests(answers: pd.DataFrame, dataset_key: str, usual: pd.Series, *,
                        n_draws: int = 10000, seed: int = 0) -> pd.DataFrame:
    """For each group, every test of `TESTS`: the observed statistic and the share of `n_draws`
    random placements at least as far from even. `usual` is each participant's usual first look
    (`scan.usual_starts`), missing for those without a clear one. One draw re-places every unit
    once, and serves every group and test."""
    tr = _trial_arrays(answers, usual)
    names = [name for name, _ in subsets(answers, dataset_key)]
    masks = {name: np.ones(len(tr["unit"]), bool) if name == ALL_PARTICIPANTS else tr["condition"] == name
             for name in names}
    observed = _trial_statistics(tr["observed"], tr, masks)
    extreme = dict.fromkeys(observed, 0)
    for place in random_placements(len(tr["observed"]), n_draws, seed):
        for key, value in _trial_statistics(place, tr, masks).items():
            extreme[key] += value >= observed[key] - 1e-12
    rows = []
    for name in names:
        m = masks[name]
        pos = tr["observed"][tr["unit"], tr["chosen"]]
        w = m & ~tr["correct"]
        top = tr["observed"][tr["unit"], 0] == 0
        u = m & (tr["usual"] >= 0)
        g = m & (tr["first"] >= 0)
        described = {
            "positions": (int(m.sum()), np.nan),
            "top": (int(m.sum()), (pos[m] == 0).mean()),
            "errors_positions": (int(w.sum()), np.nan),
            "errors_top": (int(w.sum()), (pos[w] == 0).mean() if w.any() else np.nan),
            "accuracy": (int(m.sum()), tr["correct"][m].mean()),
            "accuracy_top": (int(m.sum()), tr["correct"][m & top].mean() - tr["correct"][m & ~top].mean()),
            "usual": (int(u.sum()), (pos[u] == tr["usual"][u]).mean()),
            "first_kind": (int(g.sum()), np.nan),
        }
        for key, test in TESTS.items():
            n, value = described[key]
            rows.append({SUBSET: name, "test": test, "n_trials": n, "value": value,
                         "statistic": observed[(name, key)],
                         "p": (extreme[(name, key)] + 1) / (n_draws + 1)})
    out = pd.DataFrame(rows)
    out[["n_draws", "seed"]] = n_draws, seed
    return out


# --- Conditional logit ------------------------------------------------------------------------------

def _design(d: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    """The model's columns for each answer. References: answer A, the top position, the answer
    reached first."""
    X = {}
    if "kind" in terms:
        X |= {f"kind_{l}": d["label"] == l for l in LABELS[1:]}
    if "position" in terms:
        X |= {f"position_{p}": d["position"] == p for p in POSITIONS[1:]}
    if "top" in terms:
        X["at_top"] = d["position"] == "top"
    if "first" in terms:
        X["reached_first"] = d["order"] == 1
    if "order" in terms:
        X |= {f"reached_{ORDER_NAMES[k]}": d["order"] == k for k in (2, 3, 4)}
    if "length" in terms:
        X["log2_words"] = np.log2(d["n_words"])
    if "attractiveness" in terms:
        X["attractiveness"] = d["attractiveness"]
    return pd.DataFrame(X, index=d.index).astype(float)


def conditional_logit(X: np.ndarray, chosen: np.ndarray, max_iter: int = 100, start: np.ndarray | None = None):
    """McFadden's conditional logit by Newton-Raphson. `X` is trials x 4 answers x terms;
    `chosen` is the index of the chosen answer on each trial; `start` is where Newton begins
    (zeros by default). Returns the coefficients, each trial's score and the information
    matrix, all at the estimate. Stops with an error if it does not converge -- as when a kind
    of answer is never chosen, and its estimate runs off to minus infinity."""
    n = X.shape[0]
    beta = np.zeros(X.shape[2]) if start is None else start.copy()
    for _ in range(max_iter):
        u = X @ beta
        p = np.exp(u - u.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        mean_x = np.einsum("tj,tjk->tk", p, X)
        scores = X[np.arange(n), chosen] - mean_x
        dev = X - mean_x[:, None, :]
        info = np.einsum("tj,tjk,tjl->kl", p, dev, dev)
        step = np.linalg.solve(info, scores.sum(axis=0))
        if np.abs(step).max() < 1e-9:
            return beta, scores, info
        beta = beta + step
    raise RuntimeError("the conditional logit did not converge")


def _model_rows(d: pd.DataFrame, which: str) -> pd.DataFrame:
    """The rows a model of `MODELS` is fitted on."""
    if which == "all trials":
        return d
    g = with_gaze(d)
    if which == "trials with gaze":
        return g
    if which == "complete scans":
        return g[g["n_reached"] == 4]
    if which == "trials with attractiveness":
        return d[d["attractiveness"].notna().groupby([d[k] for k in KEYS]).transform("all")]
    raise KeyError(which)


def _arrays(d: pd.DataFrame, terms: Sequence[str]):
    X = _design(d, terms)
    n = len(d) // 4
    keys = d[KEYS].to_numpy().reshape(n, 4, len(KEYS))
    assert len(d) == 4 * n and (keys == keys[:, :1, :]).all(), "rows are not four answers per trial, in order"
    chosen = d["is_chosen"].to_numpy().reshape(n, 4).argmax(axis=1)
    return list(X.columns), X.to_numpy().reshape(n, 4, -1), chosen, keys[:, 0, 0]


def _bootstrap(X: np.ndarray, chosen: np.ndarray, people: np.ndarray, n_boot: int, seed: int,
               start: np.ndarray) -> np.ndarray:
    """Coefficients refitted on participants drawn with replacement, Newton starting from the
    full-sample estimate `start`. The same seed gives the same draws to every model."""
    ids = np.unique(people)
    rows = {p: np.flatnonzero(people == p) for p in ids}
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, X.shape[2]))
    for b in range(n_boot):
        pick = np.concatenate([rows[p] for p in rng.choice(ids, size=len(ids), replace=True)])
        out[b] = conditional_logit(X[pick], chosen[pick], start=start)[0]
    return out


def choice_models(answers: pd.DataFrame, dataset_key: str, *, n_boot: int = 2000, ci: float = 0.95,
                  seed: int = 0) -> pd.DataFrame:
    """Every model of `MODELS` in every group of `MODELLED`. One row per term: the coefficient
    and its odds ratio; the odds ratio's interval and p from resampling participants; and the
    participant-clustered standard error with its p."""
    rows = []
    for name, d_all in subsets(answers, dataset_key):
        if name not in MODELLED[dataset_key]:
            continue
        for model, (terms, which) in MODELS.items():
            names, X, chosen, people = _arrays(_model_rows(d_all, which), terms)
            beta, scores, info = conditional_logit(X, chosen)
            bread = np.linalg.inv(info)
            S = pd.DataFrame(scores).groupby(people).sum().to_numpy()
            G = len(S)
            se = np.sqrt(np.diag(bread @ S.T @ S @ bread * G / (G - 1)))
            boot = _bootstrap(X, chosen, people, n_boot, seed, start=beta)
            lo, hi = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
            p_boot = np.clip(2 * np.minimum((boot <= 0).mean(axis=0), (boot >= 0).mean(axis=0)), 1 / n_boot, 1)
            for i, term in enumerate(names):
                rows.append({
                    SUBSET: name, "model": model, "trials": which, "term": term,
                    "coef": beta[i], "odds_ratio": np.exp(beta[i]),
                    "ci_low": np.exp(lo[i]), "ci_high": np.exp(hi[i]), "p_boot": p_boot[i],
                    "se_cluster": se[i], "p_cluster": 2 * stats.norm.sf(abs(beta[i] / se[i])),
                    "n_trials": len(chosen), "n_participants": G,
                })
    out = pd.DataFrame(rows)
    out[["n_boot", "ci", "seed"]] = n_boot, ci, seed
    return out
