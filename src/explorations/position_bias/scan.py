"""What the opening scan and the gaze do with each position.

The choice tests (`choice.py`) ask whether the choice leans towards a position; this module
shows what the eyes do, in every condition, so the two can be read side by side: where the first
look at an answer lands, each participant's usual first look, how much gaze each position draws
on the answers *not* chosen (the chosen one draws gaze because it is being chosen), how often
each position is visited, and where the eyes go when they come back down from the question.
And, for the choices: whether the pull to the top depends on how fast the decision was, and
whether it goes with lower confidence (Study 2 only: Study 1 has no rating).

Everything here needs gaze before the decision, so it uses the trials that have it
(`data.with_gaze`); each table says how many.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import columns as Con
from src.config.datasets import dataset
from src.explorations.drift_diffusion.data import CONDITION, KEYS
from src.explorations.position_bias.data import (
    LOC_TO_POSITION, PID, POSITIONS, participant_mean, subsets, with_gaze,
)
from src.features.strategies import build_starting_strategies, dominant_strategy_by_participant

SUBSET = "subset"
SPEED_THIRDS = ["fastest third", "middle third", "slowest third"]


def usual_starts(answers: pd.DataFrame, dataset_key: str) -> Tuple[pd.Series, pd.DataFrame]:
    """Each participant's usual first look: the position their first look at an answer lands on
    most often, over all their trials with gaze.

    It counts only when it leads the runner-up by at least two trials, so that no single trial
    decides it -- which keeps the test of choices at the usual first look exact
    (`choice.randomisation_tests`). Returns it per participant (missing where none is that
    clear), and a table per participant: the counts per position, the share and lead of the
    usual one, and their dominant starting strategy (`features/strategies.py`, over all their
    trials, question visits removed) with its dominance score."""
    trials = with_gaze(answers).drop_duplicates(KEYS)
    counts = pd.crosstab(trials[PID], trials["first_position"]).reindex(columns=POSITIONS, fill_value=0)
    top_two = np.sort(counts.to_numpy(), axis=1)[:, -2:]
    table = counts.add_prefix("n_first_").copy()
    table["usual_start"] = counts.idxmax(axis=1).where(top_two[:, 1] - top_two[:, 0] >= 2)
    table["usual_start_share"] = counts.max(axis=1) / counts.sum(axis=1)
    table["usual_start_lead"] = top_two[:, 1] - top_two[:, 0]

    raw = pd.read_csv(dataset(dataset_key).trials, usecols=[*KEYS, Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION],
                      dtype={PID: str})
    dom = dominant_strategy_by_participant(build_starting_strategies(raw, kind="location", drop_question=True))
    dom = dom.set_index(PID)
    table["dominant_strategy"] = dom[Con.DOMINANT_STARTING_STRATEGY].map(
        lambda s: " → ".join(LOC_TO_POSITION[x] for x in s))
    table[Con.DOMINANCE_SCORE] = dom[Con.DOMINANCE_SCORE]
    table[Con.N_STRATEGY_TRIALS] = dom[Con.N_STRATEGY_TRIALS]
    return table["usual_start"], table.reset_index()


def choices_by_start(answers: pd.DataFrame, dataset_key: str, usual: pd.Series) -> pd.DataFrame:
    """Per participant and group: how often the scan started at their usual first look, and how
    often the choice went to the top answer, to the answer at their usual first look, and to
    the first answer reached."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        chosen = with_gaze(d)[lambda x: x["is_chosen"] == 1].copy()
        chosen["usual_start"] = chosen[PID].map(usual)
        chosen = chosen.assign(
            started_at_usual_start=chosen["first_position"] == chosen["usual_start"],
            chose_top=chosen["position"] == "top",
            chose_usual_start=chosen["position"] == chosen["usual_start"],
            chose_first_reached=chosen["order"] == 1,
        )
        t = chosen.groupby(PID).agg(
            usual_start=("usual_start", "first"), n_trials=("label", "size"),
            share_started_at_usual_start=("started_at_usual_start", "mean"),
            share_chose_top=("chose_top", "mean"), share_chose_usual_start=("chose_usual_start", "mean"),
            share_chose_first_reached=("chose_first_reached", "mean"),
        )
        parts.append(t.reset_index().assign(**{SUBSET: name}))
    return pd.concat(parts, ignore_index=True)


def first_look_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Per group: the share of trials whose first look at an answer landed at each position."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        t = with_gaze(d).drop_duplicates(KEYS)
        x = pd.concat([t.assign(position=p, value=(t["first_position"] == p).astype(float)) for p in POSITIONS])
        per = participant_mean(x, ["position"], "value").rename(columns={"mean": "participant_mean"})
        counts = t["first_position"].value_counts().reindex(POSITIONS, fill_value=0)
        per["share"] = per["position"].map(counts / len(t))
        parts.append(per.assign(**{SUBSET: name, "n_trials": len(t)}))
    return pd.concat(parts, ignore_index=True)


def gaze_table(answers: pd.DataFrame, dataset_key: str, by: str) -> pd.DataFrame:
    """Per group: each answer's share of the answer gaze before the decision, by `by`
    (`position` or `order`), for the answer chosen and the ones not chosen: means over
    participants."""
    parts = [participant_mean(with_gaze(d), ["is_chosen", by], "gaze").assign(**{SUBSET: name})
             for name, d in subsets(answers, dataset_key)]
    return pd.concat(parts, ignore_index=True).rename(columns={"mean": "gaze_share"})


def returns_table(fix: pd.DataFrame, answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Per group and position, before the decision: visits per trial, and where the eyes go when
    they come back down from the question -- the share of answer visits right after a visit to
    the question, other than the trial's opening one, that land there. A visit is a run of
    fixations on one area, as in `drift_diffusion.signatures.answer_visits`."""
    rows = with_gaze(answers)
    f = fix[fix["before_decision"]].merge(rows[KEYS].drop_duplicates(), on=KEYS).sort_values([*KEYS, "start"])
    new_visit = f[KEYS].ne(f[KEYS].shift()).any(axis=1) | f["area"].ne(f["area"].shift())
    v = f[new_visit][[*KEYS, "area"]].copy()
    v["previous"] = v.groupby(KEYS)["area"].shift()
    v = v[v["area"].isin(Con.ANSWER_AREAS)].copy()
    v["is_opening"] = ~v.duplicated(KEYS)
    v["label"] = v["area"].str.removeprefix(Con.ANSWER_PREFIX)
    v = v.merge(rows[[*KEYS, "label", "position", CONDITION]], on=[*KEYS, "label"], how="left")
    assert v["position"].notna().all(), "a visit to an answer has no position"

    parts = []
    for (name, d), (_, dv) in zip(subsets(rows, dataset_key), subsets(v, dataset_key)):
        n_trials = len(d) // 4
        visits = dv["position"].value_counts().reindex(POSITIONS, fill_value=0)
        back = dv[(dv["previous"] == Con.QUESTION_AREA) & ~dv["is_opening"]]["position"].value_counts().reindex(
            POSITIONS, fill_value=0)
        parts.append(pd.DataFrame({
            SUBSET: name, "position": POSITIONS,
            "visits_per_trial": (visits / n_trials).to_numpy(),
            "n_returns_from_question": back.to_numpy(),
            "share_of_returns_from_question": (back / back.sum()).to_numpy(),
            "n_trials": n_trials,
        }))
    return pd.concat(parts, ignore_index=True)


def _choices(d: pd.DataFrame) -> pd.DataFrame:
    """One row per trial with gaze: whether the choice went to the top answer and to the first
    one reached, and the third of the participant's own decision times it falls in (within the
    group)."""
    c = with_gaze(d)[lambda x: x["is_chosen"] == 1].copy()
    c["chose_top"] = (c["position"] == "top").astype(float)
    c["chose_first_reached"] = (c["order"] == 1).astype(float)
    third = c.groupby(PID)["decision_ms"].transform(lambda s: pd.qcut(s, 3, labels=False))
    c["speed"] = third.map(dict(enumerate(SPEED_THIRDS)))
    return c


def speed_table(answers: pd.DataFrame, dataset_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per group, in each third of a participant's own decision times: how often the choice went
    to the top answer and to the first answer reached (participant means), and the median
    decision time. And per group, a chi-squared test of whether the share going to the top is
    the same in the three thirds."""
    parts, tests = [], []
    for name, d in subsets(answers, dataset_key):
        c = _choices(d)
        top = participant_mean(c, ["speed"], "chose_top").rename(columns={"mean": "share_chose_top", "se": "se_top"})
        first = participant_mean(c, ["speed"], "chose_first_reached")[["speed", "mean", "se"]].rename(
            columns={"mean": "share_chose_first_reached", "se": "se_first_reached"})
        median = c.groupby("speed")["decision_ms"].median().rename("median_decision_ms").reset_index()
        t = top.merge(first, on="speed").merge(median, on="speed").set_index("speed").loc[SPEED_THIRDS].reset_index()
        parts.append(t.assign(**{SUBSET: name}))
        chi = stats.chi2_contingency(pd.crosstab(c["speed"], c["chose_top"]))
        tests.append({SUBSET: name, "test": "choices at the top: the same share in each third of decision times",
                      "n_trials": len(c), "statistic": chi.statistic, "df": chi.dof, "p": chi.pvalue})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(tests)


def confidence_table(answers: pd.DataFrame, dataset_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Study 2, per regime: the mean confidence rating when the choice went to the top answer and
    when it went elsewhere (participant means), and the same for the first answer reached; and a
    test over participants of each difference (Wilcoxon signed-rank, participants who made both
    kinds of choice)."""
    parts, tests = [], []
    for name, d in subsets(answers, dataset_key):
        c = _choices(d)
        for col in ["chose_top", "chose_first_reached"]:
            target = col.removeprefix("chose_")
            t = participant_mean(c, [col], Con.CONFIDENCE_RATING_COLUMN).rename(
                columns={col: "went_there", "mean": "confidence"})
            parts.append(t.assign(target=target, **{SUBSET: name}))
            per = c.groupby([PID, col])[Con.CONFIDENCE_RATING_COLUMN].mean().unstack().dropna()
            w = stats.wilcoxon(per[1.0] - per[0.0])
            tests.append({SUBSET: name, "test": f"confidence: the same whether or not the {target.replace('_', ' ')} answer was chosen",
                          "n_trials": len(c), "n_participants": len(per), "statistic": w.statistic, "p": w.pvalue})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(tests)


def deviation_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Per group: where the chosen answer sat, by where the trial's first look landed. A count,
    and the share within each starting position."""
    parts = []
    for name, d in subsets(answers, dataset_key):
        c = with_gaze(d)[lambda x: x["is_chosen"] == 1]
        counts = pd.crosstab(c["first_position"], c["position"]).reindex(columns=POSITIONS, fill_value=0)
        t = counts.stack().rename("n").reset_index().rename(columns={"position": "chosen_position"})
        t["share_of_row"] = t["n"] / t.groupby("first_position")["n"].transform("sum")
        parts.append(t.assign(**{SUBSET: name}))
    return pd.concat(parts, ignore_index=True)
