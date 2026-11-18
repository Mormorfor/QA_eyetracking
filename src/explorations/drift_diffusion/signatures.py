"""What the attentional drift-diffusion papers look for in the data, before fitting anything.

Each figure here is a pattern that the gaze-and-choice papers (Shimojo et al. 2003; Krajbich
et al. 2010; Krajbich & Rangel 2011) show as evidence that where people look shapes what they
choose. None of them needs a model.

* `plot_gaze_cascade` -- where the eyes are at each moment, from the first fixation and up to
  and past the decision. The "gaze cascade" is the rising share of gaze on the answer that is
  about to be chosen.
* `plot_cascade_by_condition` -- the same rise, the chosen answer only, one line per group or
  knowledge regime.
* `plot_gaze_bias` -- how often an answer is chosen against how much more it was looked at than
  the other three, separately for the correct answer and each kind of wrong one. Rising lines
  within one kind of answer mean gaze goes with choice beyond what the answer is.
* `plot_first_last` -- how often the first and the last answer looked at is the one chosen,
  against how often that kind of answer is chosen at all.
* `plot_visit_durations` -- how long the first, middle and last visits to an answer last. In
  the aDDM the last visit is cut short by the decision.
* `plot_decision_times` -- the decision-time distributions of right and wrong answers, which
  every drift-diffusion model has to reproduce.

**Means are over participants**: each participant's own share or mean first, then the mean of
those with its standard error, so a participant with many trials does not count for more. The
gaze-bias curve is the exception (`plot_gaze_bias` says why).

**A caveat that applies to all of them:** the same patterns arise if people look at an answer
because they have already decided on it (Zylberberg, Krajbich & Shadlen 2026). The figures show
that gaze and choice go together, not which way round.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.explorations.drift_diffusion.data import CONDITION, CONDITION_ORDER, KEYS, LABELS, GAZE_COLS, usable

ANALYSIS = "explorations/drift_diffusion"

# Where the eyes are, relative to the trial's choice.
ROLE_COLOURS = {
    "chosen": "#4c72b0",
    "correct, not chosen": "#55a868",
    "other answer": "#8c8c8c",
    "question": "#dd8452",
}
LABEL_COLOURS = {"A": "#55a868", "B": "#c44e52", "C": "#dd8452", "D": "#8172b3"}
CONDITION_COLOURS = {
    "hunters": "#4c72b0",
    "gatherers": "#dd8452",
    "no knowledge": "#c44e52",
    "partial knowledge": "#dd8452",
    "full knowledge": "#4c72b0",
}
OUTCOME_NAMES = {1: "correct trials", 0: "wrong trials"}


def _participant_mean(df: pd.DataFrame, by: list, value: str) -> pd.DataFrame:
    """Each participant's mean of `value` within `by`, then the mean over participants, its
    standard error and the number of participants and rows behind it."""
    per = df.groupby([*by, Con.PARTICIPANT_ID])[value].agg(["mean", "size"]).reset_index()
    out = per.groupby(by).agg(
        mean=("mean", "mean"), sd=("mean", "std"), n_participants=("mean", "size"), n_rows=("size", "sum")
    )
    out["se"] = out["sd"] / np.sqrt(out["n_participants"])
    return out.drop(columns="sd").reset_index()


# --- Gaze cascade ------------------------------------------------------------------------

def _roles(fix: pd.DataFrame, trials: pd.DataFrame) -> pd.Series:
    """Each fixation's area named by its relation to the trial's choice."""
    chosen = fix.merge(trials[[*KEYS, "chosen"]], on=KEYS, how="left")["chosen"].to_numpy()
    area = fix["area"].to_numpy()
    role = np.full(len(fix), "other answer", dtype=object)
    role[area == Con.QUESTION_AREA] = "question"
    role[area == Con.CORRECT_ANSWER_AREA] = "correct, not chosen"
    role[area == np.char.add(Con.ANSWER_PREFIX, chosen.astype(str))] = "chosen"
    return pd.Series(role, index=fix.index)


def gaze_timecourse(
    fix: pd.DataFrame,
    trials: pd.DataFrame,
    lock: str,
    offsets: np.ndarray,
) -> pd.DataFrame:
    """Where the eyes were at each time point, on every usable trial.

    `lock` is `"onset"` (time from the first fixation) or `"decision"` (time from the final
    selection press). Returns one row per trial and time point with `t` (ms from the lock)
    and `role`: `chosen`, `correct, not chosen` (wrong trials only), `other answer`,
    `question`, or `no fixation` (between fixations: a saccade or a blink). Time points
    when the answer screen was not up -- before the first fixation, after the confirming
    press -- are left out, so each time point holds only the trials still on screen.
    """
    tr = usable(trials)[[*KEYS, "decision_ms", "confirm_ms", "chosen", Con.IS_CORRECT_COLUMN, CONDITION]]
    f = fix.merge(tr[KEYS], on=KEYS)
    f = f.assign(role=_roles(f, tr))[[*KEYS, "start", "end", "role"]]
    # The join on time needs one type on both sides; the decision times are decimals.
    f["start"] = f["start"].astype("float64")

    anchor = tr["decision_ms"] if lock == "decision" else 0
    grid = tr.assign(anchor=anchor).merge(pd.DataFrame({"t": offsets}), how="cross")
    grid["t_abs"] = (grid["anchor"] + grid["t"]).astype("float64")
    grid = grid[(grid["t_abs"] >= 0) & (grid["t_abs"] <= grid["confirm_ms"])]

    m = pd.merge_asof(
        grid.sort_values("t_abs"), f.sort_values("start"),
        left_on="t_abs", right_on="start", by=KEYS, direction="backward",
    )
    m["role"] = m["role"].where(m["t_abs"] < m["end"], "no fixation")
    return m[[*KEYS, CONDITION, Con.IS_CORRECT_COLUMN, "t", "role"]]


def cascade_table(tc: pd.DataFrame, by: list) -> pd.DataFrame:
    """The share of gaze in each role at each time point, among the time points that fell on a
    fixation, averaged over participants. "other answer" is per answer: the share on the
    unchosen answers divided by how many of them there are (three on correct trials, two
    besides the correct one on wrong trials), so every line is comparable to one answer and
    to chance (one in four)."""
    on = tc[tc["role"] != "no fixation"]
    shares = []
    for role in ROLE_COLOURS:
        x = on.assign(value=(on["role"] == role).astype(float))
        if role == "other answer":
            n_other = np.where(x[Con.IS_CORRECT_COLUMN] == 1, 3, 2)
            x["value"] = x["value"] / n_other
        if role == "correct, not chosen":
            x = x[x[Con.IS_CORRECT_COLUMN] == 0]
        shares.append(_participant_mean(x, [*by, "t"], "value").assign(role=role))
    return pd.concat(shares, ignore_index=True)


def plot_gaze_cascade(
    fix: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    dataset_key: str,
    step_ms: int = 50,
    save: Optional[bool] = None,
) -> Dict[str, pd.DataFrame]:
    """Where the eyes are from the first fixation (left) and around the decision (right), on
    correct (top) and wrong (bottom) trials, all groups or regimes together."""
    onset = gaze_timecourse(fix, trials, "onset", np.arange(0, 4001, step_ms))
    decision = gaze_timecourse(fix, trials, "decision", np.arange(-4000, 1501, step_ms))
    tables = {
        "onset": cascade_table(onset, [Con.IS_CORRECT_COLUMN]),
        "decision": cascade_table(decision, [Con.IS_CORRECT_COLUMN]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for col, (lock, tab) in enumerate(tables.items()):
        for row, outcome in enumerate([1, 0]):
            ax = axes[row, col]
            for role, colour in ROLE_COLOURS.items():
                d = tab[(tab["role"] == role) & (tab[Con.IS_CORRECT_COLUMN] == outcome)]
                if d.empty:
                    continue
                ax.plot(d["t"] / 1000, d["mean"], color=colour, lw=2, label=role)
                ax.fill_between(d["t"] / 1000, d["mean"] - d["se"], d["mean"] + d["se"], color=colour, alpha=0.2, lw=0)
            ax.axhline(0.25, color="k", lw=0.8, ls=":")
            if lock == "decision":
                ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{OUTCOME_NAMES[outcome]}, from the {'first fixation' if lock == 'onset' else 'decision'}")
            ax.set_xlabel("time from the first fixation (s)" if lock == "onset" else "time from the final selection press (s)")
            if col == 0:
                ax.set_ylabel("share of gaze (per answer)")
    axes[1, 0].legend(frameon=False, fontsize=9)  # the wrong-trial panel has all four lines
    fig.suptitle(f"{dataset_key}: gaze cascade (dotted line: one answer in four)")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="gaze_cascade", tables=tables, save=save, dataset=dataset_key)
    return tables


def plot_cascade_by_condition(
    fix: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    dataset_key: str,
    step_ms: int = 50,
    save: Optional[bool] = None,
) -> pd.DataFrame:
    """The share of gaze on the answer about to be chosen, around the decision, one line per
    group (Study 1) or knowledge regime (Study 2)."""
    decision = gaze_timecourse(fix, trials, "decision", np.arange(-4000, 1501, step_ms))
    tab = cascade_table(decision, [CONDITION, Con.IS_CORRECT_COLUMN])
    tab = tab[tab["role"] == "chosen"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, outcome in zip(axes, [1, 0]):
        for cond in CONDITION_ORDER[dataset_key]:
            d = tab[(tab[CONDITION] == cond) & (tab[Con.IS_CORRECT_COLUMN] == outcome)]
            ax.plot(d["t"] / 1000, d["mean"], color=CONDITION_COLOURS[cond], lw=2, label=cond)
            ax.fill_between(d["t"] / 1000, d["mean"] - d["se"], d["mean"] + d["se"], color=CONDITION_COLOURS[cond], alpha=0.2, lw=0)
        ax.axhline(0.25, color="k", lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(OUTCOME_NAMES[outcome])
        ax.set_xlabel("time from the final selection press (s)")
    axes[0].set_ylabel("share of gaze on the chosen answer")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{dataset_key}: gaze cascade by {'group' if dataset_key == 'l1' else 'knowledge regime'}")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="gaze_cascade_by_condition", tables={"chosen": tab}, save=save, dataset=dataset_key)
    return tab


# --- Gaze and choice -----------------------------------------------------------------------

def answers_long(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per usable trial and answer: its kind (`label`), its share of the answer
    dwell (`gaze`), how much more than the mean of the other three (`gaze_advantage`), and
    whether it was chosen."""
    tr = usable(trials)
    g = tr[GAZE_COLS].to_numpy()
    rows = []
    for i, label in enumerate(LABELS):
        others = np.delete(g, i, axis=1).mean(axis=1)
        rows.append(pd.DataFrame({
            **{k: tr[k].to_numpy() for k in KEYS},
            CONDITION: tr[CONDITION].to_numpy(),
            "label": label,
            "gaze": g[:, i],
            "gaze_advantage": g[:, i] - others,
            "is_chosen": (tr["chosen"] == label).to_numpy().astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


def gaze_bias_table(long: pd.DataFrame, n_bins: int = 8, value: str = "is_chosen") -> pd.DataFrame:
    """The choice rate of each kind of answer within bins of its gaze advantage. The bins are
    quantiles of each kind's own advantage, so every bin holds the same number of answers.

    The rate is over the answers in the bin, not averaged over participants: one participant
    has too few wrong answers in a bin for a mean of their own. The standard error is the
    binomial one, so it ignores that answers from one person are alike and is too small."""
    out = []
    for label, d in long.groupby("label"):
        d = d.assign(bin=pd.qcut(d["gaze_advantage"], n_bins, labels=False, duplicates="drop"))
        b = d.groupby("bin").agg(x=("gaze_advantage", "mean"), rate=(value, "mean"), n=(value, "size")).reset_index()
        b["se"] = np.sqrt(b["rate"] * (1 - b["rate"]) / b["n"])
        out.append(b.assign(label=label))
    return pd.concat(out, ignore_index=True)


def plot_gaze_bias(trials: pd.DataFrame, *, dataset_key: str, n_bins: int = 8, save: Optional[bool] = None) -> pd.DataFrame:
    """How often each kind of answer is chosen against how much more it was looked at than the
    other three, one panel per group or regime."""
    long = answers_long(trials)
    conds = CONDITION_ORDER[dataset_key]
    tabs = []
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        tab = gaze_bias_table(long[long[CONDITION] == cond], n_bins).assign(**{CONDITION: cond})
        tabs.append(tab)
        for label in LABELS:
            d = tab[tab["label"] == label]
            ax.errorbar(d["x"], d["rate"], yerr=d["se"], color=LABEL_COLOURS[label], marker="o", ms=4, lw=1.5,
                        capsize=0, label=f"answer {label}" + (" (correct)" if label == "A" else ""))
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_title(cond)
        ax.set_xlabel("its share of answer gaze minus the others' mean")
    axes[0, 0].set_ylabel("P(answer chosen)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset_key}: choice against gaze, by kind of answer (bins of equal size)")
    fig.tight_layout()
    tab = pd.concat(tabs, ignore_index=True)
    save_output(fig, analysis=ANALYSIS, plot="gaze_bias", tables={"binned": tab}, save=save, dataset=dataset_key)
    return tab


def plot_first_last(trials: pd.DataFrame, *, dataset_key: str, save: Optional[bool] = None) -> pd.DataFrame:
    """For each kind of answer: how often it is chosen at all, how often it is chosen when it
    was the first answer looked at, and when it was the last one before the decision."""
    tr = usable(trials)
    rows = []
    for cond in CONDITION_ORDER[dataset_key]:
        d = tr[tr[CONDITION] == cond]
        for label in LABELS:
            base = d.assign(v=(d["chosen"] == label).astype(float))
            first = d[d["first_answer"] == label].assign(v=lambda x: (x["chosen"] == label).astype(float))
            last = d[d["last_answer"] == label].assign(v=lambda x: (x["chosen"] == label).astype(float))
            for what, x in [("chosen at all", base), ("chosen | looked at first", first), ("chosen | looked at last", last)]:
                m = _participant_mean(x.assign(_k=0), ["_k"], "v").drop(columns="_k")
                rows.append(m.assign(**{CONDITION: cond, "label": label, "what": what}))
    tab = pd.concat(rows, ignore_index=True)

    whats = ["chosen at all", "chosen | looked at first", "chosen | looked at last"]
    shades = ["#bbbbbb", "#8fb0d9", "#4c72b0"]
    conds = CONDITION_ORDER[dataset_key]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4), sharey=True, squeeze=False)
    w = 0.27
    for ax, cond in zip(axes[0], conds):
        for k, (what, shade) in enumerate(zip(whats, shades)):
            d = tab[(tab[CONDITION] == cond) & (tab["what"] == what)].set_index("label").reindex(LABELS)
            ax.bar(np.arange(4) + (k - 1) * w, d["mean"], w * 0.92, yerr=d["se"], color=shade, label=what)
        ax.set_xticks(range(4), [f"{l}{' (correct)' if l == 'A' else ''}" for l in LABELS])
        ax.set_title(cond)
    axes[0, 0].set_ylabel("P(this answer chosen)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset_key}: the first and the last answer looked at, and the choice")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="first_last_look", tables={"rates": tab}, save=save, dataset=dataset_key)
    return tab


# --- Visits and decision times -----------------------------------------------------------------

def answer_visits(fix: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Visits to the answers before the decision: runs of consecutive fixations on the same
    area, the last one cut at the press. A visit to the question ends a visit to an answer.
    Each visit is `first`, `last` or `middle` among the trial's answer visits; a trial with
    one answer visit has only a `first` visit, one with two has no `middle`."""
    tr = usable(trials)
    f = fix[fix["before_decision"]].merge(tr[[*KEYS, Con.IS_CORRECT_COLUMN, CONDITION]], on=KEYS)
    f = f.sort_values([*KEYS, "start"])
    f["cut_duration"] = np.minimum(f["end"], f["decision_ms"]) - f["start"]
    new_trial = f[KEYS].ne(f[KEYS].shift()).any(axis=1)
    new_visit = new_trial | f["area"].ne(f["area"].shift())
    f["visit"] = new_visit.cumsum()
    # A visit's duration is the sum of its fixations, the saccades between them left out.
    v = f.groupby("visit").agg(
        **{k: (k, "first") for k in [*KEYS, Con.IS_CORRECT_COLUMN, CONDITION, "area"]},
        start=("start", "first"), duration=("cut_duration", "sum"),
    )
    v = v[v["area"].isin(Con.ANSWER_AREAS)].copy()
    rank = v.groupby(KEYS).cumcount()
    count = v.groupby(KEYS)["area"].transform("size")
    v["position"] = np.where(rank == 0, "first", np.where(rank == count - 1, "last", "middle"))
    return v.reset_index(drop=True)


def plot_visit_durations(fix: pd.DataFrame, trials: pd.DataFrame, *, dataset_key: str, save: Optional[bool] = None) -> pd.DataFrame:
    """Mean duration of the first, middle and last answer visits, on correct and wrong
    trials, per group or regime. Trials with a single answer visit are left out (their one
    visit is both first and last); the table says how many visits are behind each mean."""
    v = answer_visits(fix, trials)
    multi = v.groupby(KEYS)["area"].transform("size") >= 2
    tab = _participant_mean(v[multi], [CONDITION, Con.IS_CORRECT_COLUMN, "position"], "duration")
    n_single = int((~multi).sum())

    conds = CONDITION_ORDER[dataset_key]
    positions = ["first", "middle", "last"]
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.8), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for outcome, colour in [(1, "#4c72b0"), (0, "#c44e52")]:
            d = tab[(tab[CONDITION] == cond) & (tab[Con.IS_CORRECT_COLUMN] == outcome)].set_index("position").reindex(positions)
            ax.errorbar(range(3), d["mean"], yerr=d["se"], color=colour, marker="o", lw=2, capsize=0, label=OUTCOME_NAMES[outcome])
        ax.set_xticks(range(3), positions)
        ax.set_title(cond)
        ax.set_xlabel("answer visit")
    axes[0, 0].set_ylabel("visit duration (ms)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{dataset_key}: answer visits before the decision ({n_single} single-visit trials left out)")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="visit_durations", tables={"durations": tab}, save=save, dataset=dataset_key)
    return tab


def plot_decision_times(trials: pd.DataFrame, *, dataset_key: str, save: Optional[bool] = None) -> pd.DataFrame:
    """Decision-time distributions of right and wrong answers, per group or regime, on a log
    time axis, with the quantiles a drift-diffusion fit is usually checked against."""
    tr = usable(trials).assign(rt_s=lambda d: d["decision_ms"] / 1000)
    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    tab = (
        tr.groupby([CONDITION, Con.IS_CORRECT_COLUMN])["rt_s"].quantile(qs).unstack()
        .rename(columns=lambda q: f"q{int(q * 100)}").reset_index()
    )
    tab["n"] = tr.groupby([CONDITION, Con.IS_CORRECT_COLUMN]).size().to_numpy()

    conds = CONDITION_ORDER[dataset_key]
    bins = np.logspace(np.log10(tr["rt_s"].min()), np.log10(tr["rt_s"].max()), 50)
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        for outcome, colour in [(1, "#4c72b0"), (0, "#c44e52")]:
            d = tr[(tr[CONDITION] == cond) & (tr[Con.IS_CORRECT_COLUMN] == outcome)]["rt_s"]
            ax.hist(d, bins=bins, density=True, histtype="step", lw=2, color=colour,
                    label=f"{OUTCOME_NAMES[outcome]} (median {d.median():.1f} s)")
        ax.set_xscale("log")
        ax.set_title(cond)
        ax.set_xlabel("decision time (s)")
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("density")
    fig.suptitle(f"{dataset_key}: time to the final selection press")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="decision_times", tables={"quantiles": tab}, save=save, dataset=dataset_key)
    return tab
