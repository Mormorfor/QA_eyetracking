"""The answer screen as one decision: when it was made, what was chosen, and where the eyes
were until then.

**The decision** is the final selection press: the last time the participant pressed an
answer before confirming. A trial can have several selection presses when the participant
changed their mind; only the last one counts as the choice, as in `features/last_visited.py`.

**Time zero** is the start of the trial's first fixation on the answer screen. The fixation
report and the press table share the tracker's clock but neither records the moment the
screen appeared, and `CONFIRM_FINAL_ANSWER_RT` is measured from the same first fixation, so
every time here is on the same footing as that column. `decision_ms` is the decision time.

**What the models see stops at the decision.** Fixations that start after the final selection
press (the participant looking around before confirming) are left out of the dwell times and
gaze shares, and the fixation under way at the press is cut at the press. They are kept in
`load_fixations`, so the figures that run past the decision can show them. Both counts are
printed.

**Trials that cannot enter the models**, counted and printed by `load_trials` and kept in
the table with the reason in `excluded`:

* no recorded selection press: there is no decision time;
* no fixation on any answer before the decision: there are no gaze shares.

Study 1's answer-screen fixation report is about 1.8 GB. The placed fixations are saved once
under `reports/_cache/drift_diffusion/` and read back while the report, the word table and the
trial table are unchanged (by size and modification time); `rerun=True` rebuilds them.
"""

from __future__ import annotations

import ast
import hashlib
import json

import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import FIX_ANSWERS_PATH, KNOW_QA_FIX_ANSWERS_PATH, dataset
from src.config.outputs import CACHE_ROOT
from src.features.fixations import load_fixations as read_fixation_report
from src.features.fixations import place_fixations

CACHE_DIR = CACHE_ROOT / "drift_diffusion"
FIXATION_REPORTS = {"l1": FIX_ANSWERS_PATH, "knowqa": KNOW_QA_FIX_ANSWERS_PATH}

KEYS = list(TRIAL_ID_COLS)
LABELS = ["A", "B", "C", "D"]  # A is always the correct answer
CONDITION = "condition"  # hunters / gatherers on Study 1, the knowledge regime on Study 2

# Study 1's question_preview flag, named as everywhere else in the project.
PREVIEW_GROUPS = {True: "hunters", False: "gatherers"}
REGIME_ORDER = ["no knowledge", "partial knowledge", "full knowledge"]
CONDITION_ORDER = {"l1": ["hunters", "gatherers"], "knowqa": REGIME_ORDER}

DWELL_COLS = [f"dwell_{l}" for l in LABELS]
GAZE_COLS = [f"gaze_{l}" for l in LABELS]


def _last_press(times) -> float:
    """The latest of a trial's presses. The press table holds them as a text list, e.g.
    "[45120, 47380]"; a trial with no press of that kind has no list."""
    return max(ast.literal_eval(times), default=np.nan) if isinstance(times, str) else np.nan


def _fingerprint(ds, report) -> str:
    """Size and modification time of every file the placed fixations are built from."""
    files = [report, ds.interest_areas, ds.trials, ds.button_clicks]
    stamp = [(str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files]
    return hashlib.sha1(json.dumps(stamp).encode()).hexdigest()[:12]


def _build_fixations(dataset_key: str, verbose: bool) -> pd.DataFrame:
    ds = dataset(dataset_key)
    trials = pd.read_csv(ds.trials, usecols=KEYS, dtype={Con.PARTICIPANT_ID: str})
    words = pd.read_csv(
        ds.interest_areas,
        usecols=[*KEYS, Con.INTEREST_AREA_ID, Con.AREA_LABEL_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )
    fix = read_fixation_report(
        FIXATION_REPORTS[dataset_key],
        [Con.CURRENT_FIX_START, Con.CURRENT_FIX_DURATION, Con.CURRENT_FIX_INTEREST_AREA_ID, Con.NEAREST_IA],
        verbose=verbose,
    )
    placed = place_fixations(fix, words, [Con.AREA_LABEL_COLUMN], label="drift_diffusion", verbose=verbose)
    # Only the analysed trials; the report also holds practice and excluded ones.
    placed = placed.merge(trials, on=KEYS, how="inner")
    unplaced = int(placed[Con.AREA_LABEL_COLUMN].isna().sum())
    if unplaced:
        # Every fixation of an analysed trial gets a word through the nearest-word rule, so
        # one without an area means the report is not what this code expects.
        raise ValueError(f"{dataset_key}: {unplaced} fixation(s) in analysed trials have no area.")
    out = placed[[*KEYS, Con.CURRENT_FIX_START, Con.CURRENT_FIX_DURATION, Con.AREA_LABEL_COLUMN]].rename(
        columns={Con.CURRENT_FIX_START: "clock_start", Con.CURRENT_FIX_DURATION: "duration", Con.AREA_LABEL_COLUMN: "area"}
    )
    return out.sort_values([*KEYS, "clock_start"], kind="stable").reset_index(drop=True)


def _placed_fixations(dataset_key: str, rerun: bool, verbose: bool) -> pd.DataFrame:
    ds = dataset(dataset_key)
    path = CACHE_DIR / f"{dataset_key}_fixations__{_fingerprint(ds, FIXATION_REPORTS[dataset_key])}.pkl"
    if path.exists() and not rerun:
        if verbose:
            print(f"[cache] {dataset_key} fixations: read {path.name}")
        return pd.read_pickle(path)
    out = _build_fixations(dataset_key, verbose)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    if verbose:
        print(f"[cache] {dataset_key} fixations: saved {path.name}")
    return out


def _press_times(dataset_key: str, trials: pd.DataFrame) -> pd.DataFrame:
    ds = dataset(dataset_key)
    presses = pd.read_csv(
        ds.button_clicks,
        usecols=[*KEYS, Con.SELECT_ANS_TIMESTAMPS, Con.CONFIRM_TIMESTAMPS],
        dtype={Con.PARTICIPANT_ID: str},
    )
    presses[Con.TRIAL_ID] = presses[Con.TRIAL_ID].astype(trials[Con.TRIAL_ID].dtype)
    presses = presses.assign(
        clock_decision=presses[Con.SELECT_ANS_TIMESTAMPS].map(_last_press),
        clock_confirm=presses[Con.CONFIRM_TIMESTAMPS].map(_last_press),
    )[[*KEYS, "clock_decision", "clock_confirm"]]
    presses = presses.merge(trials[KEYS], on=KEYS, how="inner")
    assert_full_coverage(trials, presses, KEYS, "press times")
    return presses


def load_fixations(dataset_key: str, *, rerun: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Every answer-screen fixation of every analysed trial, in time order.

    One row per fixation: the trial ids, `area` (`question`, `answer_A` .. `answer_D`),
    `start` and `end` in ms from the trial's first fixation, `duration`, and
    `decision_ms` / `confirm_ms` (the final selection press and the confirming press on the
    same clock). Nothing is cut at the decision here; `before_decision` says which fixations
    started before it.
    """
    ds = dataset(dataset_key)
    trials = pd.read_csv(ds.trials, usecols=KEYS, dtype={Con.PARTICIPANT_ID: str})
    fix = _placed_fixations(dataset_key, rerun, verbose)
    presses = _press_times(dataset_key, trials)

    zero = fix.groupby(KEYS)["clock_start"].min().rename("clock_zero").reset_index()
    fix = fix.merge(zero, on=KEYS).merge(presses, on=KEYS, how="left")
    fix["start"] = fix["clock_start"] - fix["clock_zero"]
    fix["end"] = fix["start"] + fix["duration"]
    fix["decision_ms"] = fix["clock_decision"] - fix["clock_zero"]
    fix["confirm_ms"] = fix["clock_confirm"] - fix["clock_zero"]
    fix["before_decision"] = fix["start"] < fix["decision_ms"]
    return fix[[*KEYS, "area", "start", "end", "duration", "decision_ms", "confirm_ms", "before_decision"]]


def load_trials(dataset_key: str, fixations: pd.DataFrame | None = None, *, verbose: bool = True) -> pd.DataFrame:
    """One row per analysed trial, with what the decision models need.

    `condition`, `chosen` (A-D), `is_correct`, `decision_ms` (final selection press),
    `confirm_ms`, `n_select` (selection presses), the dwell on each answer before the
    decision (`dwell_A` .. `dwell_D`, ms, the fixation under way at the press cut at it),
    `gaze_A` .. `gaze_D` (each answer's share of that answer dwell), `first_answer` and
    `last_answer` (the first and last answer looked at before the decision), and
    `excluded`, empty for a trial the models can use and the reason otherwise.
    """
    ds = dataset(dataset_key)
    extra = [Con.QUESTION_PREVIEW_COLUMN] if dataset_key == "l1" else [Con.REGIME_COLUMN]
    trials = pd.read_csv(
        ds.trials,
        usecols=[*KEYS, Con.SELECTED_ANSWER_LABEL_COLUMN, Con.IS_CORRECT_COLUMN,
                 Con.CONFIRM_FINAL_ANSWER_RT, "ANSWER_PRESS_NUMBER", *extra],
        dtype={Con.PARTICIPANT_ID: str},
    )
    trials[CONDITION] = (
        trials[Con.QUESTION_PREVIEW_COLUMN].map(PREVIEW_GROUPS) if dataset_key == "l1" else trials[Con.REGIME_COLUMN]
    )
    trials = trials.rename(columns={Con.SELECTED_ANSWER_LABEL_COLUMN: "chosen", "ANSWER_PRESS_NUMBER": "n_select"})
    if fixations is None:
        fixations = load_fixations(dataset_key, verbose=verbose)

    times = fixations.groupby(KEYS)[["decision_ms", "confirm_ms"]].first().reset_index()
    assert_full_coverage(trials, times, KEYS, "decision times")
    trials = trials.merge(times, on=KEYS, how="left")
    gap = (trials["confirm_ms"] - trials[Con.CONFIRM_FINAL_ANSWER_RT]).abs()
    assert gap.median() < 100, (
        f"{dataset_key}: the confirming press differs from CONFIRM_FINAL_ANSWER_RT by "
        f"{gap.median():.0f} ms on the typical trial; the presses and fixations are not on one clock."
    )

    before = fixations[fixations["before_decision"] & fixations["area"].isin(Con.ANSWER_AREAS)].copy()
    before["cut_dwell"] = np.minimum(before["end"], before["decision_ms"]) - before["start"]
    before["label"] = before["area"].str.removeprefix(Con.ANSWER_PREFIX)
    dwell = before.pivot_table(index=KEYS, columns="label", values="cut_dwell", aggfunc="sum", fill_value=0)
    # An answer with no fixation before the decision had zero dwell: a real amount of attention.
    dwell = dwell.reindex(columns=LABELS, fill_value=0).add_prefix("dwell_").reset_index()
    order = before.sort_values([*KEYS, "start"]).groupby(KEYS)["label"]
    firstlast = pd.concat([order.first().rename("first_answer"), order.last().rename("last_answer")], axis=1).reset_index()

    out = trials.merge(dwell, on=KEYS, how="left").merge(firstlast, on=KEYS, how="left")
    # A trial with no answer fixation before the decision is absent from `dwell`: zero dwell
    # on all four, and no gaze shares (0/0), which `excluded` records.
    out[DWELL_COLS] = out[DWELL_COLS].fillna(0)
    total = out[DWELL_COLS].sum(axis=1)
    for d, g in zip(DWELL_COLS, GAZE_COLS):
        out[g] = out[d] / total.where(total > 0)

    out["excluded"] = ""
    out.loc[total == 0, "excluded"] = "no answer looked at before the decision"
    out.loc[out["decision_ms"].isna(), "excluded"] = "no selection press"

    if verbose:
        n_after = int((~fixations["before_decision"]).sum())
        n_cut = int((fixations["before_decision"] & (fixations["end"] > fixations["decision_ms"])).sum())
        print(
            f"{dataset_key}: {len(out)} trials | fixations after the decision (left out of the "
            f"models): {n_after:,} of {len(fixations):,} | cut at the press: {n_cut:,}"
        )
        reasons = out.loc[out["excluded"] != "", "excluded"].value_counts()
        print(f"  trials the models cannot use: {int(reasons.sum())}" + "".join(f" | {r}: {n}" for r, n in reasons.items()))
    return out


def usable(trials: pd.DataFrame) -> pd.DataFrame:
    """The trials the models can use (`excluded` empty)."""
    return trials[trials["excluded"] == ""]
