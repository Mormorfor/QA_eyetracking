"""The simplest drift-diffusion model: EZ-diffusion, one fit per participant.

**The model.** Evidence drifts between two bounds, "correct" and "wrong", until it reaches one.
Three numbers describe a participant:

* `v`, the drift rate: how fast evidence builds up -- how easy the questions are for them;
* `a`, the boundary separation: how much evidence they wait for -- how cautious they are;
* `Ter`, the non-decision time: the part of the response time spent on anything else, here
  mostly reading the question and the answers.

**EZ** (Wagenmakers, van der Maas & Grasman 2007) gets the three in closed form from the
accuracy and the mean and variance of the correct decision times, with no fitting. Its scale
is the usual one: within-trial noise `s = 0.1`, times in seconds.

**Two simplifications, both large.** The task has four answers and EZ has two outcomes, so
the three wrong answers are pooled into one "wrong" bound, and the model knows nothing about
the eyes. That is the point of starting here: it is the baseline the gaze models add to.

**Where EZ cannot give a value, as the method states it:**

* accuracy of exactly 1: the log-odds is infinite. The method's own fix is to use
  1 - 1/(2n) instead, for n trials; the rows where this was done say so in `edge_corrected`.
* accuracy of exactly 0.5: the drift is 0 and the boundary is 0/0, so all three are left
  missing.
* accuracy below 0.5 gives a negative drift: with four answers, a participant can be well above
  chance (0.25) and still be wrong more often than right. Study 2's no-knowledge regime is
  around there, so its drift is a statement about "more right than wrong", not about chance.

**The variance of the correct times is very sensitive to the slowest trials**, and decision
times here run to more than a minute. Nothing is trimmed; the table carries the variance so
its influence can be seen.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.explorations.drift_diffusion.data import CONDITION, CONDITION_ORDER, usable
from src.explorations.drift_diffusion.signatures import ANALYSIS, CONDITION_COLOURS

S = 0.1  # the within-trial noise that sets EZ's scale


def ez_diffusion(pc: float, vrt: float, mrt: float, s: float = S) -> tuple[float, float, float]:
    """Drift rate, boundary separation and non-decision time from the proportion correct and
    the variance and mean of the correct decision times (seconds)."""
    L = np.log(pc / (1 - pc))
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt
    v = np.sign(pc - 0.5) * s * x**0.25
    if v == 0:
        return np.nan, np.nan, np.nan
    a = s**2 * L / v
    y = -v * a / s**2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    return v, a, mrt - mdt


def fit_ez(trials: pd.DataFrame) -> pd.DataFrame:
    """One EZ fit per participant and group or regime, on the usable trials.

    Returns the trial counts, the accuracy used (`pc_used`, after the edge correction),
    the mean and variance of the correct decision times, and `v`, `a`, `Ter`."""
    tr = usable(trials).assign(rt_s=lambda d: d["decision_ms"] / 1000)
    rows = []
    for (pid, cond), d in tr.groupby([Con.PARTICIPANT_ID, CONDITION]):
        n = len(d)
        pc = d[Con.IS_CORRECT_COLUMN].mean()
        correct = d.loc[d[Con.IS_CORRECT_COLUMN] == 1, "rt_s"]
        edge = pc == 1
        pc_used = 1 - 1 / (2 * n) if edge else pc
        v, a, ter = ez_diffusion(pc_used, correct.var(), correct.mean())
        rows.append({
            Con.PARTICIPANT_ID: pid, CONDITION: cond, "n_trials": n, "n_correct": len(correct),
            "pc": pc, "pc_used": pc_used, "edge_corrected": edge,
            "mean_rt_correct_s": correct.mean(), "var_rt_correct_s2": correct.var(),
            "v": v, "a": a, "Ter": ter,
        })
    out = pd.DataFrame(rows)
    n_edge = int(out["edge_corrected"].sum())
    n_half = int((out["pc"] == 0.5).sum())
    n_missing = int(out[["v", "a", "Ter"]].isna().any(axis=1).sum())
    if n_missing != n_half:
        raise ValueError(f"EZ: {n_missing} fits have no value but only {n_half} have accuracy 0.5.")
    print(f"EZ: {len(out)} fits | accuracy 1, edge-corrected: {n_edge} | accuracy 0.5, no value: {n_half}")
    return out


def plot_ez(fits: pd.DataFrame, *, dataset_key: str, save: Optional[bool] = None) -> pd.DataFrame:
    """The three EZ parameters per group or regime, one point per participant. On Study 2 every
    participant has all three regimes, so their points are joined; hollow points are fits
    whose accuracy was edge-corrected."""
    conds = CONDITION_ORDER[dataset_key]
    params = [("v", "drift rate v"), ("a", "boundary separation a"), ("Ter", "non-decision time Ter (s)")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (p, name) in zip(axes, params):
        if dataset_key == "knowqa":
            wide = fits.pivot(index=Con.PARTICIPANT_ID, columns=CONDITION, values=p).reindex(columns=conds)
            for _, row in wide.iterrows():
                ax.plot(range(len(conds)), row.to_numpy(), color="#bbbbbb", lw=1, zorder=1)
        for i, cond in enumerate(conds):
            d = fits[fits[CONDITION] == cond]
            jitter = 0 if dataset_key == "knowqa" else rng.uniform(-0.15, 0.15, len(d))
            x = i + jitter
            edge = d["edge_corrected"].to_numpy()
            colour = CONDITION_COLOURS[cond]
            ax.scatter(np.asarray(x + np.zeros(len(d)))[~edge], d[p][~edge], s=18, color=colour, alpha=0.7, zorder=2)
            ax.scatter(np.asarray(x + np.zeros(len(d)))[edge], d[p][edge], s=18, facecolors="none", edgecolors=colour, zorder=2)
            ax.hlines(d[p].median(), i - 0.25, i + 0.25, color="k", lw=2, zorder=3)
        if p == "v":
            ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(name)
    fig.suptitle(f"{dataset_key}: EZ-diffusion per participant (black bar: median; hollow: accuracy 1, edge-corrected)")
    fig.tight_layout()
    summary = fits.groupby(CONDITION)[["pc", "v", "a", "Ter"]].median().add_prefix("median_").reset_index()
    save_output(fig, analysis=ANALYSIS, plot="ez_diffusion", tables={"fits": fits, "medians": summary},
                save=save, dataset=dataset_key)
    return summary
