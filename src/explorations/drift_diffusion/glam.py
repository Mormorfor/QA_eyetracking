"""A gaze-weighted race between the four answers: the Gaze-weighted Linear Accumulator Model.

GLAM (Thomas, Molter, Krajbich, Heekeren & Mohr 2019) is the simplest model in the
attentional drift-diffusion family that handles more than two options and can be fitted
exactly. It keeps the aDDM's central idea -- an option gathers evidence faster while it is
looked at -- but uses only how much of the trial each option was looked at, not the order
of the looks.

**The model, for one trial.** Each answer i has a value `r_i` and a share `g_i` of the
answer gaze before the decision.

1. Its evidence is its value, discounted while it is not looked at:
   `A_i = g_i * r_i + (1 - g_i) * gamma * r_i`. `gamma = 1` means gaze does nothing; below 1,
   the less an answer is looked at, the less its value counts.
2. What matters is how it compares with the best of the others:
   `R_i = logistic(tau * (A_i - max of the other A_j))`; `tau` says how sharply.
3. Each answer has an accumulator that rises at rate `v * R_i` with noise `sigma` and stops
   at 1. The first to stop is the choice, and when it stops is the decision time.

The time for one accumulator to reach its bound has a known distribution (the Wald, or
inverse Gaussian), so the likelihood of a choice at a time is exact: the chosen answer's
accumulator stopping then, and the other three not yet.

**What the values are here.** In the papers `r_i` is a rating the participant gave the item
beforehand. These answers have no rating, but they have a kind: A is correct, B misreads the
right sentence, C comes from another sentence, D is not in the text (`research-context.md`).
So `r` is one value per kind, fitted: `r_A = 1` sets the scale, and `r_B`, `r_C`, `r_D` say
how convincing each kind of wrong answer is.

**One fit per group or regime, pooling the participants**, as Krajbich & Rangel (2011) fitted
the aDDM. People differ a lot in speed, so a pooled `sigma` and `v` describe nobody in
particular; the comparison that means something is with and without gaze (`gamma` free, and
fixed at 1) on the same trials. There is no non-decision time: the whole time to the final
selection press is treated as the decision. Fitted and checked on the same trials.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.explorations.drift_diffusion.data import CONDITION, CONDITION_ORDER, GAZE_COLS, LABELS, usable
from src.explorations.drift_diffusion.signatures import ANALYSIS, LABEL_COLOURS, answers_long, gaze_bias_table

PARAMS = ["v", "gamma", "sigma", "tau", "r_B", "r_C", "r_D"]
BOUNDS = {
    "v": (1e-3, 20.0),
    # The aDDM's range for its attention discount: unattended evidence counts between not at
    # all (0) and fully (1). Below 0 an answer would lose evidence while not looked at,
    # which is a different model.
    "gamma": (0.0, 1.0),
    "sigma": (1e-3, 5.0),
    "tau": (1e-3, 100.0),
    "r_B": (0.0, 2.0),
    "r_C": (0.0, 2.0),
    "r_D": (0.0, 2.0),
}
MODELS = {"gaze": {}, "no gaze": {"gamma": 1.0}}  # parameters each model holds fixed


def drift_rates(g: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """Each answer's drift rate on each trial; `g` is trials x answers (A-D) gaze shares."""
    r = np.array([1.0, p["r_B"], p["r_C"], p["r_D"]])
    A = r * (g + (1 - g) * p["gamma"])
    top2 = np.sort(A, axis=1)[:, -2:]
    max_other = np.where(A == top2[:, [1]], top2[:, [0]], top2[:, [1]])
    return p["v"] * expit(p["tau"] * (A - max_other))


def _log1mexp(d: np.ndarray) -> np.ndarray:
    """log(1 - exp(d)) for d < 0, accurate at both ends."""
    return np.where(d > -np.log(2), np.log(-np.expm1(d)), np.log1p(-np.exp(d)))


def wald_logpdf(t: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Log density of the time a drift-mu, noise-sigma accumulator takes to reach 1."""
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(t) - (1 - mu * t) ** 2 / (2 * sigma**2 * t)


def wald_logsf(t: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Log probability that the accumulator has not reached 1 by time t."""
    root = sigma * np.sqrt(t)
    lo = log_ndtr(-(mu * t - 1) / root)
    d = 2 * mu / sigma**2 + log_ndtr(-(mu * t + 1) / root) - lo
    return lo + _log1mexp(d)


def trial_loglik(p: Dict[str, float], g: np.ndarray, choice: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """Log likelihood of each trial's choice (0-3 for A-D) at its decision time (s)."""
    mu = drift_rates(g, p)
    t = rt[:, None]
    logf = wald_logpdf(t, mu, p["sigma"])
    logS = wald_logsf(t, mu, p["sigma"])
    rows = np.arange(len(rt))
    return logf[rows, choice] + logS.sum(axis=1) - logS[rows, choice]


def _arrays(trials: pd.DataFrame):
    tr = usable(trials)
    g = tr[GAZE_COLS].to_numpy()
    choice = tr["chosen"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    rt = (tr["decision_ms"] / 1000).to_numpy()
    assert (rt > 0).all(), "a decision time is not positive"
    return g, choice, rt


def fit_glam(trials: pd.DataFrame, fixed: Dict[str, float], n_starts: int = 8, seed: int = 0) -> Dict[str, float]:
    """Maximum-likelihood fit on the usable trials of `trials`, with the parameters in `fixed`
    held at their values. Several random starts; the best is kept, and the fit stops with
    an error unless the best start converged."""
    g, choice, rt = _arrays(trials)
    free = [k for k in PARAMS if k not in fixed]
    rng = np.random.default_rng(seed)

    def nll(x):
        ll = trial_loglik({**fixed, **dict(zip(free, x))}, g, choice, rt).sum()
        # A non-finite value is a parameter region the optimizer must leave, not data.
        return -ll if np.isfinite(ll) else np.inf

    best = None
    for _ in range(n_starts):
        x0 = [rng.uniform(*{"v": (0.1, 1.0), "sigma": (0.1, 1.0), "tau": (0.5, 10.0)}.get(k, BOUNDS[k])) for k in free]
        if not np.isfinite(nll(x0)):
            continue
        res = minimize(nll, x0, method="L-BFGS-B", bounds=[BOUNDS[k] for k in free])
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not best.success:
        raise RuntimeError(f"GLAM fit did not converge: {None if best is None else best.message}")
    params = {**fixed, **dict(zip(free, best.x))}
    at_bound = [k for k in free if np.isclose(params[k], BOUNDS[k][0]) or np.isclose(params[k], BOUNDS[k][1])]
    if at_bound:
        print(f"  GLAM: at a bound: {', '.join(at_bound)}")
    return {**params, "loglik": -best.fun, "n_free": len(free), "n_trials": len(rt), "aic": 2 * best.fun + 2 * len(free)}


def fit_all(trials: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Both models (gaze, no gaze) in every group or regime."""
    rows = []
    for cond in CONDITION_ORDER[dataset_key]:
        d = trials[trials[CONDITION] == cond]
        for model, fixed in MODELS.items():
            fit = fit_glam(d, fixed)
            rows.append({CONDITION: cond, "model": model, **fit})
            print(f"  {cond} / {model}: " + ", ".join(f"{k} {fit[k]:.3g}" for k in [*PARAMS, "loglik", "aic"]))
    return pd.DataFrame(rows)


def wald_sample(rng: np.random.Generator, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Times for drift-mu, noise-sigma accumulators to reach 1 (Michael, Schucany & Haas
    1976), written in the drift rather than the mean time. numpy's `wald` takes the mean time,
    1/mu, and for the near-zero drifts of an answer with almost no value it loses all
    precision and returns times near 0, so that answer wins every race it is in."""
    lam = 1 / sigma**2
    y = rng.standard_normal(mu.shape) ** 2
    x1 = 4 * lam * y / (y + np.sqrt(4 * lam * y * mu + y**2)) ** 2
    accept = rng.uniform(size=mu.shape) <= 1 / (1 + mu * x1)
    # np.where works out both branches. Where mu**2 underflows to 0 the second one is
    # infinite, but there `accept` is certain, so it is never the one returned.
    with np.errstate(divide="ignore"):
        return np.where(accept, x1, 1 / (mu**2 * x1))


def simulate(trials: pd.DataFrame, p: Dict[str, float], n_draws: int = 20, seed: int = 1) -> pd.DataFrame:
    """`n_draws` simulated choices and decision times for every usable trial, each with that
    trial's own gaze shares."""
    g, _, _ = _arrays(trials)
    mu = drift_rates(g, p)
    rng = np.random.default_rng(seed)
    t = wald_sample(rng, np.repeat(mu, n_draws, axis=0), p["sigma"])
    tr = usable(trials).loc[lambda d: d.index.repeat(n_draws)].reset_index(drop=True)
    tr["sim_choice"] = np.array(LABELS)[t.argmin(axis=1)]
    tr["sim_rt_s"] = t.min(axis=1)
    return tr


def plot_glam_check(trials: pd.DataFrame, fits: pd.DataFrame, *, dataset_key: str, n_bins: int = 8,
                    save: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
    """The fitted models against the data, per group or regime. Top: choice against gaze
    (points: data; solid: the gaze model; dashed: without gaze). Bottom: decision-time
    quantiles, the models' against the data's, for correct and wrong answers."""
    conds = CONDITION_ORDER[dataset_key]
    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    bias_rows, rt_rows = [], []
    fig, axes = plt.subplots(2, len(conds), figsize=(4.4 * len(conds), 8), squeeze=False)
    for col, cond in enumerate(conds):
        d = trials[trials[CONDITION] == cond]
        obs = gaze_bias_table(answers_long(d), n_bins).assign(source="data")
        bias_rows.append(obs.assign(**{CONDITION: cond}))
        ax = axes[0, col]
        for label in LABELS:
            o = obs[obs["label"] == label]
            ax.errorbar(o["x"], o["rate"], yerr=o["se"], color=LABEL_COLOURS[label], marker="o", ms=4, lw=0, elinewidth=1,
                        label=f"answer {label}")
        rt_ax = axes[1, col]
        obs_q = usable(d).groupby(Con.IS_CORRECT_COLUMN)["decision_ms"].quantile(qs).div(1000)
        for model, ls in [("gaze", "-"), ("no gaze", "--")]:
            p = fits[(fits[CONDITION] == cond) & (fits["model"] == model)].iloc[0][PARAMS].to_dict()
            sim = simulate(d, p)
            long = answers_long(d).sort_values(["label"], kind="stable")
            # Each trial-answer row's simulated choice rate, over the draws.
            rate = sim.assign(_trial=np.repeat(np.arange(len(usable(d))), len(sim) // len(usable(d))))
            share = pd.crosstab(rate["_trial"], rate["sim_choice"], normalize="index").reindex(columns=LABELS, fill_value=0)
            long["sim_is_chosen"] = np.concatenate([share[l].to_numpy() for l in LABELS])
            pred = gaze_bias_table(long, n_bins, value="sim_is_chosen").assign(source=model)
            bias_rows.append(pred.assign(**{CONDITION: cond}))
            for label in LABELS:
                q = pred[pred["label"] == label]
                ax.plot(q["x"], q["rate"], color=LABEL_COLOURS[label], ls=ls, lw=1.5)
            sim["sim_correct"] = (sim["sim_choice"] == "A").astype(int)
            sim_q = sim.groupby("sim_correct")["sim_rt_s"].quantile(qs)
            for outcome, marker in [(1, "o"), (0, "s")]:
                rt_ax.plot(obs_q.loc[outcome], sim_q.loc[outcome], ls=ls, marker=marker, ms=5,
                           color="#4c72b0" if outcome else "#c44e52",
                           label=f"{model}, {'correct' if outcome else 'wrong'}")
                rt_rows.append(pd.DataFrame({CONDITION: cond, "model": model, Con.IS_CORRECT_COLUMN: outcome, "quantile": qs,
                                             "data_s": obs_q.loc[outcome].to_numpy(), "model_s": sim_q.loc[outcome].to_numpy()}))
        ax.set_title(cond)
        ax.set_xlabel("its share of answer gaze minus the others' mean")
        lim = [0, max(rt_ax.get_xlim()[1], rt_ax.get_ylim()[1])]
        rt_ax.plot(lim, lim, color="k", lw=0.8, ls=":")
        rt_ax.set_xlabel("data: decision-time quantile (s)")
        rt_ax.set_ylabel("model: decision-time quantile (s)")
    axes[0, 0].set_ylabel("P(answer chosen)")
    axes[0, 0].legend(frameon=False, fontsize=8, title="points: data\nsolid: gaze model\ndashed: no gaze", title_fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{dataset_key}: GLAM fitted, against the data")
    fig.tight_layout()
    tables = {"fits": fits, "gaze_bias": pd.concat(bias_rows, ignore_index=True), "rt_quantiles": pd.concat(rt_rows, ignore_index=True)}
    save_output(fig, analysis=ANALYSIS, plot="glam_check", tables=tables, save=save, dataset=dataset_key)
    return tables


def plot_glam_parameters(fits: pd.DataFrame, *, dataset_key: str, save: Optional[bool] = None) -> pd.DataFrame:
    """The gaze model's gaze discount and answer values, and how much better it fits than the
    model without gaze (AIC difference; above 0 favours gaze), per group or regime."""
    conds = CONDITION_ORDER[dataset_key]
    g = fits[fits["model"] == "gaze"].set_index(CONDITION).reindex(conds)
    n = fits[fits["model"] == "no gaze"].set_index(CONDITION).reindex(conds)
    tab = g[["gamma", "tau", "r_B", "r_C", "r_D", "loglik", "aic", "n_trials"]].copy()
    tab["aic_no_gaze_minus_gaze"] = n["aic"] - g["aic"]
    tab = tab.reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(conds))
    axes[0].bar(x, tab["gamma"], color="#4c72b0", width=0.6)
    axes[0].axhline(1, color="k", lw=0.8, ls=":")
    axes[0].set_title("gaze discount gamma (1: gaze does nothing)")
    w = 0.25
    for k, label in enumerate(["B", "C", "D"]):
        axes[1].bar(x + (k - 1) * w, tab[f"r_{label}"], w * 0.92, color=LABEL_COLOURS[label], label=f"answer {label}")
    axes[1].axhline(1, color="k", lw=0.8, ls=":")
    axes[1].set_title("value of each wrong answer (A = 1)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].bar(x, tab["aic_no_gaze_minus_gaze"], color="#8c8c8c", width=0.6)
    axes[2].set_title("AIC without gaze minus with gaze")
    for ax in axes:
        ax.set_xticks(x, conds)
    fig.suptitle(f"{dataset_key}: GLAM parameters, one pooled fit per {'group' if dataset_key == 'l1' else 'regime'}")
    fig.tight_layout()
    save_output(fig, analysis=ANALYSIS, plot="glam_parameters", tables={"parameters": tab}, save=save, dataset=dataset_key)
    return tab
