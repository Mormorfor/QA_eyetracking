"""Figures only the mixed model has: its random effects, and its coefficients beside others.

The random-effect figures read `get_random_effects()`, so they need the mixed model. The
coefficient comparison takes any coefficient tables with `feature`, `coef`, `ci_low`
and `ci_high`, such as the logistic regression's and the mixed model's.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lib.plotting.output import save_output

ANALYSIS = "correctness_prediction"


def _effects(random_effects_df: pd.DataFrame, effect_col: str) -> pd.Series:
    """The effect column as floats. A random effect is estimated for every level, so a
    missing one means the table is not what it should be."""
    vals = random_effects_df[effect_col].astype(float)
    if vals.isna().any():
        raise ValueError(f"{int(vals.isna().sum())} missing values in {effect_col!r}.")
    return vals


def plot_random_effects_barh(
    random_effects_df: pd.DataFrame,
    id_col: str,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    top_n: int = 30,
    sort_by_abs: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "random_effects_barh",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """The `top_n` levels with the largest random effects (in absolute size, or by value
    with `sort_by_abs=False`), as horizontal bars. Returns the figure, the plotted table
    and the saved paths."""
    df = random_effects_df[[id_col]].assign(**{effect_col: _effects(random_effects_df, effect_col)})

    if sort_by_abs:
        df = df.assign(_abs=df[effect_col].abs()).sort_values("_abs", ascending=False)
    else:
        df = df.sort_values(effect_col, ascending=False)
    df = df.head(top_n).sort_values(effect_col, ascending=True).drop(columns="_abs", errors="ignore")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df[id_col].astype(str), df[effect_col])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Random effect (log-odds)")
    ax.set_ylabel(id_col)
    ax.set_title(title or f"Random effects: {id_col}")
    fig.tight_layout()

    paths = save_output(
        fig, analysis=ANALYSIS, plot=plot, tables={"data": df},
        save=save, to_paper=to_paper, dpi=dpi, close=close, subdir=subdir, **facets,
    ).paths
    return fig, df, paths


def plot_random_effects_distribution(
    random_effects_df: pd.DataFrame,
    effect_col: str = "random_intercept",
    title: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (8, 5),
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "random_effects_distribution",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """A histogram of one random effect over all levels. Returns the figure, the values
    and the saved paths."""
    vals = _effects(random_effects_df, effect_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=bins)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Random effect (log-odds)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Random-effects distribution")
    fig.tight_layout()

    paths = save_output(
        fig, analysis=ANALYSIS, plot=plot, tables={"data": vals.to_frame()},
        save=save, to_paper=to_paper, dpi=dpi, close=close, subdir=subdir, **facets,
    ).paths
    return fig, vals, paths


def summarize_random_effects(
    random_effects_df: pd.DataFrame,
    group_name: str,
    effect_col: str = "random_intercept",
) -> pd.DataFrame:
    """One row: how many levels, and the spread of their random effects."""
    vals = _effects(random_effects_df, effect_col)
    return pd.DataFrame([{
        "group_name": group_name,
        "n_levels": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "min": float(vals.min()),
        "q25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "q75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
        "mean_abs": float(vals.abs().mean()),
        "max_abs": float(vals.abs().max()),
    }])


def plot_coefficient_comparison(
    coef_summaries: Mapping[str, pd.DataFrame],
    *,
    order: Optional[Sequence[str]] = None,
    colors: Optional[Mapping[str, str]] = None,
    title: str = "The same features in several fits",
    xlabel: str = "Coefficient (log-odds per SD of the feature)",
    figsize: Tuple[int, int] = (9, 7),
    save: Optional[bool] = None,
    subdir: Optional[str] = "glmm/full_fit/coefficients",
    plot: str = "coefficient_comparison",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    **facets,
):
    """Several fits' coefficients for the same features on one axis, each with its interval.

    `coef_summaries` maps a fit's name to its coefficient table (`feature`, `coef`,
    `ci_low`, `ci_high`). Every table must hold the same features. `order` sets the
    features top to bottom; by default the first table's order. Returns the figure, the
    plotted table (long: one row per fit and feature) and the saved paths.
    """
    names = list(coef_summaries)
    order = list(order) if order is not None else list(coef_summaries[names[0]]["feature"])
    for name, coef in coef_summaries.items():
        if set(coef["feature"]) != set(order):
            raise ValueError(f"{name!r} has different features from the others.")

    long = pd.concat(
        [c.set_index("feature").loc[order, ["coef", "ci_low", "ci_high"]].reset_index().assign(fit=n)
         for n, c in coef_summaries.items()],
        ignore_index=True,
    )[["fit", "feature", "coef", "ci_low", "ci_high"]]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(order))[::-1]
    step = 0.6 / max(len(names), 1)
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * -step
    for offset, name in zip(offsets, names):
        c = long[long["fit"] == name].set_index("feature").loc[order]
        ax.errorbar(
            c["coef"], y + offset,
            xerr=[c["coef"] - c["ci_low"], c["ci_high"] - c["coef"]],
            fmt="o", ms=4, capsize=2, label=name,
            color=(colors or {}).get(name),
        )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    paths = save_output(
        fig, analysis=ANALYSIS, plot=plot, tables={"coefficients": long},
        save=save, to_paper=to_paper, dpi=dpi, close=close, subdir=subdir, **facets,
    ).paths
    return fig, long, paths
