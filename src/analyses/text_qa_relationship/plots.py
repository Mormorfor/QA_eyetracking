"""
Heatmaps, forest plots and summary tables for the correlation maps.

`plot_corr_map` / `plot_corr_map_pair` draw the dict returned by
`correlations.corr_map_stats` (or `bootstrap.cluster_bootstrap_corr_map`) with
its significance marked:

- by default the colour scale runs to the largest |r| in the map instead of a
  fixed [-1, 1], on which a map whose largest |r| is 0.1 looks uniformly white
- cells that do not survive the multiple-comparison correction are hatched and
  greyed, so noise is not mistaken for structure
- each cell prints r, its stars, and optionally its 95% interval

`plot_pooled_maps` is the plain view: pooled r on a fixed [-1, 1] scale, with
no tests. `plot_contrasts` draws a table from `comparisons` as a forest plot.
`summarise_map` turns a map into a readable table.

Figures and their tables are saved through `save_output`, under
`reports/text_qa_relationship/`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.lib.plotting.output import save_output
import pandas as pd

from src.analyses.text_qa_relationship._utils import significance_stars
from src.analyses.text_qa_relationship.columns import (
    METRICS,
    answer_cols,
    pretty_labels,
    region_cols,
)
from src.analyses.text_qa_relationship.correlations import DEFAULT_METHOD, pooled_corr_map

__all__ = [
    "plot_corr_map",
    "plot_corr_map_pair",
    "plot_contrasts",
    "plot_pooled_maps",
    "shared_vmax",
    "summarise_map",
]


def _round_up(vmax: float, step: float = 0.05) -> float:
    return max(step, float(np.ceil(vmax / step) * step))


def shared_vmax(*stats, step: float = 0.05) -> float:
    """One colour limit covering several maps, so they can be read against each other.

    Each figure otherwise scales to its own range, so the same colour would mean
    different things in two figures side by side. Takes stats dicts at any
    nesting -- a single map, `{metric: stats}`, or `{group: {metric: stats}}` --
    and returns the largest |r| rounded up to the next `step`.

    >>> vmax = shared_vmax(*boot_by_group.values())
    """
    def walk(obj):
        if isinstance(obj, dict):
            if "r" in obj and isinstance(obj["r"], pd.DataFrame):
                yield float(np.nanmax(np.abs(obj["r"].values)))
                return
            for value in obj.values():
                yield from walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                yield from walk(value)

    found = [v for arg in stats for v in walk(arg)]
    if not found:
        raise ValueError("no correlation maps found in the given arguments")
    return _round_up(max(found), step=step)


def plot_corr_map(
    stats: dict,
    ax=None,
    title: str = "",
    vmax: float | None = None,
    show_ci: bool = False,
    cmap: str = "RdBu_r",
    annotate_ns: bool = True,
):
    """Draw one correlation map as an annotated heatmap; returns `(ax, im)`.

    `stats` is a dict from `corr_map_stats` or `cluster_bootstrap_corr_map`;
    only "r" is required. Stars come from "p_adj". Cells whose "sig" is False
    print in grey and, with `annotate_ns` (the default), are hatched; a dict
    without "sig" marks no cell. `vmax` is the symmetric colour limit, by
    default the largest |r| in the map rounded up to the next 0.05, so small
    but real structure stays visible. `show_ci` prints the interval under each r.
    """
    r = stats["r"]
    p_adj = stats.get("p_adj")
    sig = stats.get("sig")

    if vmax is None:
        vmax = _round_up(float(np.nanmax(np.abs(r.values))))

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))

    im = ax.imshow(r.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(r.shape[1]))
    ax.set_xticklabels(pretty_labels(r.columns), rotation=45, ha="right")
    ax.set_yticks(range(r.shape[0]))
    ax.set_yticklabels(pretty_labels(r.index))

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            is_sig = True if sig is None else bool(sig.values[i, j])
            stars = "" if p_adj is None else significance_stars(p_adj.values[i, j])

            if annotate_ns and not is_sig:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="grey", lw=0,
                    )
                )

            colour = "white" if abs(v) > 0.6 * vmax else "black"
            if not is_sig:
                colour = "grey"

            label = f"{v:.2f}{stars}"
            if show_ci and "ci_low" in stats:
                label += (
                    f"\n[{stats['ci_low'].values[i, j]:.2f},"
                    f" {stats['ci_high'].values[i, j]:.2f}]"
                )
            ax.text(
                j, i, label, ha="center", va="center",
                color=colour, fontsize=8 if show_ci else 9,
            )

    ax.set_title(title, fontsize=10)
    return ax, im


def plot_corr_map_pair(
    stats_by_metric: dict[str, dict],
    suptitle: str = "",
    vmax: float | None = None,
    show_ci: bool = False,
    figsize: tuple[float, float] = (14, 4.5),
    footnote: str | None = (
        "cell = mean within-participant r [95% CI]; "
        "stars = BH-adjusted p; hatched = not significant"
    ),
    save: Optional[bool] = None,
    to_paper=None,
    plot: str = "text_qa_corr_map",
    show: bool = True,
    **facets,
):
    """Several maps side by side (typically RT and TFD) on one colour scale,
    saved with their numbers; returns `(fig, axes)`.

    `stats_by_metric` maps each panel's title to a stats dict. The colour limit
    is the largest |r| across the panels unless `vmax` is given. The figure is
    saved through `save_output` (name from `plot` and the `facets`), with each
    panel's "r", "p_adj" and "n" as tables (`<panel>__r`, ...). A bootstrap
    dict has no "n", so only its first two are saved.
    """
    keys = list(stats_by_metric)
    if vmax is None:
        vmax = _round_up(
            max(float(np.nanmax(np.abs(s["r"].values))) for s in stats_by_metric.values())
        )

    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    axes = np.atleast_1d(axes)
    im = None
    for ax, key in zip(axes, keys):
        _, im = plot_corr_map(
            stats_by_metric[key], ax=ax, title=key, vmax=vmax, show_ci=show_ci
        )

    fig.colorbar(im, ax=axes, label="r", fraction=0.02, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle, y=1.04)
    if footnote:
        fig.text(0.5, -0.12, footnote, ha="center", fontsize=8, color="dimgrey")

    # The numbers behind each panel are saved next to the figure, so they can be
    # quoted without reading them off the image.
    tables = {}
    for key, st in stats_by_metric.items():
        for field in ("r", "p_adj", "n"):
            mat = st.get(field)
            if mat is not None and hasattr(mat, "reset_index"):
                tables[f"{key}__{field}"] = mat.reset_index(names="text_region")

    save_output(
        fig,
        analysis="text_qa_relationship",
        plot=plot,
        tables=tables,
        save=save,
        to_paper=to_paper,
        dpi=140,
        **facets,
    )

    if show:
        plt.show()
    return fig, axes


def plot_pooled_maps(
    df: pd.DataFrame,
    group_label: str = "All participants",
    metrics: Sequence[str] = METRICS,
    scaling: str = "normalized",
    method: str = DEFAULT_METHOD,
    vmax: float | None = 1.0,
    figsize: tuple[float, float] = (13, 4),
    show: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
):
    """The plain view: pooled r for each metric, one panel each, with no tests.

    `vmax=1.0` (the default) fixes the colour scale at [-1, 1]; `None` scales it
    to the largest |r|. Saved through `save_output` as `corr_map_pooled`, with
    each panel's map as a table.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    axes = np.atleast_1d(axes)

    maps = {
        metric: pooled_corr_map(
            df, region_cols(metric, scaling), answer_cols(metric, scaling),
            method=method,
        )
        for metric in metrics
    }
    if vmax is None:
        vmax = _round_up(max(float(np.nanmax(np.abs(m.values))) for m in maps.values()))

    im = None
    for ax, metric in zip(axes, metrics):
        # No p-values here on purpose: a pooled p-value would treat every trial
        # as independent. The tests are in corr_map_stats.
        _, im = plot_corr_map(
            {"r": maps[metric]}, ax=ax, title=f"{scaling} {metric}", vmax=vmax
        )

    fig.colorbar(im, ax=axes, label=f"{method} r (pooled)", fraction=0.025)
    fig.suptitle(f"{group_label}: text regions vs. answers/question", y=1.05)

    save_output(
        fig,
        analysis="text_qa_relationship",
        plot="corr_map_pooled",
        tables={m: mat.reset_index(names="text_region") for m, mat in maps.items()},
        save=save,
        to_paper=to_paper,
        group=group_label,
        scaling=scaling,
        method=method,
    )

    if show:
        plt.show()
    return fig, axes


def plot_contrasts(
    table: pd.DataFrame,
    title: str = "",
    value: str = "delta_z",
    label_col: str | None = None,
    group_col: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (8, 4.5),
    xlabel: str | None = None,
    show: bool = True,
):
    """Forest plot of a comparison table: each contrast's estimate with its interval.

    Takes the output of `comparisons.compare_cells`, `compare_rows_within_column`
    or `compare_groups` -- anything with `ci_low` / `ci_high` next to `value`.

    A contrast that survives the multiple-comparison correction is drawn filled,
    one that does not hollow. The intervals are not corrected, so a hollow
    marker whose interval just clears 0 is the correction at work, not a
    contradiction.

    `label_col` names the rows; by default `cell_b` when every row shares one
    `cell_a`, else "cell_a vs cell_b", else "row ~ col". `group_col` draws one
    series per value of that column (e.g. "label"), offset vertically so they
    can be read against each other.
    """
    t = table.copy()
    if {"ci_low", "ci_high"} - set(t.columns):
        raise ValueError("table has no ci_low/ci_high; nothing to plot")

    if label_col is None:
        if "cell_a" in t.columns and "cell_b" in t.columns:
            if t["cell_a"].nunique() == 1:
                t["_label"] = pretty_labels(t["cell_b"])
            else:
                t["_label"] = [
                    f"{a}  vs  {b}"
                    for a, b in zip(pretty_labels(t["cell_a"]),
                                    pretty_labels(t["cell_b"]))
                ]
        elif {"row", "col"} <= set(t.columns):
            t["_label"] = [
                f"{r} ~ {c}"
                for r, c in zip(pretty_labels(t["row"]), pretty_labels(t["col"]))
            ]
        else:
            raise ValueError("cannot infer label_col; pass it explicitly")
    else:
        t["_label"] = pretty_labels(t[label_col])

    # One row order for every series: the largest mean `value` at the top.
    order = (
        t.groupby("_label")[value].mean().sort_values(ascending=True).index.tolist()
    )
    pos = {lab: i for i, lab in enumerate(order)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    series = [(None, t)] if group_col is None else list(t.groupby(group_col, sort=False))
    offsets = np.linspace(0.26, -0.26, len(series)) if len(series) > 1 else [0.0]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (name, sub), off, colour in zip(series, offsets, colours):
        y = [pos[lab] + off for lab in sub["_label"]]
        est = sub[value].values
        err = np.vstack([est - sub["ci_low"].values, sub["ci_high"].values - est])
        # Filled = significant after correction (`sig`). A table without `sig`
        # falls back to whether the uncorrected interval excludes 0.
        clears = (
            sub["sig"].astype(bool)
            if "sig" in sub.columns
            else (sub["ci_low"] > 0) | (sub["ci_high"] < 0)
        )
        ax.errorbar(
            est, y, xerr=err, fmt="none", ecolor=colour, elinewidth=1.4, capsize=3,
        )
        ax.scatter(
            est, y, s=34, zorder=3, label=name,
            facecolors=[colour if c else "white" for c in clears],
            edgecolors=colour, linewidths=1.4,
        )

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel or f"{value}  (95% CI)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.2)
    ax.margins(y=0.12)
    if group_col is not None:
        ax.legend(title=group_col, fontsize=8, title_fontsize=8)
    plt.tight_layout()

    save_output(
        ax.figure,
        analysis="text_qa_relationship",
        plot=plot,
        tables={"contrasts": t.drop(columns="_label", errors="ignore")},
        save=save,
        to_paper=to_paper,
        value=value,
    )

    if show:
        plt.show()
    return ax


def summarise_map(stats: dict) -> pd.DataFrame:
    """A map as a readable table: one row per cell, sorted by p_adj.

    Keeps whichever of n, r, the interval, d, t, p, p_adj, stars and the pooled
    r the dict has.
    """
    t = stats["table"].copy()
    t["row"] = pretty_labels(t["row"])
    t["col"] = pretty_labels(t["col"])
    keep = [
        c for c in
        ["row", "col", "n", "r", "ci_low", "ci_high", "d", "t", "p", "p_adj",
         "stars", "r_pooled"]
        if c in t.columns
    ]
    return t[keep].sort_values("p_adj").reset_index(drop=True)
