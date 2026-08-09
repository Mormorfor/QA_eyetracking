"""
One-call helpers, so the notebook is calls and displays and nothing else.

Typical use::

    from src.statistics.RT_correlations import (
        load_all_features, metric_map_stats, plot_maps, summarise_map,
        region_contrasts,
    )

    df = load_all_features()
    stats = metric_map_stats(df)                 # {"RT": {...}, "TFD": {...}}
    plot_maps(stats, "All participants")
    summarise_map(stats["RT"])
    region_contrasts(stats, "RT")
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.statistics.RT_correlations.bootstrap import cluster_bootstrap_corr_map
from src.statistics.RT_correlations.columns import (
    ANSWERS,
    METRICS,
    answer_col,
    answer_cols,
    region_col,
    region_cols,
)
from src.statistics.RT_correlations.comparisons import (
    compare_cells,
    compare_groups,
    compare_rows_within_column,
)
from src.statistics.RT_correlations.correlations import DEFAULT_METHOD, corr_map_stats
from src.statistics.RT_correlations.plots import plot_corr_map_pair

__all__ = [
    "metric_map_stats",
    "group_map_stats",
    "plot_maps",
    "region_contrasts",
    "reference_contrasts",
    "group_contrast",
    "bootstrap_maps",
    "group_bootstrap_maps",
    "plot_bootstrap_maps",
]


def metric_map_stats(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
    scaling: str = "normalized",
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict[str, dict]:
    """Participant-level correlation-map stats for each metric of one frame.

    Returns `{"RT": corr_map_stats(...), "TFD": corr_map_stats(...)}`; extra
    kwargs go through to `corr_map_stats` (`adjust`, `alpha`, `min_trials`, ...).
    """
    return {
        metric: corr_map_stats(
            df,
            region_cols(metric, scaling),
            answer_cols(metric, scaling),
            method=method,
            **kwargs,
        )
        for metric in metrics
    }


def group_map_stats(
    group_dfs: dict[str, pd.DataFrame],
    metrics: Sequence[str] = METRICS,
    scaling: str = "normalized",
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict[str, dict[str, dict]]:
    """`metric_map_stats` for each group, e.g. hunters and gatherers."""
    return {
        name: metric_map_stats(
            frame, metrics=metrics, scaling=scaling, method=method, **kwargs
        )
        for name, frame in group_dfs.items()
    }


def plot_maps(
    stats_by_metric: dict[str, dict],
    label: str = "All participants",
    scaling: str = "normalized",
    **kwargs,
):
    """Side-by-side annotated maps for the metrics of one group."""
    method = next(iter(stats_by_metric.values())).get("method", DEFAULT_METHOD)
    return plot_corr_map_pair(
        {f"{scaling} {metric}": st for metric, st in stats_by_metric.items()},
        suptitle=(
            f"{label} - text regions vs. answers/question "
            f"({method}, participant-level)"
        ),
        **kwargs,
    )


def bootstrap_maps(
    df: pd.DataFrame,
    metrics: Sequence[str] = METRICS,
    scaling: str = "normalized",
    method: str = DEFAULT_METHOD,
    n_boot: int = 1000,
    **kwargs,
) -> dict[str, dict]:
    """Cluster-bootstrapped pooled maps for each metric of one frame.

    About 30 s for both metrics at `n_boot=1000`; drop `n_boot` for a quick look,
    at the cost of p-value resolution (see `cluster_bootstrap_corr_map`).
    """
    return {
        metric: cluster_bootstrap_corr_map(
            df,
            region_cols(metric, scaling),
            answer_cols(metric, scaling),
            method=method,
            n_boot=n_boot,
            **kwargs,
        )
        for metric in metrics
    }


def group_bootstrap_maps(
    group_dfs: dict[str, pd.DataFrame],
    metrics: Sequence[str] = METRICS,
    scaling: str = "normalized",
    method: str = DEFAULT_METHOD,
    n_boot: int = 1000,
    **kwargs,
) -> dict[str, dict[str, dict]]:
    """`bootstrap_maps` for each group, e.g. hunters and gatherers.

    Each group has half the participants, so its intervals are roughly sqrt(2)
    wider than the all-participants ones. For testing whether a cell *differs*
    between the groups, use `group_contrast` rather than comparing these
    intervals by eye.
    """
    return {
        name: bootstrap_maps(
            frame, metrics=metrics, scaling=scaling, method=method,
            n_boot=n_boot, **kwargs,
        )
        for name, frame in group_dfs.items()
    }


def plot_bootstrap_maps(
    boot_by_metric: dict[str, dict],
    label: str = "All participants",
    scaling: str = "normalized",
    **kwargs,
):
    """Side-by-side bootstrap maps, labelled for the pooled estimand.

    Same layout as `plot_maps`, but the cells are pooled r with percentile CIs,
    not within-participant r -- hence the separate footnote.
    """
    first = next(iter(boot_by_metric.values()))
    method = first.get("method", DEFAULT_METHOD)
    n_boot = first.get("n_boot")
    kwargs.setdefault("show_ci", True)
    kwargs.setdefault(
        "footnote",
        f"cell = pooled r [95% percentile CI over {n_boot} participant resamples]; "
        f"stars = BH-adjusted bootstrap p (floored at 1/{n_boot}); "
        "hatched = not significant",
    )
    return plot_corr_map_pair(
        {f"{scaling} {metric}": b for metric, b in boot_by_metric.items()},
        suptitle=(
            f"{label} - pooled r with participant-cluster bootstrap ({method})"
        ),
        **kwargs,
    )


def region_contrasts(
    stats_by_metric: dict[str, dict],
    metric: str = "RT",
    scaling: str = "normalized",
    **kwargs,
) -> pd.DataFrame:
    """Region-vs-region comparisons within each answer column, for one metric."""
    return compare_rows_within_column(
        stats_by_metric[metric]["z"],
        region_cols(metric, scaling),
        answer_cols(metric, scaling),
        **kwargs,
    )


def group_contrast(
    stats_by_group: dict[str, dict[str, dict]],
    metric: str = "RT",
    label_a: str = "hunters",
    label_b: str = "gatherers",
    region: str | None = None,
    scaling: str = "normalized",
    **kwargs,
) -> pd.DataFrame:
    """Between-group comparison of the cells of one metric's map.

    `region` restricts the comparison to one row (e.g. `"critical"`), which also
    makes that row the multiple-comparison family: 5 tests instead of 15. Do
    that when the row is the question you came with, not after seeing which
    cells looked interesting.
    """
    z_a = stats_by_group[label_a][metric]["z"]
    z_b = stats_by_group[label_b][metric]["z"]

    if region is not None:
        row = region_col(region, metric, scaling)
        keep = [c for c in z_a.columns if c[0] == row]
        if not keep:
            raise KeyError(f"no cells for region {region!r} in the {metric} map")
        z_a, z_b = z_a[keep], z_b[keep]

    return compare_groups(z_a, z_b, label_a, label_b, **kwargs)


def reference_contrasts(
    stats_by_label: dict[str, dict[str, dict]],
    region: str = "critical",
    answer: str = "answer_A",
    metric: str = "RT",
    scaling: str = "normalized",
    others: Sequence[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """One cell of the map against the other answers in its own row.

    Answers "is `region` -> `answer` stronger than `region` -> anything else?",
    for each labelled set of stats -- e.g. `{"all": ..., "hunters": ...,
    "gatherers": ...}`. Paired t-test on the within-participant z differences,
    exactly as `compare_cells`.

    `others` defaults to every answer column except `answer`. Correction is
    applied within each label separately: each is an independent sample asked
    the same question, so they are separate families.

    Returns one long table with a `label` column, sorted by label then by the
    size of the difference.
    """
    reference = (
        region_col(region, metric, scaling),
        answer_col(answer, metric, scaling),
    )
    other_answers = [a for a in (others or ANSWERS) if a != answer]
    pairs = [
        (reference, (reference[0], answer_col(a, metric, scaling)))
        for a in other_answers
    ]

    out = []
    for label, stats_by_metric in stats_by_label.items():
        table = compare_cells(stats_by_metric[metric]["z"], pairs, **kwargs)
        out.append(table.assign(label=label))

    combined = pd.concat(out, ignore_index=True)
    cols = ["label"] + [c for c in combined.columns if c != "label"]
    return combined[cols].sort_values(
        ["label", "delta_z"], ascending=[True, False]
    ).reset_index(drop=True)
