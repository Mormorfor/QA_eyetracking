"""
Cluster bootstrap of the pooled correlation map.

This is the robustness check next to `correlations.corr_map_stats`: it keeps the
*pooled* r that the plain heatmaps show as the estimand, and fixes only its
standard error, by resampling participants (not trials) with replacement.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src import constants as C
from src.statistics.RT_correlations._utils import adjust_pvalues, significance_stars
from src.statistics.RT_correlations.correlations import DEFAULT_METHOD

__all__ = ["cluster_bootstrap_corr_map"]


def cluster_bootstrap_corr_map(
    df: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    cluster_col: str = C.PARTICIPANT_ID,
    method: str = DEFAULT_METHOD,
    n_boot: int = 1000,
    alpha: float = 0.05,
    adjust: str | None = "fdr_bh",
    random_state: int | None = 0,
) -> dict:
    """Percentile CIs for the pooled r, resampling participants.

    The two-sided p is the bootstrap-percentile p: 2 * min(share of resamples
    below 0, share above 0), floored at 1 / n_boot. Its resolution is therefore
    capped by `n_boot` -- at n_boot = 1000 no cell can go below p = 0.001, so
    the strongest cells all report the same p no matter how strong they are.

    Returns a dict shaped like `corr_map_stats` where they overlap ("r",
    "ci_low", "ci_high", "p", "p_adj", "sig", "table"), so `plots.plot_corr_map`
    and `plots.summarise_map` accept it directly. Also returns the raw `boots`
    array (n_boot x rows x cols).
    """
    rng = np.random.default_rng(random_state)
    cols = list(dict.fromkeys([*row_cols, *col_cols]))
    sub = df[[cluster_col, *cols]].dropna(subset=cols)

    idx_by_cluster = [
        np.asarray(v) for v in sub.groupby(cluster_col).indices.values()
    ]
    n_clusters = len(idx_by_cluster)
    values = sub[cols].values

    row_pos = [cols.index(c) for c in row_cols]
    col_pos = [cols.index(c) for c in col_cols]
    boots = np.empty((n_boot, len(row_cols), len(col_cols)), dtype=float)

    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([idx_by_cluster[i] for i in pick])
        sample = pd.DataFrame(values[idx], columns=cols)
        cmat = sample.corr(method=method).values
        boots[b] = cmat[np.ix_(row_pos, col_pos)]

    point = sub[cols].corr(method=method).loc[list(row_cols), list(col_cols)]
    lo = np.nanpercentile(boots, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2), axis=0)

    below = (boots <= 0).mean(axis=0)
    above = (boots >= 0).mean(axis=0)
    p = np.clip(2 * np.minimum(below, above), 1.0 / n_boot, 1.0)
    p_adj = adjust_pvalues(p.ravel(), adjust).reshape(p.shape)

    def frame(a):
        return pd.DataFrame(a, index=list(row_cols), columns=list(col_cols))

    out = {
        "r": point,
        "ci_low": frame(lo),
        "ci_high": frame(hi),
        "p": frame(p),
        "p_adj": frame(p_adj),
        "sig": frame(p_adj < alpha),
        "boots": boots,
        "n_clusters": n_clusters,
        "n_boot": n_boot,
        "method": method,
        "alpha": alpha,
    }

    # Tidy form, so `plots.summarise_map` works on a bootstrap result too.
    table = pd.concat(
        {
            field: out[field].stack()
            for field in ["r", "ci_low", "ci_high", "p", "p_adj"]
        },
        axis=1,
    ).reset_index(names=["row", "col"])
    table["n"] = n_clusters
    table["sig"] = table["p_adj"] < alpha
    table["stars"] = table["p_adj"].apply(significance_stars)
    out["table"] = table
    return out
