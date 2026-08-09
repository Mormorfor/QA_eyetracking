"""
The correlation maps, and the test of each cell against 0.

Two estimands, and they are not the same number:

`pooled_corr_map`
    plain `df.corr()` over all trials. Mixes within- and between-participant
    variance -- part of it is just "some people read slowly everywhere" -- and
    its textbook p-value assumes ~19k independent rows when there are only 360
    participants.

`corr_map_stats`
    correlate within each participant, Fisher-z the result, and test the 360 z
    values against 0 with a one-sample t-test. This is the within-participant
    association, and here it runs roughly half the pooled value.

Reading times are per-word RTs with skew up to ~40, so `method="spearman"` is
the default; Pearson on these columns is driven by a handful of outlier trials.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src import constants as C
from src.statistics.RT_correlations._utils import (
    adjust_pvalues,
    fisher_z,
    inv_fisher_z,
    significance_stars,
)

__all__ = ["pooled_corr_map", "fisher_z_by_cluster", "corr_map_stats"]

DEFAULT_METHOD = "spearman"


def pooled_corr_map(
    df: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    method: str = DEFAULT_METHOD,
) -> pd.DataFrame:
    """Cross-correlation block (rows x cols) pooled over all trials."""
    cols = list(dict.fromkeys([*row_cols, *col_cols]))
    return df[cols].corr(method=method).loc[list(row_cols), list(col_cols)]


# https://blogs.sas.com/content/iml/2017/09/20/fishers-transformation-correlation.html
def fisher_z_by_cluster(
    df: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    cluster_col: str = C.PARTICIPANT_ID,
    method: str = DEFAULT_METHOD,
    min_trials: int = 20,
) -> pd.DataFrame:
    """Fisher-z transformed within-participant correlation for every map cell.

    Returns
    -------
    DataFrame
        index   = cluster id (participant)
        columns = MultiIndex (row_col, col_col), one per heatmap cell
        values  = arctanh(r) computed within that participant's trials

    Participants with fewer than `min_trials` trials, or with no variance in one
    of the two columns, get NaN for the affected cells; the tests downstream drop
    them pairwise.
    """
    cols = list(dict.fromkeys([*row_cols, *col_cols]))
    sub = df[[cluster_col, *cols]].copy()

    sizes = sub.groupby(cluster_col)[cols[0]].size()
    sub = sub[sub[cluster_col].isin(sizes.index[sizes >= min_trials])]

    # One correlation matrix per participant, then slice out the cross-block.
    per_cluster = sub.groupby(cluster_col)[cols].corr(method=method)

    records = {}
    for r_col in row_cols:
        block = per_cluster.xs(r_col, level=1)[list(col_cols)]
        for c_col in col_cols:
            records[(r_col, c_col)] = fisher_z(block[c_col])

    z = pd.DataFrame(records)
    z.columns = pd.MultiIndex.from_tuples(z.columns, names=["row", "col"])
    z.index.name = cluster_col
    return z.replace([np.inf, -np.inf], np.nan)


def corr_map_stats(
    df: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    cluster_col: str = C.PARTICIPANT_ID,
    method: str = DEFAULT_METHOD,
    min_trials: int = 20,
    adjust: str | None = "fdr_bh",
    alpha: float = 0.05,
    add_wilcoxon: bool = True,
) -> dict:
    """Per-cell test of H0: correlation = 0, with the participant as the unit.

    Parameters
    ----------
    row_cols, col_cols : columns forming the rows / columns of the map.
    method : "spearman" (default, robust to the RT skew) or "pearson".
    adjust : any `statsmodels.stats.multitest.multipletests` method, or None.
        Applied across all `len(row_cols) * len(col_cols)` cells of the map.
    add_wilcoxon : also run a signed-rank test on the z values, as a check that
        the t-test is not being carried by a skewed z distribution.

    Returns
    -------
    dict with
        "r"           : DataFrame, tanh(mean z) -- the participant-level estimate
        "r_pooled"    : DataFrame, the pooled r for comparison
        "ci_low/high" : DataFrame, 95% CI on r (t-based on z, back-transformed)
        "t","p","p_adj","n","d" : DataFrames
        "sig"         : boolean DataFrame, p_adj < alpha
        "table"       : tidy long-format DataFrame of all of the above
        "z"           : per-participant z values, to feed `comparisons`
    """
    z = fisher_z_by_cluster(
        df,
        row_cols,
        col_cols,
        cluster_col=cluster_col,
        method=method,
        min_trials=min_trials,
    )
    r_pooled = pooled_corr_map(df, row_cols, col_cols, method=method)

    rows = []
    for r_col in row_cols:
        for c_col in col_cols:
            vals = z[(r_col, c_col)].dropna().values
            n = len(vals)
            rec = {"row": r_col, "col": c_col, "n": n}
            if n < 3:
                rec.update(
                    dict.fromkeys(
                        ["r", "t", "p", "ci_low", "ci_high", "d", "p_wilcoxon"], np.nan
                    )
                )
                rows.append(rec)
                continue

            t_stat, p_val = stats.ttest_1samp(vals, 0.0)
            sd = vals.std(ddof=1)
            se = sd / np.sqrt(n)
            crit = stats.t.ppf(1 - alpha / 2, n - 1)
            mean_z = vals.mean()
            rec.update(
                r=float(inv_fisher_z(mean_z)),
                t=float(t_stat),
                p=float(p_val),
                ci_low=float(inv_fisher_z(mean_z - crit * se)),
                ci_high=float(inv_fisher_z(mean_z + crit * se)),
                # Cohen's d for the one-sample test on z.
                d=float(mean_z / sd) if sd > 0 else np.nan,
            )
            if add_wilcoxon:
                rec["p_wilcoxon"] = float(stats.wilcoxon(vals).pvalue)
            rows.append(rec)

    table = pd.DataFrame(rows)
    table["p_adj"] = adjust_pvalues(table["p"].values, adjust)
    if add_wilcoxon:
        table["p_wilcoxon_adj"] = adjust_pvalues(table["p_wilcoxon"].values, adjust)
    table["sig"] = table["p_adj"] < alpha
    table["stars"] = table["p_adj"].apply(significance_stars)
    table["r_pooled"] = [r_pooled.loc[r, c] for r, c in zip(table["row"], table["col"])]

    def _mat(field):
        return table.pivot(index="row", columns="col", values=field).loc[
            list(row_cols), list(col_cols)
        ]

    return {
        "r": _mat("r"),
        "r_pooled": r_pooled,
        "ci_low": _mat("ci_low"),
        "ci_high": _mat("ci_high"),
        "t": _mat("t"),
        "d": _mat("d"),
        "p": _mat("p"),
        "p_adj": _mat("p_adj"),
        "n": _mat("n"),
        "sig": _mat("sig").astype(bool),
        "table": table,
        "z": z,
        "method": method,
        "adjust": adjust,
        "alpha": alpha,
    }
