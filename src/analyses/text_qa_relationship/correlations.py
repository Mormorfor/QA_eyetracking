"""
The correlation maps, and the test of each cell against 0.

Two different quantities, which give different numbers:

`pooled_corr_map`
    plain `df.corr()` over all trials. It mixes differences within a person
    with differences between people -- part of it is just "some people read
    slowly everywhere" -- and its textbook p-value would treat ~19k trials as
    independent when there are only 360 participants.

`corr_map_stats`
    correlate within each participant, turn each r into a Fisher z, and test
    the 360 z values against 0 with a one-sample t-test. This is the
    association within a person. On Study 1 it is about a third of the pooled
    value.

Reading times are strongly right-skewed, so Spearman (`method="spearman"`) is
the default; Pearson on these columns is driven by a handful of outlier trials.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.config import columns as C
from src.analyses.text_qa_relationship._utils import (
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
    """The map over all trials together: `row_cols` against `col_cols`."""
    cols = list(dict.fromkeys([*row_cols, *col_cols]))
    return df[cols].corr(method=method).loc[list(row_cols), list(col_cols)]


# Fisher's z-transformation of a correlation, explained:
# https://blogs.sas.com/content/iml/2017/09/20/fishers-transformation-correlation.html
def fisher_z_by_cluster(
    df: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    cluster_col: str = C.PARTICIPANT_ID,
    method: str = DEFAULT_METHOD,
    min_trials: int = 20,
) -> pd.DataFrame:
    """Each participant's correlation in every cell of the map, as a Fisher z.

    Returns one row per participant (`cluster_col`) and one column per cell,
    keyed `(row_col, col_col)`; each value is arctanh(r) over that participant's
    trials.

    Participants with fewer than `min_trials` trials are left out. A participant
    with no variation in one of a cell's two columns (e.g. one who never looked
    at that answer) gets NaN in that cell, and the tests leave them out of that
    cell only.
    """
    cols = list(dict.fromkeys([*row_cols, *col_cols]))
    sub = df[[cluster_col, *cols]].copy()

    sizes = sub.groupby(cluster_col)[cols[0]].size()
    sub = sub[sub[cluster_col].isin(sizes.index[sizes >= min_trials])]

    # One correlation matrix per participant; only its rows-by-columns part is kept.
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
    """Test every cell of the map against 0, with the participant as the unit.

    Per cell: each participant's Fisher z (`fisher_z_by_cluster`), then a
    one-sample t-test of those z values against 0. A cell with fewer than 3
    participants gets NaN throughout. `adjust` corrects the p-values over all
    cells of the map together (Benjamini-Hochberg by default; None for no
    correction). `add_wilcoxon` also runs a signed-rank test on the same z
    values, to check that the t-test is not carried by a skewed z distribution.

    Returns a dict. These are frames shaped like the map:
        "r"                 tanh(mean z): the participant-level correlation
        "r_pooled"          the pooled r, for comparison (`pooled_corr_map`)
        "ci_low", "ci_high" the (1 - alpha) interval: a t-interval on the mean z,
                            turned back into r
        "t", "p", "p_adj"   the t-test, and p after correction
        "n"                 participants with a value in the cell
        "d"                 Cohen's d: mean z / SD of z
        "sig"               p_adj < alpha
    and also
        "table"             all of the above in long form, one row per cell,
                            with the stars (and the Wilcoxon p-values)
        "z"                 the per-participant z values, for `comparisons`
        "method", "adjust", "alpha"   the settings used
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
