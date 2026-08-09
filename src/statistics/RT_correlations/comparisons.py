"""
Testing correlations against each other rather than against 0.

`compare_cells` / `compare_rows_within_column`
    Two cells of the same map are measured on the same participants, so this is
    a paired t-test on the within-participant difference of Fisher z. The
    pairing absorbs the dependence between the two correlations -- including the
    case where they share a variable, which under i.i.d. assumptions would need
    Steiger's or Williams' test.

`compare_groups`
    Hunters and gatherers are disjoint sets of 180 participants, so their z
    values are independent: a Welch two-sample t-test per cell.

All of these take the per-participant z frame produced by
`correlations.corr_map_stats(...)["z"]`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.statistics.RT_correlations._utils import (
    adjust_pvalues,
    inv_fisher_z,
    significance_stars,
)

__all__ = ["compare_cells", "compare_rows_within_column", "compare_groups"]


def compare_cells(
    z: pd.DataFrame,
    pairs: Iterable[tuple[tuple[str, str], tuple[str, str]]],
    adjust: str | None = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired test of H0: two map cells have the same correlation.

    Parameters
    ----------
    z : per-participant Fisher z, from `corr_map_stats(...)["z"]`.
    pairs : iterable of ((row_a, col_a), (row_b, col_b)); build the cell keys
        with `columns.map_cell`.

    Returns
    -------
    DataFrame with r_a, r_b, their difference, the CI on the *z* difference,
    t, p, adjusted p and stars, sorted by adjusted p.
    """
    rows = []
    for cell_a, cell_b in pairs:
        both = z[[tuple(cell_a), tuple(cell_b)]].dropna()
        a = both.iloc[:, 0].values
        b = both.iloc[:, 1].values
        n = len(both)
        rec = {
            "cell_a": f"{cell_a[0]} ~ {cell_a[1]}",
            "cell_b": f"{cell_b[0]} ~ {cell_b[1]}",
            "n": n,
        }
        if n < 3:
            rec.update(dict.fromkeys(
                ["r_a", "r_b", "delta_r", "delta_z", "ci_low", "ci_high",
                 "t", "p", "d"], np.nan))
            rows.append(rec)
            continue

        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        sd = diff.std(ddof=1)
        se = sd / np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / 2, n - 1)
        rec.update(
            r_a=float(inv_fisher_z(a.mean())),
            r_b=float(inv_fisher_z(b.mean())),
            delta_r=float(inv_fisher_z(a.mean()) - inv_fisher_z(b.mean())),
            delta_z=float(diff.mean()),
            # CI is on the z difference, not the r difference.
            ci_low=float(diff.mean() - crit * se),
            ci_high=float(diff.mean() + crit * se),
            t=float(t_stat),
            p=float(p_val),
            d=float(diff.mean() / sd) if sd > 0 else np.nan,
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["p_adj"] = adjust_pvalues(out["p"].values, adjust)
    out["sig"] = out["p_adj"] < alpha
    out["stars"] = out["p_adj"].apply(significance_stars)
    return out.sort_values("p_adj").reset_index(drop=True)


def compare_rows_within_column(
    z: pd.DataFrame,
    row_cols: Sequence[str],
    col_cols: Sequence[str],
    adjust: str | None = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All row-vs-row comparisons within each column of the map.

    E.g. "is critical -> answer_A stronger than distractor -> answer_A?", for
    every answer column: 3 regions x 5 columns = 15 tests. 
    """
    pairs = [
        ((row_cols[i], c), (row_cols[j], c))
        for c in col_cols
        for i in range(len(row_cols))
        for j in range(i + 1, len(row_cols))
    ]
    return compare_cells(z, pairs, adjust=adjust, alpha=alpha)


def compare_groups(
    z_a: pd.DataFrame,
    z_b: pd.DataFrame,
    label_a: str = "group_a",
    label_b: str = "group_b",
    adjust: str | None = "fdr_bh",
    alpha: float = 0.05,
    equal_var: bool = False,
) -> pd.DataFrame:
    """Between-subjects test of H0: a map cell is equal in both groups.

    Welch two-sample t-test on the per-participant z per cell, adjusted across
    the cells of the map. Pass `z_a`/`z_b` already restricted to a subset of
    cells to make that subset the correction family (see
    `report.group_contrast(region=...)`).

    `ci_low`/`ci_high` bound the *z* difference, not the r difference.
    """
    cells = [c for c in z_a.columns if c in set(z_b.columns)]
    rows = []
    for cell in cells:
        a = z_a[cell].dropna().values
        b = z_b[cell].dropna().values
        rec = {
            "row": cell[0],
            "col": cell[1],
            f"n_{label_a}": len(a),
            f"n_{label_b}": len(b),
        }
        if len(a) < 3 or len(b) < 3:
            rec.update(dict.fromkeys(
                [f"r_{label_a}", f"r_{label_b}", "delta_r", "delta_z",
                 "ci_low", "ci_high", "t", "p", "d"], np.nan))
            rows.append(rec)
            continue

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var)
        r_a = float(inv_fisher_z(a.mean()))
        r_b = float(inv_fisher_z(b.mean()))

        # CI on the z difference, matching the test: Welch unless equal_var.
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        if equal_var:
            df_pooled = len(a) + len(b) - 2
            pooled = (
                (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
            ) / df_pooled
            se = np.sqrt(pooled * (1 / len(a) + 1 / len(b)))
            dof = df_pooled
        else:
            se = np.sqrt(va + vb)
            dof = (va + vb) ** 2 / (
                va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)
            )
        crit = stats.t.ppf(1 - alpha / 2, dof)
        delta_z = float(a.mean() - b.mean())

        # Cohen's d on the pooled SD of z.
        sd_pooled = np.sqrt(
            ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
            / (len(a) + len(b) - 2)
        )
        rec.update(**{
            f"r_{label_a}": r_a,
            f"r_{label_b}": r_b,
            "delta_r": r_a - r_b,
            "delta_z": delta_z,
            "ci_low": float(delta_z - crit * se),
            "ci_high": float(delta_z + crit * se),
            "t": float(t_stat),
            "p": float(p_val),
            "d": float(delta_z / sd_pooled) if sd_pooled > 0 else np.nan,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["p_adj"] = adjust_pvalues(out["p"].values, adjust)
    out["sig"] = out["p_adj"] < alpha
    out["stars"] = out["p_adj"].apply(significance_stars)
    return out
