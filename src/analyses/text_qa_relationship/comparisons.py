"""
Testing correlations against each other rather than against 0.

`compare_cells` / `compare_rows_within_column`
    Two cells of the same map are measured on the same participants, so this is
    a paired t-test on each participant's difference in Fisher z. The pairing
    takes care of the dependence between the two correlations, including when
    they share a variable (which, with independent trials, would need a
    dedicated test such as Steiger's or Williams').

`compare_groups`
    Hunters and gatherers are different people, 180 in each group, so their z
    values are independent: a Welch two-sample t-test per cell.

All of these take the per-participant z values, `corr_map_stats(...)["z"]`.
The intervals they report are on the difference in Fisher z, not in r, and
are not corrected for multiple comparisons; the p-values are.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.analyses.text_qa_relationship._utils import (
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
    """Paired test of whether two cells of a map have the same correlation.

    `pairs` holds `((row_a, col_a), (row_b, col_b))` cell keys, which
    `columns.map_cell` builds. Each pair uses the participants with a value in
    both cells; a pair with fewer than 3 gets NaN.

    Returns one row per pair: `r_a` and `r_b` (each cell's mean z turned back
    into r) and their difference `delta_r`; `delta_z`, the mean difference in z,
    with its interval (`ci_low`, `ci_high`); t, p, Cohen's d; and p corrected
    across the pairs (`p_adj`), `sig` and stars. Sorted by `p_adj`.
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
    """Every pair of rows compared within each column of the map.

    E.g. "does critical -> answer_A differ from distractor -> answer_A?", for
    every column. With 3 rows and 5 columns that is 3 pairs x 5 columns = 15
    tests, corrected together.
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
    """Test whether each cell of the map differs between two groups of people.

    Per cell, a two-sample t-test on the participants' z values (Welch's, unless
    `equal_var=True`), with p corrected across the cells compared; a cell with
    fewer than 3 participants in either group gets NaN. Only cells in both
    `z_a` and `z_b` are compared, so passing frames cut down to some cells
    makes those cells the correction family (as `report.group_contrast(region=...)`
    does).

    Returns one row per cell, with each group's `n_<label>` and `r_<label>`.
    `delta_z` is group a minus group b, and `ci_low` / `ci_high` bound it, in
    Fisher z, not in r. Cohen's d uses the pooled SD of z.
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
