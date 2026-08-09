"""Small shared helpers: Fisher z, multiple-comparison adjustment, stars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

__all__ = ["fisher_z", "inv_fisher_z", "adjust_pvalues", "significance_stars"]

# Fisher z is undefined at |r| = 1; clip just inside.
_R_CLIP = 1 - 1e-9


def fisher_z(r):
    """arctanh(r), clipped so |r| = 1 does not become infinite."""
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -_R_CLIP, _R_CLIP))


def inv_fisher_z(z):
    """tanh(z) -- back from Fisher z to an r."""
    return np.tanh(np.asarray(z, dtype=float))


def adjust_pvalues(pvals, method: str | None = "fdr_bh") -> np.ndarray:
    """Multiple-comparison adjustment tolerant of NaNs. `method=None` -> unchanged.

    `method` is any name `statsmodels.stats.multitest.multipletests` accepts
    ("fdr_bh", "holm", "bonferroni", ...).
    """
    pvals = np.asarray(pvals, dtype=float)
    if method is None:
        return pvals
    out = pvals.copy()
    ok = ~np.isnan(pvals)
    if ok.any():
        out[ok] = multipletests(pvals[ok], method=method)[1]
    return out


def significance_stars(p: float) -> str:
    """Star notation matching `src.statistics.mixed_text_answer_effects`."""
    if pd.isna(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    if p < 1e-1:
        return "."
    return ""
