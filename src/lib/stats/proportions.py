"""Confidence intervals for proportions, and significance stars.

This is the project's only copy of `wilson_ci`; use it rather than writing
another.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Confidence interval for a proportion k/n, by the Wilson method.

    The Wilson interval behaves well near 0 and 1 and for small n, where the
    simpler "p +- 1.96 * se" interval goes wrong. `z` sets the confidence level;
    the default 1.96 gives 95%. Returns (nan, nan) when n is 0.
    """
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def p_to_stars(p: Optional[float]) -> str:
    """Significance stars for a p-value: *** < .001, ** < .01, * < .05, else "n.s.".

    Returns "n/a" when there is no p-value.
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
