"""Binomial proportions and significance notation.

`wilson_ci` came from derived/correctness_measures.py, where it was the real
one; three nested near-copies of it elsewhere were deleted in T1.5. One home
now, so the next caller finds it instead of writing a fourth.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def p_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
