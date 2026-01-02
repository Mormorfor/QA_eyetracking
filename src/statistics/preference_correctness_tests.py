from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src import constants as C


def correctness_by_pref_group_test(
    df: pd.DataFrame,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """
    Fisher exact test comparing correctness between matching and not_matching.

    Returns dict with:
      - contingency_table (2x2)
      - odds_ratio
      - p_value
      - counts
    """
    sub = df[[pref_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)

    a = int(((sub[pref_col] == "matching") & (sub[correct_col] == 1)).sum())
    b = int(((sub[pref_col] == "matching") & (sub[correct_col] == 0)).sum())
    c = int(((sub[pref_col] == "not_matching") & (sub[correct_col] == 1)).sum())
    d = int(((sub[pref_col] == "not_matching") & (sub[correct_col] == 0)).sum())

    table = np.array([[a, b],
                      [c, d]], dtype=int)

    odds_ratio, p_value = fisher_exact(table)

    return {
        "contingency_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "counts": {
            "matching": {"correct": a, "incorrect": b},
            "not_matching": {"correct": c, "incorrect": d},
        },
    }
