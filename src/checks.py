"""A check for joins that must match every trial.

The pipeline joins many tables onto the trial table with left joins, and each
of them is built from the same trials, so every trial should match. When one
does not, a left join does not fail: it fills in NaN, the model then turns that
NaN into 0, and the trial looks like an average one. That is a wrong number
that looks right, so these joins are checked instead of trusted.
"""

from typing import Sequence

import pandas as pd


def assert_full_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    name: str,
    max_reported: int = 5,
) -> None:
    """Stop with an error unless `right` can be left-joined onto `left` without
    losing or repeating a row.

    Two checks, because a left join hides both problems:

    1. **Each key appears once in `right`.** A repeated key would repeat rows,
       so the table would stop being one row per trial.
    2. **Every key in `left` is in `right`.** A missing key would become a row
       of NaN that looks like a real absence.

    Checked before joining, because afterwards a missing key and a real NaN
    look the same. `name` says which block failed.
    """
    keys = list(keys)

    dup_mask = right.duplicated(subset=keys, keep=False)
    if dup_mask.any():
        dups = right.loc[dup_mask, keys].drop_duplicates()
        raise ValueError(
            f"{name}: right side is not one row per {tuple(keys)} -- "
            f"{len(dups)} key(s) repeat, so a left join would multiply rows. "
            f"First {min(max_reported, len(dups))}: "
            f"{dups.head(max_reported).to_dict('records')}"
        )

    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    missing = left_keys - right_keys
    if missing:
        raise ValueError(
            f"{name}: {len(missing)} of {len(left_keys)} trial(s) are absent "
            f"from this feature block and would silently become NaN. Every "
            f"block is built from the same trial set, so this is a pipeline "
            f"fault, not missing data -- do not impute it. "
            f"First {min(max_reported, len(missing))}: "
            f"{sorted(missing)[:max_reported]}"
        )
