# checks.py
"""
Invariant checks for joins the pipeline assumes are total.

`docs/todo.md` T3.17. The pipeline is full of `how="left"` merges whose right
side is built from the same trial set as the left, so every key must match.
When one does not, a left join does not fail -- it writes NaN, and the model
then fills NaN with 0.0. A dropped trial therefore arrives at the model as a
trial that behaved like the population mean, which is a plausible-looking
wrong number rather than a crash. That is exactly the case
`docs/conventions.md` says to assert on.

Deliberately *not* a general-purpose validation layer: these raise on the two
failure modes that are silent, and on nothing else.

The planned home for this module is `lib/checks.py` (`docs/restructure-map.md`
§3); it sits at `src/` top level until Stage B creates that package.
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
    """
    Raise unless `right` can be left-joined onto `left` without losing or
    duplicating a row.

    Two things are checked, because a left join hides both:

    1. **`right`'s keys are unique.** Duplicates do not drop rows, they
       *multiply* them -- the frame silently stops being one row per trial,
       and every downstream count is wrong by the duplication factor.
    2. **`right` covers every key in `left`.** A missing key becomes a row of
       NaN, indistinguishable from a measured absence.

    Checked before the merge rather than after, because afterwards an
    unmatched key and a matched-but-NaN value look the same.

    `name` names the feature block, so the error says which of the nine
    merges failed rather than only that one did.
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
