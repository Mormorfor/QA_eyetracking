"""Did the participant choose the answer their eyes favoured? As model features, without the
answer key.

The correctness-associations analysis (`features/preference.py::compute_trial_matching`)
calls a trial "matching" when the answer the participant chose is the one that stands out on
a measure. Which answer was chosen is observed and whether it was right is not, so the idea
needs no answer key, and here it becomes features on the answer-hidden table: one 0/1 column
per measure and way of standing out.

* `<measure>__matching_polarity` -- the chosen answer has the highest value of the four, or
  the lowest where a low value means more attention (skip rate, smallest pupil size). The
  direction of each measure is `columns.PREF_SPECS`.
* `<measure>__matching_relative` -- the chosen answer is the one furthest from the four
  answers' mean, in either direction.

The measures are the ten per-area measures of the paper's model (`PREF_SPECS`).

**A tie counts as matching** when the chosen answer is among the tied ones, the same rule as
`compute_trial_matching`. Breaking the tie by letter would leak the answer key, since
`answer_A` is always the correct answer. Ties are common on the counted measures -- on
Study 1, `num_label_visits` ties at its top on 45% of trials and `skip_rate` at its bottom
on 18%.

Answers with no value (pupil size and first-fixation duration on an answer never fixated)
are left out of the comparison, as in `compute_trial_matching`: a chosen answer with no value
is not matching, and a trial where none of the four has a value has none either (28 trials
on Study 1, 1 on Study 2). The model fills those with 0 and counts them, as it does every
missing pupil and first-fixation value.

The columns are worked out from the saved answer-hidden table when it is loaded
(`with_matching`), not saved in it: they take a moment, and adding them to the saved table
would change its fingerprint and so every result cached from it.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import REGIME_COL, REGIME_ORDER
from src.explorations.answer_hidden.deliberation import (
    BASE_NAME,
    CHOSEN_GROUPS,
    base_cols,
    plot_search_heatmap,
    run_deliberation_search,
    selected_feature_sets,
)
from src.explorations.answer_hidden.run import load_tables
from src.explorations.answer_hidden.table import POSITION_AREAS, SEP

MODES = ("polarity", "relative")


def matching_col(measure: str, mode: str) -> str:
    """The column saying whether the chosen answer stood out on `measure`, the `mode` way."""
    return f"{measure}{SEP}matching_{mode}"


# One group per way of standing out, each with all ten measures.
MATCHING_GROUPS: Dict[str, List[str]] = {
    f"matching_{mode}": [matching_col(m, mode) for m, _ in Con.PREF_SPECS] for mode in MODES
}

# What the matching groups are added to: nothing, the base, the chosen set
# (`deliberation.CHOSEN_GROUPS`), and the two together.
CHOSEN_KEY = "+".join(CHOSEN_GROUPS)
MATCHING_ANCHORS = ("", BASE_NAME, CHOSEN_KEY, f"{BASE_NAME}+{CHOSEN_KEY}")


def add_matching_features(table: pd.DataFrame) -> pd.DataFrame:
    """`table`, an answer-hidden table, with the twenty matching columns (`MATCHING_GROUPS`)
    added, worked out from each measure's four positions and its value on the chosen answer.

    Stops unless the chosen answer's value is one of the four positions' on every trial
    where it has one, since the comparison rests on the two agreeing.
    """
    out = {}
    for measure, direction in Con.PREF_SPECS:
        values = table[[f"{measure}{SEP}{p}" for p in POSITION_AREAS]].to_numpy(float)
        chosen = table[f"{measure}{SEP}selected"].to_numpy(float)
        missing = np.isnan(values)
        none = missing.all(axis=1)
        has_value = ~np.isnan(chosen)
        found = (values[has_value] == chosen[has_value, None]).any(axis=1)
        assert found.all(), (
            f"{measure}: on {int((~found).sum())} trial(s) the selected answer's value is not "
            f"one of the four positions' values."
        )

        # The extreme of the answers that have a value; a missing chosen value compares
        # unequal to it, so is not matching.
        if direction == "high":
            extreme = np.where(missing, -np.inf, values).max(axis=1)
        else:
            extreme = np.where(missing, np.inf, values).min(axis=1)
        polarity = (chosen == extreme).astype(float)

        # The furthest from the mean of the answers that have a value. The chosen answer's
        # distance is computed exactly as its position's is, so a tie compares equal.
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(missing, 0.0, values).sum(axis=1) / (~missing).sum(axis=1)
        distance = np.abs(values - mean[:, None])
        furthest = np.where(missing, -np.inf, distance).max(axis=1)
        relative = (np.abs(chosen - mean) == furthest).astype(float)

        polarity[none] = np.nan
        relative[none] = np.nan
        out[matching_col(measure, "polarity")] = polarity
        out[matching_col(measure, "relative")] = relative
    return pd.concat([table, pd.DataFrame(out, index=table.index)], axis=1)


def with_matching(tables: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """`tables` (`run.load_tables`) with the matching columns added to both answer-hidden
    tables."""
    out = dict(tables)
    for key in ("l1_hidden", "knowqa_hidden"):
        out[key] = add_matching_features(tables[key])
    return out


def matching_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """For each matching column, on Study 1 and on each Study 2 regime: the share of trials
    that match, accuracy on those and on the rest, and the difference. One row per column,
    tests as column groups. Trials with no value are left out of their column's numbers."""
    parts = {("l1", "all"): tables["l1_hidden"]}
    knowqa = tables["knowqa_hidden"]
    for regime in REGIME_ORDER:
        parts[("knowqa", regime)] = knowqa[knowqa[REGIME_COL] == regime]
    parts[("knowqa", "all")] = knowqa

    rows = []
    for (ds, test), part in parts.items():
        correct = part[Con.IS_CORRECT_COLUMN] == 1
        for group, cols in MATCHING_GROUPS.items():
            for col in cols:
                match = part[col]
                known = match.notna()
                m = match[known] == 1
                c = correct[known]
                rows.append({
                    "feature": col, "group": group, "dataset": ds, "test": test,
                    "share matching": m.mean(),
                    "accuracy if matching": c[m].mean(),
                    "accuracy if not": c[~m].mean(),
                    "difference": c[m].mean() - c[~m].mean(),
                })
    long = pd.DataFrame(rows)
    return long.pivot_table(
        index=["group", "feature"], columns=["dataset", "test"],
        values=["share matching", "difference"], sort=False,
    )


def matching_feature_sets() -> Dict[str, List[str]]:
    """Each of `MATCHING_ANCHORS` alone, with each matching group, and with both; and the
    groups on their own."""
    return selected_feature_sets(base_cols(), anchors=MATCHING_ANCHORS, groups=MATCHING_GROUPS)


def main(save: bool = True, verbose: bool = True, rerun: bool = False) -> Dict[str, pd.DataFrame]:
    """The matching search, saved as a heatmap with its scores. Sets scored before are read
    back unless `rerun=True`. Returns the comparison."""
    tables = with_matching(load_tables(verbose=verbose))
    sets = matching_feature_sets()
    search = run_deliberation_search(tables, sets, verbose=verbose, rerun=rerun)
    plot_search_heatmap(search["scores"], search="matching", order=list(sets), save=save, close=True)
    return search


if __name__ == "__main__":
    main()
