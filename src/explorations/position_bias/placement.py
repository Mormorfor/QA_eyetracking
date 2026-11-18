"""Was anything put at the top more often than chance?

The tests of position rest on the answers being placed at random. This checks the placement as
it actually came out: that the correct answer, each kind of answer, long answers, and the
answers other people chose sat at each position about one time in four. It counts shuffles
(`data.UNIT_COLS`), not trials, since the shuffle is what is being checked, and judges each
statistic against placements drawn at random the way the design drew them.

**Why not the usual chi-squared test.** In every shuffle each kind takes exactly one position
and each position exactly one kind, so the counts of a kind-by-position table vary more than
that test assumes: by chance alone its statistic averages 12, not 9, and its p-values come out
too small. The random draws get this right.

`expected_shares` turns the check into the number that matters: the share of choices each
position would get if people chose by content alone -- by kind, or by how often other people
chose that very answer -- given what was actually placed there.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.explorations.drift_diffusion.data import KEYS, LABELS
from src.explorations.position_bias.data import POSITIONS, random_placements, subsets

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
SUBSET = "subset"


def _units(d: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per unit of `d`, by kind (A-D): its position (0-3), its log2 length in words, and how
    often other people chose it (`data._attractiveness`)."""
    u = d.drop_duplicates(["unit", "label"]).sort_values(["unit", "label"])
    assert (u.groupby("unit").size() == 4).all(), "a unit does not hold four answers"
    n = len(u) // 4
    assert (u["label"].to_numpy().reshape(n, 4) == np.array(LABELS)).all()
    pos = u["position"].map(POS_INDEX).to_numpy().reshape(n, 4)
    words = np.log2(u["n_words"].to_numpy(float)).reshape(n, 4)
    attract = u["attractiveness"].to_numpy(float).reshape(n, 4)
    return pos, words, attract


def _by_position(values: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """`values[u, kind]` laid out by position: `out[u, p]` is the value of the kind at p."""
    out = np.empty_like(values)
    np.put_along_axis(out, pos, values, axis=1)
    return out


def _statistics(pos: np.ndarray, words: np.ndarray, attract: np.ndarray) -> Dict[str, float]:
    """How far one placement is from even. Kinds: the chi-squared distance of the kind-by-
    position counts from one in four. Correct answer: how far its share at the top is from one
    in four. Length and attractiveness: the mean, over units, of the value at each position
    minus the unit's mean; the omnibus is the units times the sum of their squares, and the top
    is the absolute value at the top."""
    n = len(pos)
    counts = np.stack([(pos == p).sum(axis=0) for p in range(4)], axis=1)
    out = {
        "kinds": float(((counts - n / 4) ** 2 / (n / 4)).sum()),
        "correct_at_top": abs(float((pos[:, 0] == 0).mean()) - 0.25),
    }
    lengths = _by_position(words - words.mean(axis=1, keepdims=True), pos).mean(axis=0)
    out["length"] = float(n * (lengths ** 2).sum())
    out["length_at_top"] = abs(float(lengths[0]))
    ok = ~np.isnan(attract).any(axis=1)
    shares = _by_position(attract[ok] - 0.25, pos[ok]).mean(axis=0)  # a unit's four shares sum to 1
    out["attractiveness"] = float(ok.sum() * (shares ** 2).sum())
    out["attractiveness_at_top"] = abs(float(shares[0]))
    return out


TESTS = {
    "kinds": "each kind at each position one time in four",
    "correct_at_top": "the correct answer at the top one time in four",
    "length": "answers equally long at every position",
    "length_at_top": "the top answer as long as the others",
    "attractiveness": "answers others chose equally often at every position",
    "attractiveness_at_top": "the top answer chosen by others as often as the rest",
}


def placement_tests(answers: pd.DataFrame, dataset_key: str, *, n_draws: int = 10000, seed: int = 0) -> pd.DataFrame:
    """For each group: every statistic of `_statistics` on the placement as it came out, and
    the share of `n_draws` random placements that are at least as far from even."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        pos, words, attract = _units(d)
        observed = _statistics(pos, words, attract)
        extreme = dict.fromkeys(observed, 0)
        for draw in random_placements(len(pos), n_draws, seed):
            for key, value in _statistics(draw, words, attract).items():
                extreme[key] += value >= observed[key] - 1e-12
        for key, test in TESTS.items():
            rows.append({SUBSET: name, "test": test, "statistic": observed[key],
                         "p": (extreme[key] + 1) / (n_draws + 1), "n_units": len(pos),
                         "n_units_with_attractiveness": int((~np.isnan(attract).any(axis=1)).sum())})
    out = pd.DataFrame(rows)
    out[["n_draws", "seed"]] = n_draws, seed
    return out


def kind_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group: how many units put each kind at each position, and the share of units."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        u = d.drop_duplicates(["unit", "label"])
        counts = pd.crosstab(u["label"], u["position"]).reindex(index=LABELS, columns=POSITIONS, fill_value=0)
        t = counts.stack().rename("n_units").reset_index()
        t["share"] = t["n_units"] / u["unit"].nunique()
        rows.append(t.assign(**{SUBSET: name}))
    return pd.concat(rows, ignore_index=True)


def length_attractiveness_table(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group and position, over units: the mean length of the answer placed there (in
    words, and in log2 words above the unit's mean), and how often other people chose it."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        pos, words, attract = _units(d)
        ok = ~np.isnan(attract).any(axis=1)
        w = _by_position(2 ** words, pos)
        dev = _by_position(words - words.mean(axis=1, keepdims=True), pos)
        a = _by_position(attract[ok], pos[ok])
        for p, position in enumerate(POSITIONS):
            rows.append({SUBSET: name, "position": position, "mean_words": w[:, p].mean(),
                         "log2_words_above_unit_mean": dev[:, p].mean(), "n_units": len(pos),
                         "attractiveness": a[:, p].mean() if ok.any() else np.nan,
                         "n_units_with_attractiveness": int(ok.sum())})
    return pd.DataFrame(rows)


def expected_shares(answers: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """For each group and position: the share of choices that went there, against the share
    content alone would send there, given what sat there. "By kind" gives each answer the
    group's share of choices of its kind; "by attractiveness" gives it how often other people
    chose it, on the trials where all four answers have that."""
    rows = []
    for name, d in subsets(answers, dataset_key):
        kind_share = d.loc[d["is_chosen"] == 1, "label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
        d = d.assign(by_kind=d["label"].map(kind_share))
        full = d[d["attractiveness"].notna().groupby([d[k] for k in KEYS]).transform("all")]
        n_trials = len(d) // 4
        for position in POSITIONS:
            at = d[d["position"] == position]
            at_full = full[full["position"] == position]
            rows.append({SUBSET: name, "position": position, "observed": at["is_chosen"].mean(),
                         "expected_by_kind": at["by_kind"].mean(),
                         "observed_where_attractiveness": at_full["is_chosen"].mean() if len(at_full) else np.nan,
                         "expected_by_attractiveness": at_full["attractiveness"].mean() if len(at_full) else np.nan,
                         "n_trials": n_trials, "n_trials_with_attractiveness": len(at_full)})
    return pd.DataFrame(rows)
