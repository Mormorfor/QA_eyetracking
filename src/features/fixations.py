"""The fixation report: reading it once, and placing each fixation on a word.

The eye tracker's fixation report has one row per fixation, in time order. Each
screen's pipeline reads it once, with `load_fixations`, and hands that one table
to everything that needs single fixations: the reading times, the fixation
sequences, the pupil baseline and the area looked at before each button press.

`place_fixations` gives every fixation its word and area by one rule -- the
word it landed on, else the word it landed nearest -- so everything that needs
to know where a fixation was places it the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import columns as C

KEYS = [C.PARTICIPANT_ID, C.TRIAL_ID]

# Columns that are turned into numbers after reading. The report writes "." for
# "no value", which becomes NaN. Every other column stays text.
NUMERIC_COLUMNS = (
    C.CURRENT_FIX_START,
    C.CURRENT_FIX_DURATION,
    C.CURRENT_FIX_INTEREST_AREA_ID,
    C.NEAREST_IA,
    C.CURRENT_FIX_PUPIL_SIZE,
)


def load_fixations(
    path: Path,
    columns: Sequence[str],
    participants: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read `columns` of a fixation report, plus participant_id and TRIAL_INDEX.

    Everything is read as text: the report writes "." for a missing value, so
    letting pandas guess the types gives different types in different parts of
    the file. The columns in `NUMERIC_COLUMNS` are then turned into numbers,
    with "." becoming NaN. participant_id and TRIAL_INDEX stay text; give them
    the word table's types with `match_key_types` before joining the two.

    The reports are large (Study 1's paragraph report is 6 GB), so the file is
    read in chunks and only the requested columns are kept. `participants`
    keeps only those participants, for a quick check.

    Stops with an error if a fixation has no trial id or no start time: every
    fixation in the current reports has both, so a missing one means the
    report is not what this code expects.
    """
    usecols = KEYS + [c for c in columns if c not in KEYS]
    numeric = [c for c in usecols if c in NUMERIC_COLUMNS]

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=str):
        if participants is not None:
            chunk = chunk[chunk[C.PARTICIPANT_ID].isin(participants)]
        for c in numeric:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)[usecols]

    no_trial = int(out[C.TRIAL_ID].isna().sum())
    no_start = int(out[C.CURRENT_FIX_START].isna().sum()) if C.CURRENT_FIX_START in out else 0
    if no_trial or no_start:
        raise ValueError(
            f"{Path(path).name}: {no_trial} fixation(s) without a TRIAL_INDEX and "
            f"{no_start} without a start time. Every fixation should have both."
        )
    if verbose:
        print(f"  read {len(out):,} fixations ({len(usecols)} columns) from {Path(path).name}")
    return out


def match_key_types(fixations: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Give the fixation table's participant_id and TRIAL_INDEX the types they
    have in `table` (the word table), so the two can be joined.

    The fixation table holds them as text; the word table may hold them as
    numbers (Study 1's TRIAL_INDEX, Study 2's all-digit participant ids). A
    value that cannot be converted stops with an error.
    """
    out = fixations.copy()
    for k in KEYS:
        if out[k].dtype != table[k].dtype:
            out[k] = out[k].astype(table[k].dtype)
    return out


def place_fixations(
    fixations: pd.DataFrame,
    words: pd.DataFrame,
    area_cols: Sequence[str],
    *,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Give every fixation the word it belongs to, and that word's area(s).

    The rule: the word the fixation landed on; if it landed on none, the word
    it landed nearest. When the tracker fills both columns they always agree, so
    "the word it is on, else the nearest one" is the same as "the nearest one"
    wherever both exist.

    `words` is the word table of the same screen (one row per word, with
    `IA_ID` and the columns in `area_cols`).

    A fixation that cannot be placed -- no word and no nearest word -- gets no
    word and no area; the reading times treat it as separating one visit from
    the next. A fixation in a trial that is not in `words` (the report also
    holds practice and excluded trials) gets a word but no area. Both are
    counted and printed.

    Stops with an error if a fixation in a trial of `words` points at a word that
    trial does not have: that means the fixation report and the word table do
    not match, and the area would wrongly read as not looked at.

    Returns `fixations` in the same order, with `IA_ID` (the word placed on) and
    the `area_cols` added.
    """
    area_cols = list(area_cols)
    fix = match_key_types(fixations, words)

    exact = fix[C.CURRENT_FIX_INTEREST_AREA_ID]
    nearest = fix[C.NEAREST_IA]
    resolved = exact.fillna(nearest).astype("float64")
    n_filled = int(exact.isna().sum() - resolved.isna().sum())

    labels = (
        words[KEYS + [C.INTEREST_AREA_ID] + area_cols]
        .dropna(subset=[C.INTEREST_AREA_ID] + area_cols)
        .drop_duplicates(subset=KEYS + [C.INTEREST_AREA_ID])
        .copy()
    )
    # Word ids as decimals on both sides: the fixation report has NaN for
    # fixations on no word, so its ids are decimals, and whole numbers on the
    # other side would match nothing.
    labels[C.INTEREST_AREA_ID] = pd.to_numeric(
        labels[C.INTEREST_AREA_ID], errors="coerce"
    ).astype("float64")

    fix[C.INTEREST_AREA_ID] = resolved
    merged = fix.merge(labels, on=KEYS + [C.INTEREST_AREA_ID], how="left")
    assert len(merged) == len(fix), (
        f"fixation -> word join duplicated rows: {len(fix)} -> {len(merged)}"
    )

    # Two very different reasons a fixation can fail to get an area:
    #   - its trial is not in the word table at all. Normal: the report also
    #     holds practice and excluded trials (about 4,600 on Study 1).
    #   - its trial IS there, but its word id is not one of that trial's
    #     words. That would mean the two tables do not match, so it stops.
    analysed = set(map(tuple, labels[KEYS].drop_duplicates().itertuples(index=False, name=None)))
    in_analysed = merged.set_index(KEYS).index.isin(analysed)
    no_word = merged[C.INTEREST_AREA_ID].isna().to_numpy()
    no_area = merged[area_cols[0]].isna().to_numpy()
    n_outside = int((~in_analysed).sum())
    n_neither = int((no_word & in_analysed).sum())
    n_unmatched = int((no_area & in_analysed & ~no_word).sum())

    if verbose:
        tag = f"{label} " if label else ""
        print(
            f"  {tag}fixation->area: {len(merged):,} fixations | "
            f"{n_filled:,} placed by the nearest-IA fallback | "
            f"{n_outside:,} in trials outside the analysis set | "
            f"{n_neither:,} in the analysis set with no interest area at all"
        )
    if n_unmatched:
        raise ValueError(
            f"{label or 'fixation->area'}: {n_unmatched:,} fixation(s) in trials the IA "
            f"table DOES contain carry an interest-area id that is not among that "
            f"trial's interest areas. Every such fixation would accrue no reading time "
            f"and the region would be written as a real zero. This is a key or dtype "
            f"mismatch between the fixation report and the IA table, not missing data."
        )
    return merged
