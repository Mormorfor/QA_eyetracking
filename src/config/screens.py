"""The two screens a trial can show, described as data.

A participant reads a **paragraph** (every Study 1 trial; a third of Study 2's)
and then sees an **answer screen** with the question and four options. Both
screens are measured the same way -- the same per-area measurements and the
same reading times -- so one piece of code serves both. A `Screen` holds only
the few things that differ between them: which column says what area a word is
in, and which areas exist.

A `Screen` holds column names and settings, never functions, because `config/`
must not import `features/`.

Each screen is still processed into its own tables. Where something needs both,
they are joined on participant_id and TRIAL_INDEX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.config import columns as C


@dataclass(frozen=True)
class Screen:
    """One screen of a trial, and the ways it differs from the other one."""

    #: Short name, and the key in `SCREENS`.
    key: str

    #: Readable name for printed output.
    label: str

    #: The column that says which area of this screen a word is in:
    #: `area_label` on the answer screen, `auxiliary_span_type` on the
    #: paragraph. Swapping this one column is what lets both screens use the
    #: same measurement code in `features/area_metrics.py`.
    area_col: str

    #: The areas on this screen, in reporting order. Also used as the suffixes
    #: of the reading-time columns, e.g. RT_pure_answer_A.
    regions: tuple[str, ...]


ANSWERS = Screen(
    key="answers",
    label="answer screen",
    area_col=C.AREA_LABEL_COLUMN,
    regions=("question", "answer_A", "answer_B", "answer_C", "answer_D"),
)

PARAGRAPH = Screen(
    key="paragraph",
    label="paragraph screen",
    area_col=C.AUXILIARY_SPAN_TYPE_COLUMN,
    regions=("outside", "distractor", "critical"),
)

SCREENS: Mapping[str, Screen] = {s.key: s for s in (ANSWERS, PARAGRAPH)}


def screen(key: str) -> Screen:
    """Look a screen up by key, failing loudly on a typo."""
    if key not in SCREENS:
        raise KeyError(f"unknown screen {key!r}; known: {sorted(SCREENS)}")
    return SCREENS[key]


# The raw pupil columns that both screens convert and z-score.
PUPIL_RAW_COLS: Sequence[str] = (
    C.IA_MAX_FIX_PUPIL_SIZE,
    C.IA_MIN_FIX_PUPIL_SIZE,
    C.IA_AVERAGE_FIX_PUPIL_SIZE,
)
