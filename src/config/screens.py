"""The two screens a trial has, as configuration rather than as two code paths.

A participant sees a **paragraph** (in Study 1; a third of Study 2's trials) and
then an **answer screen**. Both are read, both yield the same eight per-area
metrics and the same RT/TFD families, and before 2026-10-06 each had its own
copy of the orchestration that produces them -- the merge loop, the pupil prep,
the `RT_* -> TimeSinceOffset_*` rename.

A `Screen` is the handful of things that genuinely differ. Everything else is
one implementation taking one of these (`docs/decisions/2026-10-06-stage-d-proposal.md`
section 3.8, Diana's ruling: *"can the paragraph csv creator be same as QA just
with a few different flags?"*).

**This stays configuration, not behaviour.** `config/` must not import
`features/`, so a Screen holds column names and switches -- never functions. The
same rule `Dataset.skip_base_features` follows.

**A field nothing reads is a lie about the design.** `drop_leading_question` was on this record
briefly on 2026-10-06 and removed the same day: nothing consumed it, because the step that would
(`create_fixation_sequence_tags`, which builds the collapsed area sequence) has not been unified
across the screens yet. It goes back on the day that step does.

**What is NOT here, deliberately.** The two screens are still *generated
separately into separate artifacts*, joined on `(participant_id, TRIAL_INDEX)`
where something wants both (T6.1, Diana). Unifying the orchestration does not
merge the outputs, and the answer screen keeps `ingest/build.py::main` as its
production entry point -- it does much more than the paragraph path (base
features, button clicks, last-visited, the hunters/gatherers split) and reaches
trial grain later. Making those one pipeline is a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.config import columns as C


@dataclass(frozen=True)
class Screen:
    """One reading surface, and the handful of ways it differs from the other."""

    #: Short name, and the key in `SCREENS`.
    key: str

    #: Human label for printed output.
    label: str

    #: The column that says which area of this screen a word belongs to.
    #: `area_label` on the answer screen, `auxiliary_span_type` on the paragraph.
    #: This single substitution is what lets both screens share every metric
    #: builder in `features/area_metrics.py`.
    area_col: str

    #: The areas that exist, in reporting order. Also the RT/TFD column suffixes.
    regions: tuple[str, ...]

    #: Keep the raw-millimetre pupil means alongside the z-scored ones. The
    #: answer screen reports both; the paragraph screen reports only z, because
    #: nothing consumes raw paragraph pupil sizes.
    include_raw_pupil: bool = True

    #: Leave `area_skipped` on the caller's frame. The answer screen's saved
    #: IA-level table (`all_participants.csv`) has that column in its schema;
    #: the paragraph screen produces no IA-level artifact, so it does not.
    write_skip_indicator: bool = True

    #: Keep `total_area_dwell_time` / `total_dwell_time` beside the proportion.
    #: Same reason as above -- they are part of `all_participants.csv`'s schema.
    keep_dwell_totals: bool = True


ANSWERS = Screen(
    key="answers",
    label="answer screen",
    area_col=C.AREA_LABEL_COLUMN,
    regions=("question", "answer_A", "answer_B", "answer_C", "answer_D"),
    include_raw_pupil=True,
    write_skip_indicator=True,
    keep_dwell_totals=True,
)

PARAGRAPH = Screen(
    key="paragraph",
    label="paragraph screen",
    area_col=C.AUXILIARY_SPAN_TYPE_COLUMN,
    regions=("outside", "distractor", "critical"),
    include_raw_pupil=False,
    write_skip_indicator=False,
    keep_dwell_totals=False,
)

SCREENS: Mapping[str, Screen] = {s.key: s for s in (ANSWERS, PARAGRAPH)}


def screen(key: str) -> Screen:
    """Look a screen up by key, failing loudly on a typo."""
    if key not in SCREENS:
        raise KeyError(f"unknown screen {key!r}; known: {sorted(SCREENS)}")
    return SCREENS[key]


# The pupil columns both screens z-score. Raw EyeLink names, so they live here
# beside the rest of the screen vocabulary rather than being retyped per screen.
PUPIL_RAW_COLS: Sequence[str] = (
    C.IA_MAX_FIX_PUPIL_SIZE,
    C.IA_MIN_FIX_PUPIL_SIZE,
    C.IA_AVERAGE_FIX_PUPIL_SIZE,
)
