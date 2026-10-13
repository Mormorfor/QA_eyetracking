"""FUNCTION_REGISTRY -- the pipeline recipe, and the runners that execute it.

This is the project's one piece of machinery (`CLAUDE.md`): a name -> function map
with a `kind` ("base" or "group") and the join columns each group feature merges
back on. **Adding a feature means registering it here.**

**Registry order is load-bearing and stays that way** (Diana, 2026-10-06). Several
of these functions mutate the frame they are handed, so a later one can depend on
an earlier one having coerced a column -- `create_first_encounter_pupil_size`
works because `create_mean_first_fix_duration` ran first. The default runner
executes them in registry order and that is the supported way to call them;
running a hand-picked subset out of order can break. Declaring the dependencies
was considered and declined for now (`todo.md` T3.24).

**Why the registry lives in `ingest/`.** It names functions from *both* layers --
the row-level builders beside it, and the group features from `features/` -- so it
cannot sit in `features/` without that package importing `ingest/`, which the
layering rule forbids. `ingest/` is the orchestration layer and is allowed to
import downstream, so the recipe belongs here.

> **Settled by stage D, 2026-10-07 — two kinds are final.** An earlier version of this
> docstring said the registry was "about to grow a third kind (`trial`)". It did not, and
> **should not**: step 3 audited the eight `include_*` booleans that a `trial` kind would have
> selected over, found seven of them dead or compute-only, and deleted them instead. One
> survived, and it was `has_paragraph` on the `Dataset` record rather than a registry concern.
> With nothing left to select, there was nothing for a third kind to do. `base` and `group` are
> the whole vocabulary. (`docs/decisions/2026-10-06-stage-d-proposal.md`, step 3.)
>
> The layering point above is no longer provisional either: `ingest/` importing `features/` is
> now a stated exception in `CLAUDE.md` rule 1, with the narrower rule that actually holds --
> only the three orchestration modules (`build`, `registry`, `knowqa`) reach down, the
> primitives do not, and `features/` imports `ingest/` nowhere.
"""

from src.config import columns as C
from src.config.screens import ANSWERS
from src.config.datasets import FIX_ANSWERS_PATH
from src.features import area_metrics as am
from src.features.area_metrics import (
    create_dwell_proportions,
    create_mean_area_dwell_time,
    create_mean_area_fix_count,
    create_mean_first_fix_duration,
    create_skip_rate,
)
from src.features.last_visited import create_last_area_and_location_visited
from src.features.pupil import (
    add_zscored_pupil_columns,
    create_first_encounter_pupil_size,
    create_mean_pupil_size_metrics,
)
from src.features.sequences import (
    create_fixation_sequence_tags,
    create_has_xyx,
    create_has_xyxy,
    create_longest_alt_answer_run,
    create_seq_len,
    create_trial_mean_dwell,
    create_simplified_fixation_tags,
    create_simplified_visit_counts,
)
from src.ingest.base_features import (
    add_IA_answer_label,
    add_IA_screen_location,
    add_answer_text_columns,
    add_is_correct,
    add_selected_answer_label,
    add_text_id,
    add_text_id_with_q,
    add_total_answering_RT_normalized,
)


# Keys inside a group function's kwargs that configure the orchestrator (how the
# result is merged back) rather than arguments forwarded to the feature function.
GROUP_ORCHESTRATION_KEYS = {"join_columns"}


#: The two kinds that aggregate. Both are run by `generate_new_row_features` and
#: both are currently broadcast back onto the interest-area rows -- the split into
#: separate `areas_*` and `trials` artifacts is step 4 of the grain proposal. Until
#: then the distinction is descriptive: it says what grain a feature *is*, not yet
#: where it is written.
GROUP_KINDS = ("area", "trial")

#: Every kind, coarsest last. `row` stays on the interest-area frame.
KINDS = ("row",) + GROUP_KINDS


FUNCTION_REGISTRY = {
    # Base features
    "add_text_id": {
        "callable": add_text_id,
        "needs": [C.BATCH_COLUMN, C.ARTICLE_COLUMN, C.DIFFICULTY_COLUMN, C.PARAGRAPH_COLUMN],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_text_id_with_q": {
        "callable": add_text_id_with_q,
        "needs": [C.TEXT_ID_COLUMN, C.SAME_CRITICAL_SPAN_COLUMN],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_is_correct": {
        "callable": add_is_correct,
        "needs": [
            C.SELECTED_ANSWER_POSITION_COLUMN,
            C.CORRECT_ANSWER_POSITION_COLUMN,
            # read only to name the offending trials if the assertion fires
            C.PARTICIPANT_ID,
            C.TRIAL_ID,
        ],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_answer_text_columns": {
        "callable": add_answer_text_columns,
        "needs": [C.ANSWERS_ORDER_COLUMN, *C.ANSWER_TEXT_COLUMNS],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_IA_screen_location": {
        "callable": add_IA_screen_location,
        "needs": [
            C.QUESTION_TEXT_COLUMN,
            *C.ANSWER_TEXT_COLUMNS,
            C.INTEREST_AREA_ID,
            C.TRIAL_ID,
            C.PARTICIPANT_ID,
            # the geometry reconciliation (T3.18) places each word by its rectangle
            C.IA_TOP,
            C.IA_LEFT,
        ],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_IA_answer_label": {
        "callable": add_IA_answer_label,
        "needs": [C.AREA_SCREEN_LOCATION, C.ANSWERS_ORDER_COLUMN],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_selected_answer_label": {
        "callable": add_selected_answer_label,
        "needs": [C.ANSWERS_ORDER_COLUMN, C.SELECTED_ANSWER_POSITION_COLUMN],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_zscored_pupil_columns": {
        "callable": add_zscored_pupil_columns,
        "needs": [
            C.IA_MAX_FIX_PUPIL_SIZE,
            C.IA_MIN_FIX_PUPIL_SIZE,
            C.IA_AVERAGE_FIX_PUPIL_SIZE,
            C.PARTICIPANT_ID,
        ],
        "default_kwargs": {},
        "kind": "row",
    },
    "add_total_answering_RT_normalized": {
        "callable": add_total_answering_RT_normalized,
        "needs": [
            C.CONFIRM_FINAL_ANSWER_RT,
            # all produced by add_IA_screen_location, above
            "question_len",
            "1_len",
            "2_len",
            "3_len",
            "4_len",
            "n_interest_areas",
        ],
        "default_kwargs": {},
        "kind": "row",
    },
    # Group-level functions
    "create_mean_area_dwell_time": {
        "callable": create_mean_area_dwell_time,
        "needs": [C.IA_DWELL_TIME, *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_mean_area_fix_count": {
        "callable": create_mean_area_fix_count,
        "needs": [C.IA_FIXATIONS_COUNT, *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_mean_first_fix_duration": {
        "callable": create_mean_first_fix_duration,
        "needs": [C.IA_FIRST_FIXATION_DURATION, *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_skip_rate": {
        "callable": create_skip_rate,
        "needs": [C.IA_DWELL_TIME, *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_dwell_proportions": {
        "callable": create_dwell_proportions,
        "needs": [C.IA_DWELL_TIME, *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_mean_pupil_size_metrics": {
        "callable": create_mean_pupil_size_metrics,
        "needs": [
            C.IA_MAX_FIX_PUPIL_SIZE,
            C.IA_MIN_FIX_PUPIL_SIZE,
            C.IA_AVERAGE_FIX_PUPIL_SIZE,
            # produced by add_zscored_pupil_columns -- this is the ordering
            # dependency the order check below exists to verify (T3.11)
            f"{C.IA_MAX_FIX_PUPIL_SIZE}_z",
            f"{C.IA_MIN_FIX_PUPIL_SIZE}_z",
            f"{C.IA_AVERAGE_FIX_PUPIL_SIZE}_z",
            *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN],
        ],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_first_encounter_pupil_size": {
        "callable": create_first_encounter_pupil_size,
        "needs": [
            C.IA_AVERAGE_FIX_PUPIL_SIZE,
            f"{C.IA_AVERAGE_FIX_PUPIL_SIZE}_z",
            C.IA_FIRST_FIXATION_DURATION,
            C.INTEREST_AREA_ID,
            *[C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN],
        ],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    "create_last_area_and_location_visited": {
        "callable": create_last_area_and_location_visited,
        "needs": [
            C.IA_LAST_FIXATION_TIME,
            C.AREA_LABEL_COLUMN,
            C.AREA_SCREEN_LOCATION,
            C.TRIAL_ID,
            C.PARTICIPANT_ID,
        ],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    ## heavy iternal data loading here. Might fix later, slow for now.
    "create_fixation_sequence_tags": {
        "callable": create_fixation_sequence_tags,
        "needs": [
            C.AREA_LABEL_COLUMN,
            C.AREA_SCREEN_LOCATION,
            C.INTEREST_AREA_ID,
            C.TRIAL_ID,
            C.PARTICIPANT_ID,
        ],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID],
            # Forwarded to the function; main() overrides it with its fixations_path.
            "fix_path": FIX_ANSWERS_PATH,
        },
        "kind": "trial",
    },
    "create_simplified_fixation_tags": {
        "callable": create_simplified_fixation_tags,
        "needs": [
            C.FIX_SEQUENCE_BY_LABEL,
            C.FIX_SEQUENCE_BY_LOCATION,
            C.TRIAL_ID,
            C.PARTICIPANT_ID,
        ],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    "create_simplified_visit_counts": {
        "callable": create_simplified_visit_counts,
        "needs": [
            C.SIMPLIFIED_FIX_SEQ_BY_LABEL,
            C.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
            C.AREA_LABEL_COLUMN,
            C.AREA_SCREEN_LOCATION,
            C.TRIAL_ID,
            C.PARTICIPANT_ID,
        ],
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "area",
    },
    # -- the five DERIVED_COLS, moved out of features/build.py (step 5)
    "create_seq_len": {
        "callable": create_seq_len,
        "needs": [C.SIMPLIFIED_FIX_SEQ_BY_LABEL, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    "create_has_xyx": {
        "callable": create_has_xyx,
        "needs": [C.SIMPLIFIED_FIX_SEQ_BY_LABEL, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    "create_has_xyxy": {
        "callable": create_has_xyxy,
        "needs": [C.SIMPLIFIED_FIX_SEQ_BY_LABEL, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    "create_longest_alt_answer_run": {
        "callable": create_longest_alt_answer_run,
        "needs": [C.SIMPLIFIED_FIX_SEQ_BY_LABEL, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
    "create_trial_mean_dwell": {
        # the AREA metric, broadcast -- not IA_DWELL_TIME. See the docstring.
        "callable": create_trial_mean_dwell,
        "needs": [C.MEAN_DWELL_TIME, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
}


def resolve_base_functions(name_list=None):
    """
    Resolve base feature functions.

    If name_list is None, return ALL base functions from FUNCTION_REGISTRY
    in registry insertion order. Otherwise, return only the named ones.

    can accept entries in name_list as either:
    - "func_name"
    - ("func_name", {override_kwargs})
    """
    # Case 1: no explicit list → use all base functions
    if name_list is None:
        return [
            (entry["callable"], entry.get("default_kwargs", {}))
            for name, entry in FUNCTION_REGISTRY.items()
            if entry.get("kind") == "row"
        ]

    # Case 2: explicit list → validate and return only those
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "row":
                raise ValueError(
                    f"Function '{name}' is not registered as a row feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "row":
                raise ValueError(
                    f"Function '{name}' is not registered as a row feature."
                )
            merged_kwargs = {**entry.get("default_kwargs", {}), **user_kwargs}
            resolved.append((entry["callable"], merged_kwargs))
            continue

        raise ValueError(
            f"Invalid base function specification: {item}. "
            "Must be 'name' or ('name', {kwargs})."
        )

    return resolved


def resolve_group_functions(name_list=None):
    """
    Resolve group-level feature functions.

    If name_list is None, return ALL group functions from FUNCTION_REGISTRY
    (with their default kwargs). Otherwise, name_list can contain:
        - "func_name"
        - ("func_name", {override_kwargs})
    """
    # Case 1: no explicit list → all group functions with their defaults
    if name_list is None:
        return [
            (entry["callable"], entry.get("default_kwargs", {}))
            for name, entry in FUNCTION_REGISTRY.items()
            if entry.get("kind") in GROUP_KINDS
        ]

    # Case 2: explicit list
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") not in GROUP_KINDS:
                raise ValueError(
                    f"Function '{name}' is not a group (area/trial) feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") not in GROUP_KINDS:
                raise ValueError(
                    f"Function '{name}' is not a group (area/trial) feature."
                )

            merged_kwargs = {**entry.get("default_kwargs", {}), **user_kwargs}
            resolved.append((entry["callable"], merged_kwargs))
            continue

        raise ValueError(
            f"Invalid group function specification: {item}. "
            "Must be 'name' or ('name', {kwargs})."
        )

    return resolved


def add_base_features(df, functions, verbose=False):
    """
    Apply a sequence of transformation functions to a DataFrame.

    Each function in `functions` must take a single DataFrame as input and
    return a (transformed) DataFrame as output. The functions are applied
    in the order given.
    """

    out = df.copy()
    for func, kwargs in functions:
        if verbose:
            print(f"Running: {func.__name__}")
        out = func(out, **kwargs) if kwargs else func(out)
    return out.reset_index(drop=False)


def _registry_name_for(func) -> str:
    """The registry key a callable is registered under, or its `__name__`."""
    for name, entry in FUNCTION_REGISTRY.items():
        if entry["callable"] is func:
            return name
    return getattr(func, "__name__", repr(func))


#: Columns produced outside the registry, by the two trial-level attach steps in
#: `ingest/build.py` (step 4b of the grain proposal).
#:
#: **Trial-grain by construction, not by observation.** Both steps build a frame
#: keyed on `(participant_id, TRIAL_INDEX)` and merge it on those columns -- the
#: declaration records what the code already enforces.
#:
#: Generated from `ANSWERS.regions` rather than typed out, so adding a region
#: cannot leave a column undeclared. Six families x five regions, plus the two
#: click-derived last-label columns.
#:
#: > **One thing this does NOT do.** Section 1 of the proposal argues the RT/TFD
#: > families are *area*-grain by meaning -- one reading time per region per trial
#: > -- and should be stored long rather than wide. That is a **reshape**, and
#: > reshaping is not what step 4 does: Diana, 2026-10-13, *"keep the existing
#: > computation and feature definitions unchanged."* As the columns exist today,
#: > one row per trial carrying five regions side by side, they are trial-grain,
#: > and that is what is declared here. The long-form reshape stays an open piece
#: > of section 1, to be done with the `model_ready` pivot that would consume it.
ATTACH_PRODUCES: dict = {
    **{
        f"{prefix}{region}": "trial"
        for prefix in (
            "RT_pure_",
            "RT_normalized_",
            "TFD_pure_",
            "TFD_normalized_",
            "TimeSinceOffset_pure_",
            "TimeSinceOffset_normalized_",
        )
        for region in ANSWERS.regions
    },
    C.LAST_LBL_BEFORE_SELECT: "trial",
    C.LAST_LBL_BEFORE_CONFIRM: "trial",
}

#: Raw report columns that are not per-word (step 4c of the grain proposal).
#:
#: Nothing in the pipeline produces these -- the tracker and the stimulus file do
#: -- so no registry entry can own them and this map is the only honest home.
#: Grouped by what they *are*, because that is the argument for each one being
#: trial-grain: an identifier of the item, a condition of the trial, a property of
#: the response, or a property of the recording. None is a measurement of a word.
#:
#: `participant_id` and `TRIAL_INDEX` are deliberately absent: they are the join
#: keys and every grain carries them (`check_join_keys`).
RAW_COLUMN_GRAIN: dict = {
    # -- which item was shown
    C.ARTICLE_COLUMN: "trial",
    C.BATCH_COLUMN: "trial",
    C.DIFFICULTY_COLUMN: "trial",
    C.PARAGRAPH_COLUMN: "trial",
    C.SAME_CRITICAL_SPAN_COLUMN: "trial",
    C.ONESTOPQA_QUESTION_ID: "trial",
    C.LIST_COLUMN: "trial",
    # -- the stimulus text itself, one string per trial
    C.QUESTION_TEXT_COLUMN: "trial",
    **{col: "trial" for col in C.ANSWER_TEXT_COLUMNS},
    C.ANSWERS_ORDER_COLUMN: "trial",
    # -- the condition this trial was run under
    C.QUESTION_PREVIEW_COLUMN: "trial",
    C.PRACTICE_TRIAL_COLUMN: "trial",
    C.REPEATED_TRIAL_COLUMN: "trial",
    # -- what the participant did
    C.SELECTED_ANSWER_POSITION_COLUMN: "trial",
    C.CORRECT_ANSWER_POSITION_COLUMN: "trial",
    C.NUM_OF_SELECTS: "trial",
    C.CONFIRM_FINAL_ANSWER_RT: "trial",
    # -- how the trial was recorded
    C.DOMINANT_EYE_COLUMN: "trial",
    C.INTEREST_AREA_FIXATION_SEQUENCE: "trial",
    # -- Study 2 only: the knowledge-regime manipulation, its counterbalancing,
    # and the per-trial confidence self-report. Absent from L1's report, so these
    # rows are inert there, exactly like their `PUBLISHED_NEEDS` counterparts.
    #
    # Declared here as well as there because the two maps answer different
    # questions and both have to be answered: `PUBLISHED_NEEDS` says *read this
    # column*, `RAW_COLUMN_GRAIN` says *and it belongs to the trial*. Adding only
    # the first, on 2026-10-13, kept the columns but left them on
    # `interest_areas` -- broadcast across ~39 rows instead of sitting once on
    # `trials`. Nothing was lost, because an undeclared grain defaults to staying
    # put; it was simply the wrong grain.
    C.SESSION_ID: "trial",
    C.TRIAL_NUMBER: "trial",
    C.REGIME_COLUMN: "trial",
    C.REGIME_LIST_COLUMN: "trial",
    C.REGIME_ORDER_COLUMN: "trial",
    C.REGIME_ORDER_POSITION_COLUMN: "trial",
    C.CONFIDENCE_RATING_COLUMN: "trial",
    C.CONFIDENCE_RATING_TXT_COLUMN: "trial",
    C.ARTICLE_GROUP_COLUMN: "trial",
    C.REPHRASED_ANSWER_A_COLUMN: "trial",
}


def column_grain() -> dict:
    """`column -> "row" | "area" | "trial"`, from the three declared sources.

    A column **not** named here is word-grain and stays on `interest_areas`.
    That default runs in the safe direction: a forgotten declaration leaves a
    column exactly where it is today, rather than silently moving it to a grain
    where a `.first()` would collapse values that differ.

    Assembled, never observed. `check_declared_grains` compares this against what
    the data actually does, but only to catch a wrong declaration -- observation
    does not assign a grain (grain proposal section 5b).
    """
    grains = {}
    for produced in _PRODUCES.values():
        grains.update(produced)
    grains.update(ATTACH_PRODUCES)
    grains.update(RAW_COLUMN_GRAIN)
    return grains


def split_by_grain(result_df, collected: dict = None):
    """Separate one broadcast interest-area frame into its three real grains.

    Step 4 of `docs/decisions/2026-10-13-feature-grain-proposal.md`. Driven by
    **declarations** (`column_grain()`), not by the shape of the collected frames
    and not by observing which columns happen to be constant -- see section 5b
    for why that distinction is the whole point.

    `collected` is unused for the split itself and kept only because callers
    already pass it; the grains come from the declarations.

    Returns `(interest_areas, areas, trials)`. Every grain carries the join keys,
    so the three rejoin without anything being carried implicitly.
    """
    grains = column_grain()
    trial_key = [C.TRIAL_ID, C.PARTICIPANT_ID]
    area_key = trial_key + [C.AREA_LABEL_COLUMN]

    present = [c for c in result_df.columns if c not in area_key]
    area_cols = [c for c in present if grains.get(c) == "area"]
    trial_cols = [c for c in present if grains.get(c) == "trial"]
    row_cols = [c for c in present if c not in set(area_cols) | set(trial_cols)]

    ia = result_df[area_key + row_cols].copy()
    areas = (
        result_df[area_key + area_cols]
        .drop_duplicates(subset=area_key)
        .reset_index(drop=True)
    )
    trials = (
        result_df[trial_key + trial_cols]
        .drop_duplicates(subset=trial_key)
        .reset_index(drop=True)
    )
    return ia, areas, trials


def check_join_keys(ia, areas, trials) -> None:
    """Every grain carries the keys the rejoin needs. Fails first, and says so."""
    trial_key = {C.TRIAL_ID, C.PARTICIPANT_ID}
    for name, frame, need in (
        ("interest_areas", ia, trial_key | {C.AREA_LABEL_COLUMN}),
        ("areas", areas, trial_key | {C.AREA_LABEL_COLUMN}),
        ("trials", trials, trial_key),
    ):
        missing = need - set(frame.columns)
        if missing:
            raise AssertionError(f"{name} is missing join key(s): {sorted(missing)}")


def check_declared_grains(result_df, sample_trials: int = None) -> list:
    """Compare every declared grain against what the data actually does.

    Catches a **wrong** declaration; it does not assign one. A column declared
    `trial` must be constant within `(participant_id, TRIAL_INDEX)`, and one
    declared `area` constant within that plus `area_label`. A column that is
    observably constant but undeclared is **reported, not moved** -- the default
    is to leave it where it is.

    Returns a list of complaints; empty means every declaration holds.
    """
    grains = column_grain()
    trial_key = [C.TRIAL_ID, C.PARTICIPANT_ID]
    area_key = trial_key + [C.AREA_LABEL_COLUMN]
    df = result_df
    if sample_trials:
        keep = df[trial_key].drop_duplicates().head(sample_trials)
        df = df.merge(keep, on=trial_key, how="inner")

    bad = []
    gt = df.groupby(trial_key, sort=False)
    ga = df.groupby(area_key, sort=False)
    for col, grain in grains.items():
        if col not in df.columns or grain == "row":
            continue
        grouped = gt if grain == "trial" else ga
        n = grouped[col].nunique(dropna=False).max()
        if n is not None and n > 1:
            bad.append(
                f"{col!r} is declared {grain}-grain but takes {int(n)} distinct "
                f"values inside one {grain}"
            )
    return bad


def check_grain_split_is_lossless(result_df, ia, areas, trials) -> None:
    """Every column of the broadcast frame survives in exactly one grain."""
    got = set(ia.columns) | set(areas.columns) | set(trials.columns)
    missing = set(result_df.columns) - got
    if missing:
        raise AssertionError(
            f"grain split lost {len(missing)} column(s): {sorted(missing)[:10]}"
        )


def generate_new_row_features(
    functions, df, default_join_columns=None, verbose=True, collect=None
):
    """
    Iteratively compute and merge group-level features into a row-level DataFrame.

    Each entry in `functions` is a tuple:
        (func, func_kwargs)

    `func_kwargs` is split into two roles:
    - "join_columns" (and any other GROUP_ORCHESTRATION_KEYS) are consumed here to
      control the left-merge and are NOT passed to `func`.
    - every remaining key is forwarded as a keyword argument to `func`, e.g. a
      `fix_path` that points the fixation-sequence feature at a specific fixations
      report.

    For each function:
    1. Compute `new_features_df = func(result_df, **forwarded_kwargs)`
    2. Merge `new_features_df` into `result_df` using a left join on `join_columns`.

    Returns
    -------
    DataFrame
        The original DataFrame enriched with all new feature columns produced
        by the functions in `functions`.
    """
    if default_join_columns is None:
        default_join_columns = [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]

    result_df = df.copy()

    # Resolve the report's "." sentinel to numbers ONCE, before any metric runs
    # (`todo.md` T3.11). This used to happen inside five of the group functions,
    # each coercing the shared frame on the way past, so which metrics you asked
    # for decided whether the others got numbers -- `create_first_encounter_pupil_size`
    # worked only because `create_mean_first_fix_duration` had run first, and
    # `group_function_names=[...]` on a subset could compare str with int.
    #
    # `inplace=True` is deliberate: the coerced columns are expected in the saved
    # IA-level table, so dropping the mutation here would change
    # `all_participants.csv`'s schema. What changes is that it is now one named
    # step with an owner, not a side effect of whichever metric ran first.
    # `pupil=False`: the pupil columns are scaled to mm and z-scored by an earlier
    # base function, and re-coercing them here would undo that.
    #
    # The paragraph pipeline already does the same thing at `paragraph_prep.py`
    # (`coerce_ia_columns(paragraph_ia, pupil=True)`), and the restructure moves
    # this line to `ingest/readers.py` -- one line relocates, nothing else.
    am.coerce_ia_columns(result_df, inplace=True)

    for func, func_kwargs in functions:
        if verbose:
            print(f"Running group feature: {func.__name__}")

        join_columns = func_kwargs.get("join_columns", default_join_columns)
        call_kwargs = {
            key: value
            for key, value in func_kwargs.items()
            if key not in GROUP_ORCHESTRATION_KEYS
        }

        new_features_df = func(result_df, **call_kwargs)
        # Capture the feature's own output -- at its real grain, before the
        # broadcast -- when the caller asked for it (`split_by_grain`). The merge
        # below is unchanged, so collecting costs a reference and changes nothing.
        if collect is not None:
            collect[_registry_name_for(func)] = new_features_df
        result_df = result_df.merge(new_features_df, on=join_columns, how="left")

    return result_df


# ---------------------------------------------------------------------------
# `needs`, and the two things that fall out of it
#
# Stage D section 3.4, approved 2026-10-06, implemented 2026-10-13 as step 1 of
# `docs/decisions/2026-10-13-feature-grain-proposal.md`.
#
# **One correction to that design, found while implementing it.** Section 3.4
# assumed the registry is the only consumer of the raw read, so the whitelist
# could be `union(entry["needs"]) & header`. It is not. Three other classes of
# consumer read raw columns the registry never names, and a whitelist built from
# the registry alone would drop every one of them:
#
#   1. the pipeline's own orchestration -- `ingest/build.py` filters on
#      `repeated_reading_trial` and `practice_trial` (:313, :317) and splits on
#      `question_preview`, all outside the registry walk;
#   2. the trial-level attach steps -- RT/TFD and last-label -- which take the IA
#      frame and read columns of their own;
#   3. **downstream readers of the published artifact.** `all_participants.csv`
#      is not an intermediate: `features/build.py` pulls `ANSWER_PRESS_NUMBER`
#      off it (:633) and `features/strategies.py` reads `EYE_TRACKED`.
#
# So the whitelist is the union of four declarations, not one. They are kept
# apart rather than merged because they answer different questions and go stale
# for different reasons.
# ---------------------------------------------------------------------------

#: Columns the prep pipeline reads outside any registry entry (class 1 above).
PIPELINE_NEEDS = [
    C.REPEATED_TRIAL_COLUMN,
    C.PRACTICE_TRIAL_COLUMN,
    C.QUESTION_PREVIEW_COLUMN,
]

#: Columns the trial-level attach steps read from the IA frame (class 2 above).
ATTACH_NEEDS = [
    C.IA_LAST_FIXATION_TIME,
    C.IA_FIRST_FIXATION_TIME,
    C.IA_LAST_FIXATION_DURATION,
]

#: Columns `all_participants.csv` must carry for its consumers (class 3 above).
#:
#: ✅ **Proved 2026-10-13** (step 2), three ways, and the filter was switched on
#: as a result:
#:   1. all 20 registry features run on a 2,496-trial sample read both ways --
#:      89 shared columns, **0 differing**;
#:   2. `model_ready` built from a 121-column interest-area frame against one
#:      built from the full 239 -- 19,436 x 219 both, 217 columns compared,
#:      **max abs diff 0.0**;
#:   3. of the 118 columns the whitelist drops, **none** is named by anything in
#:      `analyses/`, `features/` or `modeling/` (the two greps that matched were
#:      a plot frame's own `selected_answer` column and `screen("paragraph")`).
#:
#: Worth repeating the second one after any change here: it is the cheap version
#: of the full two-run comparison and it catches the same mistake.
PUBLISHED_NEEDS = [
    C.NUM_OF_SELECTS,
    C.DOMINANT_EYE_COLUMN,
    C.IA_LABEL,
    C.INTEREST_AREA_FIXATION_SEQUENCE,
    C.AUXILIARY_SPAN_TYPE_COLUMN,
    C.LIST_COLUMN,
    C.ONESTOPQA_QUESTION_ID,
    C.SESSION_ID,
    C.TRIAL_NUMBER,
    # -- Study 2 only. Not in L1's report at all, so `raw_whitelist` intersects
    # them away for L1 and keeps them for KnowQA: the declaration is universal,
    # the effect is per-dataset. Exactly the mechanism section 3.4 describes.
    #
    # ⚠️ **These were missing until 2026-10-13 and it cost a rebuild.** The step-2
    # proof was run against L1's 157-column header; KnowQA's is **342**, and the
    # whitelist silently dropped `regime` and `confidence_rating` -- Study 2's
    # entire manipulation and its outcome measure -- from a rebuilt KnowQA. The
    # run completed, wrote its tables and reported a sensible target distribution.
    # Nothing failed.
    #
    # **The lesson, and it generalises past this list:** a whitelist proved on one
    # dataset is not proved. Acceptance 3 now has to run per dataset, because the
    # thing being tested is `declared_needs() & header`, and the header is the
    # half that varies.
    C.REGIME_COLUMN,
    C.REGIME_LIST_COLUMN,
    C.REGIME_ORDER_COLUMN,
    C.REGIME_ORDER_POSITION_COLUMN,
    C.CONFIDENCE_RATING_COLUMN,
    C.CONFIDENCE_RATING_TXT_COLUMN,
    C.ARTICLE_GROUP_COLUMN,
    C.REPHRASED_ANSWER_A_COLUMN,
]


def declared_needs() -> set:
    """Every column some declaration says is read, produced or published."""
    needs = set(PIPELINE_NEEDS) | set(ATTACH_NEEDS) | set(PUBLISHED_NEEDS)
    for entry in FUNCTION_REGISTRY.values():
        needs |= set(entry["needs"])
    return needs


def raw_whitelist(header) -> list:
    """The raw columns to read, as `usecols`.

    Intersecting with the report's own header sorts raw from produced without
    anything being hand-classified: a column like `..._z` is produced by a base
    feature and simply is not in the header, so it drops out by itself.
    """
    return sorted(declared_needs() & set(header))


def whitelist_report(header) -> str:
    """What `raw_whitelist` keeps and drops, for reading before trusting it.

    Printed rather than asserted, on purpose. The whitelist is a claim that
    nothing reads a column, and `conventions.md` wants that kind of claim visible
    before it can change a result -- so this exists to be eyeballed while the
    filter is still switched off.
    """
    header = list(header)
    keep = raw_whitelist(header)
    drop = sorted(set(header) - set(keep))
    lines = [
        f"raw columns in report : {len(header)}",
        f"  kept (whitelist)    : {len(keep)}",
        f"  not read            : {len(drop)}",
        "",
        "not read:",
    ]
    lines += [f"    {c}" for c in drop]
    return "\n".join(lines)


#: What each entry adds to the frame. Used by `check_registry_order` so the walk
#: can be run without a frame; `add_base_features` / `generate_new_row_features`
#: observe the real thing at run time.
#: What each entry adds to the frame, and **at what grain the result lives.**
#:
#: Two declarations in one, deliberately -- the order check needs the column
#: names and the grain split needs the grains, and splitting them into two maps
#: would let them disagree. Grain is `"row"`, `"area"` or `"trial"`, and it is a
#: statement about the DATA, not about how it was computed: `add_IA_screen_location`
#: is `kind: "row"` because it works in place on the row, yet `question_len` and
#: `n_interest_areas` are trial-grain -- one value per trial, copied onto ~40 rows.
#: Conflating the two is what left 75 trial-constant columns on the interest-area
#: table (grain proposal section 5b).
_PRODUCES = {
    # -- row features: computed in place, but most of what they produce is trial-grain
    "add_text_id": {C.TEXT_ID_COLUMN: "trial"},
    "add_text_id_with_q": {C.TEXT_ID_WITH_Q_COLUMN: "trial"},
    "add_is_correct": {C.IS_CORRECT_COLUMN: "trial"},
    "add_answer_text_columns": {"answer_" + lab: "trial" for lab in C.ANSWER_LABELS},
    "add_IA_screen_location": {
        # the area assignment is the one genuinely per-word thing here: it says
        # which area THIS word belongs to
        C.AREA_SCREEN_LOCATION: "row",
        # the rest are per-trial properties of the stimulus
        "question_len": "trial",
        "1_len": "trial",
        "2_len": "trial",
        "3_len": "trial",
        "4_len": "trial",
        "n_interest_areas": "trial",
        "question_tokens": "trial",
        "1_tokens": "trial",
        "2_tokens": "trial",
        "3_tokens": "trial",
        "4_tokens": "trial",
    },
    "add_IA_answer_label": {C.AREA_LABEL_COLUMN: "row"},
    "add_selected_answer_label": {C.SELECTED_ANSWER_LABEL_COLUMN: "trial"},
    "add_zscored_pupil_columns": {
        C.IA_MAX_FIX_PUPIL_SIZE + "_z": "row",
        C.IA_MIN_FIX_PUPIL_SIZE + "_z": "row",
        C.IA_AVERAGE_FIX_PUPIL_SIZE + "_z": "row",
    },
    "add_total_answering_RT_normalized": {
        C.TOTAL_ANSWERING_RT_NORMALIZED: "trial",
        "total_words_on_screen": "trial",
    },
    # -- area features: one value per (trial, area)
    "create_mean_area_dwell_time": {C.MEAN_DWELL_TIME: "area"},
    "create_mean_area_fix_count": {C.MEAN_FIXATIONS_COUNT: "area"},
    "create_mean_first_fix_duration": {C.MEAN_FIRST_FIXATION_DURATION: "area"},
    "create_skip_rate": {
        C.SKIP_RATE: "area",
        # the per-word indicator the mean is taken over -- genuinely row-grain,
        # and the one intermediate that leaks into the saved table (T3.11)
        C.AREA_SKIPPED: "row",
    },
    "create_dwell_proportions": {
        C.AREA_DWELL_PROPORTION: "area",
        C.TOTAL_IA_DWELL_TIME: "area",
        C.TOTAL_TRIAL_DWELL_TIME: "trial",
    },
    "create_mean_pupil_size_metrics": {
        C.MEAN_MAX_FIX_PUPIL_SIZE: "area",
        C.MEAN_MIN_FIX_PUPIL_SIZE: "area",
        C.MEAN_AVG_FIX_PUPIL_SIZE: "area",
        C.MEAN_MAX_FIX_PUPIL_SIZE_Z: "area",
        C.MEAN_MIN_FIX_PUPIL_SIZE_Z: "area",
        C.MEAN_AVG_FIX_PUPIL_SIZE_Z: "area",
    },
    "create_first_encounter_pupil_size": {
        C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE: "area",
        C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z: "area",
    },
    "create_simplified_visit_counts": {
        C.NUM_LABEL_VISITS: "area",
        C.NUM_LOC_VISITS: "area",
    },
    # -- trial features: one value per trial
    "create_last_area_and_location_visited": {
        C.LAST_VISITED_LABEL: "trial",
        C.LAST_VISITED_LOCATION: "trial",
    },
    "create_fixation_sequence_tags": {
        C.FIX_SEQUENCE_BY_LABEL: "trial",
        C.FIX_SEQUENCE_BY_LOCATION: "trial",
    },
    "create_simplified_fixation_tags": {
        C.SIMPLIFIED_FIX_SEQ_BY_LABEL: "trial",
        C.SIMPLIFIED_FIX_SEQ_BY_LOCATION: "trial",
    },
    "create_seq_len": {"seq_len": "trial"},
    "create_has_xyx": {"has_xyx": "trial"},
    "create_has_xyxy": {"has_xyxy": "trial"},
    "create_longest_alt_answer_run": {"longest_alt_answer_run": "trial"},
    "create_trial_mean_dwell": {"trial_mean_dwell": "trial"},
}


def check_kinds_match_join_columns() -> None:
    """Assert each entry's `kind` agrees with the grain its `join_columns` imply.

    `join_columns` has always encoded the real grain -- an entry keyed on
    `area_label` produces one row per area, one keyed on the trial pair produces
    one row per trial -- but `kind` was assigned by hand and could disagree.

    It did. The stage-3 rename turned every `group` into `area` in one pass, and
    three entries are not area-grain at all: `create_last_area_and_location_visited`,
    `create_fixation_sequence_tags` and `create_simplified_fixation_tags` all join
    on the trial pair alone. Caught 2026-10-13 before step 4, which splits the
    artifacts by exactly this field and would have written three trial-grain
    blocks into the area table.

    Cheap, and it makes that class of mistake impossible to reintroduce.
    """
    for name, entry in FUNCTION_REGISTRY.items():
        kind = entry["kind"]
        if kind not in KINDS:
            raise ValueError(f"{name!r}: unknown kind {kind!r}; known: {KINDS}")
        if kind == "row":
            continue
        join = entry.get("default_kwargs", {}).get("join_columns") or []
        implied = "area" if C.AREA_LABEL_COLUMN in join else "trial"
        if kind != implied:
            raise ValueError(
                f"{name!r} is registered as kind={kind!r} but joins on "
                f"{join}, which is {implied}-grain. One of the two is wrong."
            )


def check_registry_order(header, function_names=None) -> None:
    """Walk the registry in order; raise if an entry needs what nothing produced.

    This **verifies** the existing order rather than computing one, which is how
    it meets T3.24's intent while leaving T3.24's dependency graph declined
    (stage D section 3.4). It is also the loud failure T3.11's undeclared ordering
    dependency has never had: `create_first_encounter_pupil_size` works only
    because `create_mean_first_fix_duration` coerced a column first, and running a
    hand-picked subset out of order currently fails with a type error somewhere
    downstream rather than here.
    """
    available = set(header)
    for name in list(function_names or FUNCTION_REGISTRY):
        entry = FUNCTION_REGISTRY[name]
        missing = set(entry["needs"]) - available
        if missing:
            raise KeyError(
                f"registry order: {name!r} needs {sorted(missing)}, which "
                f"nothing before it produces. Either the order is wrong or a "
                f"`needs` list is."
            )
        available |= set(_PRODUCES.get(name, {}))

