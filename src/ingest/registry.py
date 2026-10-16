"""The list of features the prep pipeline builds, and the code that runs them.

`FUNCTION_REGISTRY` has one entry per feature:

    callable        the function that builds it
    needs           the columns it reads
    default_kwargs  settings passed to it; for "area" and "trial" features this
                    includes `join_columns`, the columns its result is joined
                    back on
    kind            "row"   -- adds columns to each word's row
                    "area"  -- computes one value per trial and screen area
                    "trial" -- computes one value per trial

**To add a feature:** add an entry to `FUNCTION_REGISTRY`, and say what it adds
in `_PRODUCES` (further down), with the level each new column lives at.

**Entries run in the order listed.** Some read columns that earlier ones
create -- `needs` says which. `check_registry_order` confirms an order works
without running anything.

**At the end of a run**, every column goes into one of three output tables --
one row per word, per area, or per trial -- according to its declared level.
See `column_grain` and `split_by_grain`.

The registry lives in `ingest/` rather than `features/` because it names
functions from both folders, and `features/` is not allowed to import `ingest/`.
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


# Keys in an area/trial entry's settings that tell the runner how to join the
# result back, rather than being passed to the feature function.
GROUP_ORCHESTRATION_KEYS = {"join_columns"}


#: The two kinds that summarise many words into fewer rows. Both are run by
#: `generate_new_row_features`, which joins each result back onto the word rows
#: so that later features can read it.
GROUP_KINDS = ("area", "trial")

#: Every kind, from finest to coarsest.
KINDS = ("row",) + GROUP_KINDS


FUNCTION_REGISTRY = {
    # -- row features: add columns to each word's row ----------------------
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
            # read only to name the trials involved if its check fails
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
            # each word is placed by where its box sits on screen
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
    # -- area and trial features: summarise words into one value per area or
    # per trial, and are joined back on `join_columns` --------------------
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
            # made by add_zscored_pupil_columns, so that entry must run first
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
    # Works from the fixation report (one row per fixation). main() loads that
    # report once and passes it in as `fix_path`.
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
            # The default report; main() replaces it with the one it loaded.
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
    # -- statistics of the merged area sequence, one value per trial
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
        # average looking time per word on the whole screen; computed from the
        # area-level mean_dwell_time, which gives the same number
        "callable": create_trial_mean_dwell,
        "needs": [C.MEAN_DWELL_TIME, *[C.TRIAL_ID, C.PARTICIPANT_ID]],
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "trial",
    },
}


def resolve_base_functions(name_list=None):
    """The row features to run, as (function, settings) pairs.

    With no list, every row feature in registry order. Otherwise only the named
    ones, each given as "name" or ("name", {settings to override}).
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
    """The area and trial features to run, as (function, settings) pairs.

    With no list, every area and trial feature in registry order. Otherwise only
    the named ones, each given as "name" or ("name", {settings to override}).
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
    """Run the row features in order, each adding columns to the word rows.

    Each function takes the frame and returns it with its new columns. Works on
    a copy, so the caller's frame is not changed.
    """

    out = df.copy()
    for func, kwargs in functions:
        if verbose:
            print(f"Running: {func.__name__}")
        out = func(out, **kwargs) if kwargs else func(out)
    return out.reset_index(drop=False)


def _registry_name_for(func) -> str:
    """The registry name of a function, or its own name if it is not registered."""
    for name, entry in FUNCTION_REGISTRY.items():
        if entry["callable"] is func:
            return name
    return getattr(func, "__name__", repr(func))


#: Columns made outside the registry, by the two extra steps at the end of
#: `ingest/build.py`: reading times, and the area looked at before each button
#: press. Both build one row per trial, so these are all trial-level.
#:
#: The reading times are stored one column per area (RT_pure_answer_A, ...) on
#: the trial's row, which is why they count as trial-level rather than area-level.
#: The list is generated from the answer screen's areas, so adding an area cannot
#: leave a column out.
ATTACH_PRODUCES: dict = {
    **{
        f"{metric}_{region}": "trial"
        for metric in C.RT_TFD_METRICS
        for region in ANSWERS.regions
    },
    C.LAST_LBL_BEFORE_SELECT: "trial",
    C.LAST_LBL_BEFORE_CONFIRM: "trial",
}

#: Columns that come straight from the raw report and have one value per trial,
#: not per word: which item was shown, the trial's condition, what the
#: participant did, and how the trial was recorded. They go on the trial table.
#:
#: A raw column not listed here stays on the word table, copied onto every word.
#:
#: `participant_id` and `TRIAL_INDEX` are not listed: they identify a trial, and
#: every table carries them.
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
    # -- Study 2 only: the knowledge regime, its counterbalancing, and the
    # confidence rating. Not in Study 1's report, so these lines do nothing there.
    #
    # A Study 2 column needs a line here AND in `PUBLISHED_NEEDS` below: that
    # list makes the pipeline read the column, this one puts it on the trial
    # table.
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
    """Which table each column belongs on: `{column: "row" | "area" | "trial"}`.

    Built from the three lists above: `_PRODUCES`, `ATTACH_PRODUCES` and
    `RAW_COLUMN_GRAIN`. A column not named in any of them is treated as
    per-word and stays on the word table. That is the safe way round: a
    forgotten line leaves a column where it is, instead of moving it to a
    coarser table where differing values would be collapsed into one.

    These levels are declared, never guessed from the data.
    `check_declared_grains` can compare them against the data to catch a wrong
    declaration.
    """
    grains = {}
    for produced in _PRODUCES.values():
        grains.update(produced)
    grains.update(ATTACH_PRODUCES)
    grains.update(RAW_COLUMN_GRAIN)
    return grains


def split_by_grain(result_df, collected: dict = None):
    """Split the wide word-level frame into the three output tables.

    During a run, every area and trial value is copied onto each word's row.
    This takes that wide frame apart again, using `column_grain()` to send each
    column to its table:

        interest_areas  one row per word
        areas           one row per trial and screen area
        trials          one row per trial

    Every table keeps participant_id and TRIAL_INDEX (and the area tables keep
    area_label), so they can always be joined back together.

    `collected` is accepted but not used.
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
    """Stop with an error if any of the three tables lacks the columns needed
    to join them back together. Run automatically when the tables are saved."""
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
    """Check each column's declared level against the data.

    A column declared "trial" must have one value per trial, and one declared
    "area" one value per trial and area. Returns a list of the columns that
    break this; an empty list means every declaration holds. `sample_trials`
    checks only that many trials, to save time.

    Not run automatically -- call it by hand after changing a declaration.
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
    """Stop with an error if splitting into three tables lost any column.

    Run automatically when the tables are saved.
    """
    got = set(ia.columns) | set(areas.columns) | set(trials.columns)
    missing = set(result_df.columns) - got
    if missing:
        raise AssertionError(
            f"grain split lost {len(missing)} column(s): {sorted(missing)[:10]}"
        )


def generate_new_row_features(
    functions, df, default_join_columns=None, verbose=True, collect=None
):
    """Run the area and trial features in order, joining each result back onto
    the word rows.

    `functions` is a list of (function, settings) pairs. `join_columns` in the
    settings says which columns the result is joined back on; every other
    setting is passed to the function (for example `fix_path`).

    Each result is joined back before the next feature runs, so later features
    can read earlier ones' columns. Returns the word-level frame with all the
    new columns added.

    `collect`, if given a dict, receives each feature's own result before it is
    joined back. Nothing passes it at present.
    """
    if default_join_columns is None:
        default_join_columns = [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]

    result_df = df.copy()

    # The report writes "." for "no value". Turn those columns into numbers once,
    # here, before any measurement runs, so every measurement sees the same
    # numeric columns no matter which ones you choose to run.
    #
    # It changes the frame in place on purpose: the numeric columns are what the
    # saved word table should hold. Pupil columns are left alone (the default
    # `pupil=False`) because an earlier row feature has already converted them to
    # millimetres and z-scored them.
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
        # Keep the feature's own result, before it is copied onto every word, if
        # the caller asked for it. This does not change the join below.
        if collect is not None:
            collect[_registry_name_for(func)] = new_features_df
        result_df = result_df.merge(new_features_df, on=join_columns, how="left")

    return result_df


# ---------------------------------------------------------------------------
# Which raw columns to read
#
# The raw report has far more columns than anything uses, so the pipeline reads
# only the columns something declares it needs (see `raw_whitelist`). A column
# that nothing declares is not read at all -- and nothing will complain that it
# is missing. So every reader of raw columns has to be declared in one of four
# places:
#
#   the registry's `needs`  what each feature reads
#   PIPELINE_NEEDS          what ingest/build.py itself reads (to drop trials
#                           and to split hunters from gatherers)
#   ATTACH_NEEDS            what the reading-time and button-press steps read
#   PUBLISHED_NEEDS         raw columns nothing here reads, but later code does,
#                           so they must be carried into the output tables
#
# A new dataset can bring columns the others lack. Before trusting a run on one,
# read `whitelist_report(header)` for its report.
# ---------------------------------------------------------------------------

#: Columns ingest/build.py reads directly.
PIPELINE_NEEDS = [
    C.REPEATED_TRIAL_COLUMN,
    C.PRACTICE_TRIAL_COLUMN,
    C.QUESTION_PREVIEW_COLUMN,
]

#: Columns the reading-time and button-press steps read.
ATTACH_NEEDS = [
    C.IA_LAST_FIXATION_TIME,
    C.IA_FIRST_FIXATION_TIME,
    C.IA_LAST_FIXATION_DURATION,
]

#: Raw columns that later code reads from the output tables -- for example the
#: number of selection presses (a model feature) and the dominant eye (the
#: scan-strategy analysis).
#:
#: To check after any change that the whitelist drops nothing that matters:
#: build `model_ready` once normally and once with
#: `ingest/build.py::main(filter_raw_columns=False)`, and compare the two.
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
    # -- Study 2 only. Study 1's report does not have these, so for Study 1 they
    # are simply not found and nothing happens.
    #
    # These lines are what bring the regime and the confidence rating into the
    # tables. Leave one out and a Study 2 run still finishes normally -- just
    # without that column.
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
    """Every column named in any of the four lists above."""
    needs = set(PIPELINE_NEEDS) | set(ATTACH_NEEDS) | set(PUBLISHED_NEEDS)
    for entry in FUNCTION_REGISTRY.values():
        needs |= set(entry["needs"])
    return needs


def raw_whitelist(header) -> list:
    """The raw columns to read: those declared needed AND present in this report.

    Taking only names that appear in the report's own header means columns the
    pipeline creates itself (like the `..._z` pupil columns) drop out
    automatically. Pass the result to `pd.read_csv(usecols=...)`.
    """
    return sorted(declared_needs() & set(header))


def whitelist_report(header) -> str:
    """A readable list of which raw columns are read and which are left out.

    Look through it before trusting a run on a new dataset: anything in the
    "not read" list is invisible to the whole pipeline. Not run automatically.
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


#: What each registry entry adds, and which table each new column belongs on:
#: "row" (per word), "area" (per trial and area) or "trial" (per trial).
#:
#: This describes the data, not how it was computed. `add_IA_screen_location`
#: works row by row, but most of what it adds -- `question_len`,
#: `n_interest_areas` -- has one value per trial, so it goes on the trial table.
#:
#: Used both to split the output into three tables and by `check_registry_order`.
_PRODUCES = {
    # -- row features: worked out row by row, but mostly one value per trial
    "add_text_id": {C.TEXT_ID_COLUMN: "trial"},
    "add_text_id_with_q": {C.TEXT_ID_WITH_Q_COLUMN: "trial"},
    "add_is_correct": {C.IS_CORRECT_COLUMN: "trial"},
    "add_answer_text_columns": {"answer_" + lab: "trial" for lab in C.ANSWER_LABELS},
    "add_IA_screen_location": {
        # the only per-word column here: which area this word is in
        C.AREA_SCREEN_LOCATION: "row",
        # the rest describe the trial's text: words per area, and the words
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
        # the per-word "never looked at" flag that skip_rate averages; kept in
        # the word table
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
    """Check that each entry's `kind` matches its `join_columns`.

    An entry joined back on area_label makes one row per area, so it must be
    "area"; one joined on the trial alone must be "trial". The `kind` is typed
    by hand, so the two can disagree; this stops with an error if they do.

    Not run automatically -- call it by hand after editing the registry.
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
    """Check that every entry's `needs` are available when it runs.

    Walks the entries in order, starting from the columns in `header` (the raw
    report), and adds what each entry produces. Stops with an error if an entry
    needs a column that neither the report nor any earlier entry provides --
    meaning either the order is wrong or a `needs` list is. Checks without
    running anything.

    `function_names` checks only those entries, in the order given. Not run
    automatically -- call it by hand after editing the registry.
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

