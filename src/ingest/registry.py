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

> **Stage D will revisit exactly this.** The registry is about to grow a third
> kind (`trial`), at which point it describes the whole generator rather than the
> IA half, and "the pipeline layer is called `ingest/`" starts to read oddly. That
> is a naming question for stage D, not a reason to leave the layering broken now.
"""

from src.config import columns as C
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


FUNCTION_REGISTRY = {
    # Base features
    "add_text_id": {
        "callable": add_text_id,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_text_id_with_q": {
        "callable": add_text_id_with_q,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_is_correct": {
        "callable": add_is_correct,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_answer_text_columns": {
        "callable": add_answer_text_columns,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_IA_screen_location": {
        "callable": add_IA_screen_location,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_IA_answer_label": {
        "callable": add_IA_answer_label,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_selected_answer_label": {
        "callable": add_selected_answer_label,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_zscored_pupil_columns": {
        "callable": add_zscored_pupil_columns,
        "default_kwargs": {},
        "kind": "base",
    },
    "add_total_answering_RT_normalized": {
        "callable": add_total_answering_RT_normalized,
        "default_kwargs": {},
        "kind": "base",
    },
    # Group-level functions
    "create_mean_area_dwell_time": {
        "callable": create_mean_area_dwell_time,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_area_fix_count": {
        "callable": create_mean_area_fix_count,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_first_fix_duration": {
        "callable": create_mean_first_fix_duration,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_skip_rate": {
        "callable": create_skip_rate,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_dwell_proportions": {
        "callable": create_dwell_proportions,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_mean_pupil_size_metrics": {
        "callable": create_mean_pupil_size_metrics,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_first_encounter_pupil_size": {
        "callable": create_first_encounter_pupil_size,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
    },
    "create_last_area_and_location_visited": {
        "callable": create_last_area_and_location_visited,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    ## heavy iternal data loading here. Might fix later, slow for now.
    "create_fixation_sequence_tags": {
        "callable": create_fixation_sequence_tags,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID],
            # Forwarded to the function; main() overrides it with its fixations_path.
            "fix_path": FIX_ANSWERS_PATH,
        },
        "kind": "group",
    },
    "create_simplified_fixation_tags": {
        "callable": create_simplified_fixation_tags,
        "default_kwargs": {"join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID]},
        "kind": "group",
    },
    "create_simplified_visit_counts": {
        "callable": create_simplified_visit_counts,
        "default_kwargs": {
            "join_columns": [C.TRIAL_ID, C.PARTICIPANT_ID, C.AREA_LABEL_COLUMN]
        },
        "kind": "group",
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
            if entry.get("kind") == "base"
        ]

    # Case 2: explicit list → validate and return only those
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "base":
                raise ValueError(
                    f"Function '{name}' is not registered as a base feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown base function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "base":
                raise ValueError(
                    f"Function '{name}' is not registered as a base feature."
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
            if entry.get("kind") == "group"
        ]

    # Case 2: explicit list
    resolved = []
    for item in name_list:
        if isinstance(item, str):
            name = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(
                    f"Function '{name}' is not registered as a group feature."
                )
            resolved.append((entry["callable"], entry.get("default_kwargs", {})))
            continue

        if isinstance(item, tuple) and len(item) == 2:
            name, user_kwargs = item
            if name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown group function: {name}")
            entry = FUNCTION_REGISTRY[name]
            if entry.get("kind") != "group":
                raise ValueError(
                    f"Function '{name}' is not registered as a group feature."
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


def generate_new_row_features(functions, df, default_join_columns=None, verbose=True):
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
        result_df = result_df.merge(new_features_df, on=join_columns, how="left")

    return result_df

