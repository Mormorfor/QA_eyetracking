"""The answer-screen pipeline: raw reports in, three tables out.

`main()` turns a dataset's raw interest-area and fixation reports into:

    interest_areas.csv   one row per word
    areas_answers.csv    one row per trial and screen area
    trials.csv           one row per trial

plus side files in `interim/aux/`: button presses, the pupil baseline, reading
times and the area looked at before each press.

In order, it:

1. reads the interest-area report and drops repeated-reading and practice trials;
2. reads the fixation report, once (`features/fixations.py`); everything below
   that needs single fixations uses this one table;
3. rebuilds when each answer was selected and confirmed, from the message log;
4. runs the row features, then the area and trial features (`ingest/registry.py`);
5. adds reading times, and the area looked at before each button press;
6. splits the result into the three tables and saves them.

Nothing is reused from an earlier run: every side file is rebuilt each time.

The paragraph screen is processed separately, by `features/paragraph/spans.py`.

This file imports from `features/` as well as `ingest/` because it is where the
pieces are put together. No `features/` module imports it.
"""

import os
from pathlib import Path
import pandas as pd
from src.config import columns as C
from src.config.datasets import (
    ALL_PARTICIPANTS_LAST_PATH,
    ALL_PARTICIPANTS_PROCESSED_PATH,
    BUTTON_CLICKS_PATH,
    FIX_A_TSV_PATH,
    FIX_ANSWERS_PATH,
    IA_ANSWERS_PATH,
    PUPIL_BASELINE_ANSWERS_PATH,
    RT_AND_TFD_PATH,
)
from src.ingest.clicks import run_trial_level_pipeline
from src.features.fixations import load_fixations, match_key_types
from src.features.pupil import get_participant_pupil_stats
from src.features.reading_times import build_rt_and_tfd
from src.features.last_visited import compute_last_area_labels
from src.ingest.readers import load_raw_answers_data
from src.features.pupil import add_zscored_pupil_columns
from src.features.sequences import create_fixation_sequence_tags
from src.ingest.registry import (
    add_base_features,
    check_grain_split_is_lossless,
    check_join_keys,
    raw_whitelist,
    split_by_grain,
    generate_new_row_features,
    resolve_base_functions,
    resolve_group_functions,
)


# The fixation-report columns this pipeline reads: when each fixation started
# and how long it lasted, its word and nearest word, its pupil size, and the
# number of selection presses.
FIXATION_COLUMNS = [
    C.CURRENT_FIX_START,
    C.CURRENT_FIX_DURATION,
    C.CURRENT_FIX_INTEREST_AREA_ID,
    C.NEAREST_IA,
    C.CURRENT_FIX_PUPIL_SIZE,
    C.NUM_OF_SELECTS,
]

# When the fixation report also holds the message log (Study 2), the button
# presses are rebuilt from these columns of it too.
MESSAGE_COLUMNS = [
    C.ALL_ANSWERS,
    C.CURRENT_FIX_MSG_LIST_TEXT,
    C.CURRENT_FIX_MSG_LIST_TIME,
    C.NEXT_SAC_MSG_LIST_TEXT,
    C.NEXT_SAC_MSG_LIST_TIME,
]


def _attach_last_label_features(
    df: pd.DataFrame,
    fixations: pd.DataFrame,
    save_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add the area looked at just before the answer was selected, and just
    before it was confirmed (`last_lbl_before_select`, `last_lbl_before_confirm`).

    Worked out from the press times in the button-press table at
    `button_clicks_path` and the answer screen's `fixations`. If `save_path` is
    given, the per-trial result is also saved there.
    """
    if verbose:
        print("Computing last-area-label features…")

    last_df = compute_last_area_labels(
        df, fixations, trial_level_path=button_clicks_path, verbose=verbose
    )

    if save_path is not None:
        _save(last_df, save_path, label="last-area labels", verbose=verbose)

    merge_cols = [C.PARTICIPANT_ID, C.TRIAL_ID]
    rename_map = {
        C.AREA_LABEL_BEFORE_SELECT: C.LAST_LBL_BEFORE_SELECT,
        C.AREA_LABEL_BEFORE_CONFIRM: C.LAST_LBL_BEFORE_CONFIRM,
    }

    last_df = (
        last_df[merge_cols + list(rename_map.keys())]
        .drop_duplicates(subset=merge_cols)
        .rename(columns=rename_map)
    )

    return df.merge(last_df, on=merge_cols, how="left")


def _attach_rt_and_tfd_features(
    df: pd.DataFrame,
    fixations: pd.DataFrame,
    save_path: Path = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add the reading times for each answer-screen area, one row per trial.

    Built by `features/reading_times.py::build_rt_and_tfd` from the word table
    and this dataset's `fixations`. Adds the RT_*, TFD_* and TimeSinceOffset_*
    columns. If `save_path` is given, the reading-time table is also saved there.
    """
    if verbose:
        print("Computing RT/TFD features…")

    rt_df = build_rt_and_tfd(
        all_participants=df,
        fixations=fixations,
        save=save_path is not None,
        output_path=save_path,
        verbose=verbose,
    )

    merge_cols = [C.PARTICIPANT_ID, C.TRIAL_ID]
    rt_df = rt_df.drop_duplicates(subset=merge_cols)
    return df.merge(rt_df, on=merge_cols, how="left")


def _save(df: pd.DataFrame, output_path: Path, label: str = "", verbose: bool = True):
    """Write `df` to `output_path`, creating parent directories as needed."""
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    if verbose:
        print(f"Saving {label} features to: {output_path}")
    df.to_csv(output_path, index=False)


def _save_splits(
    df: pd.DataFrame,
    split_column: str,
    base_output_path: Path,
    split_output_paths: dict = None,
    verbose: bool = True,
):
    """Save one file per value of `split_column` (e.g. hunters and gatherers).

    The file for a value is taken from `split_output_paths` ({value: path}) if
    given; otherwise it is named "<stem>_<split_column>_<value><suffix>" next to
    `base_output_path`.
    """
    if split_column not in df.columns:
        raise ValueError(
            f"split_column '{split_column}' not found in processed data."
        )

    base_output_path = Path(base_output_path)
    split_output_paths = split_output_paths or {}

    for value, subset in df.groupby(split_column):
        if value in split_output_paths:
            out_path = Path(split_output_paths[value])
        else:
            safe_value = str(value).replace(os.sep, "_").replace(" ", "_")
            out_path = base_output_path.with_name(
                f"{base_output_path.stem}_{split_column}_{safe_value}"
                f"{base_output_path.suffix}"
            )
        _save(subset, out_path, label=f"{split_column}={value}", verbose=verbose)


def save_grain_tables(
    result_df: pd.DataFrame,
    dataset,
    split_column: str = None,
    verbose: bool = True,
) -> dict:
    """Split the processed frame into the three tables, check the split, and save.

    Each column goes to the table its declared level says (see
    `registry.column_grain`). Before anything is written, two checks run: every
    table has the columns needed to rejoin it, and no column was lost. Either
    failing stops the run.

    With `split_column` set, each group's subset is saved in the same call (e.g.
    `trials__hunters.csv`), so the subsets can never be older than the full
    tables.

    Answer screen only. The paragraph screen's features are built and saved by
    `features/paragraph/spans.py`.
    """
    ia, areas, trials = split_by_grain(result_df)
    check_join_keys(ia, areas, trials)
    check_grain_split_is_lossless(result_df, ia, areas, trials)

    written = {}
    for grain, frame in (
        ("interest_areas", ia),
        ("areas_answers", areas),
        ("trials", trials),
    ):
        path = getattr(dataset, grain)
        _save(frame, path, label=grain, verbose=verbose)
        written[grain] = path

        if split_column and split_column in result_df.columns:
            key = [C.PARTICIPANT_ID, C.TRIAL_ID]
            # The split column is already on `trials` but not on the other two
            # tables. Giving it a temporary name that cannot clash means the same
            # join works for all three.
            facet_col = "__grain_facet__"
            facets = (
                result_df[key + [split_column]]
                .drop_duplicates(subset=key)
                .rename(columns={split_column: facet_col})
            )
            merged = frame.merge(facets, on=key, how="left")
            for value, subset in merged.groupby(facet_col):
                name = {True: "hunters", False: "gatherers"}.get(value, str(value))
                out = dataset.grain_split(grain, name)
                _save(
                    subset[frame.columns], out,
                    label=f"{grain}__{name}", verbose=verbose,
                )
                written[f"{grain}__{name}"] = out

    if verbose:
        print(
            f"  grains: interest_areas {ia.shape}, areas_answers {areas.shape}, "
            f"trials {trials.shape}"
        )
    return written


def _process(
    df: pd.DataFrame,
    base_funcs,
    group_funcs,
    fixations: pd.DataFrame,
    add_last: bool = True,
    add_rts: bool = True,
    last_labels_path: Path = None,
    rt_and_tfd_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every feature on the word-level frame and return it with the new columns.

    First the row features, then the area and trial features, then (if asked)
    the area looked at before each button press and the reading times, both
    from `fixations`. `last_labels_path` and `rt_and_tfd_path`, when given, are
    where those last two are also saved."""
    if verbose:
        print(f"\nProcessing {label} (row-level)…")
    out = add_base_features(df, base_funcs, verbose=verbose)

    if verbose:
        print(f"Applying group-level features for {label}…")
    out = generate_new_row_features(group_funcs, out)

    if add_last:
        out = _attach_last_label_features(
            out,
            fixations,
            save_path=last_labels_path,
            button_clicks_path=button_clicks_path,
            verbose=verbose,
        )

    if add_rts:
        out = _attach_rt_and_tfd_features(
            out,
            fixations,
            save_path=rt_and_tfd_path,
            verbose=verbose,
        )

    return out


def main(
    ia_answers_path: Path = IA_ANSWERS_PATH,
    output_path: Path = ALL_PARTICIPANTS_PROCESSED_PATH,
    fixations_path: Path = FIX_ANSWERS_PATH,
    split_column: str = None,
    split_output_paths: dict = None,
    add_last: bool = True,
    add_rts: bool = True,
    pupil_fixations_path: Path = None,
    pupil_stats_path: Path = PUPIL_BASELINE_ANSWERS_PATH,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    button_clicks_fix_tsv_path: Path = FIX_A_TSV_PATH,
    button_clicks_msg_participant_col: str = C.RECORDING_SESSION_LABEL,
    all_answers_is_cumulative: bool = True,
    save_auxiliary: bool = True,
    last_labels_path: Path = ALL_PARTICIPANTS_LAST_PATH,
    rt_and_tfd_path: Path = RT_AND_TFD_PATH,
    remove_repeats: bool = True,
    remove_practice: bool = True,
    base_function_names: list = None,
    group_function_names: list = None,
    filter_raw_columns: bool = True,
    dataset=None,
    write_broadcast_table: bool = False,
    verbose: bool = True,
):
    """Run the whole answer-screen pipeline, from the raw reports.

    Needs nothing from an earlier run, and reuses nothing: every side file is
    rebuilt each time and saved only as a record of what this run used.

    **What gets saved:**

    - `dataset=` -- pass the dataset's record (`config/datasets.py`) to save
      the three tables (interest_areas, areas_answers, trials). **Without it, no
      table is saved** -- only the side files below.
    - `write_broadcast_table=True` -- also save the single wide table
      (one row per word, everything copied onto it) to `output_path`.
    - `split_column` -- also save each group's subset, e.g.
      `split_column=question_preview` gives `trials__hunters.csv` and
      `trials__gatherers.csv`. `split_output_paths` ({value: path}) names the
      subsets of the wide table, and is only used with `write_broadcast_table`.
    - `save_auxiliary` -- save the side files in `interim/aux/` (on by default).

    **Inputs:**

    - `ia_answers_path` -- the interest-area report (one row per word).
    - `fixations_path` -- the matching fixation report (one row per fixation),
      read once. Used for the button presses, the fixation sequence, the area
      looked at before each press, the reading times and, unless
      `pupil_fixations_path` says otherwise, the pupil baseline.
    - `button_clicks_fix_tsv_path` -- where the message log for the button
      presses is. Study 1 keeps it in a separate TSV export. For a fixation
      report that holds it too (Study 2), pass `button_clicks_fix_tsv_path=None`,
      and set `button_clicks_msg_participant_col` and
      `all_answers_is_cumulative=False` to match it.

    **Side files** (saved when `save_auxiliary=True`):

    - the answer-screen pupil baseline: each participant's typical pupil size,
      worked out from the fixation report -> `pupil_stats_path`. The paragraph
      screen has its own baseline, made in `features/paragraph/spans.py`.
    - the button presses (only when `add_last`) -> `button_clicks_path`.
    - the area looked at before each press (`add_last`) -> `last_labels_path`.
    - the reading times (`add_rts`) -> `rt_and_tfd_path`.

    **Other settings:**

    - `remove_repeats`, `remove_practice` -- drop repeated-reading and practice
      trials (both on by default).
    - `base_function_names`, `group_function_names` -- run only these registry
      entries instead of all of them.
    - `filter_raw_columns` -- read only the raw columns something declares it
      needs (`registry.raw_whitelist`). Pass False to read every column, e.g. to
      check that the filter changes nothing.
    """
    # The pupil baseline comes from the same fixation report as everything else,
    # unless pupil_fixations_path names a different one.
    if pupil_fixations_path is None:
        pupil_fixations_path = fixations_path

    if verbose:
        print(f"\nLoading raw answers from: {ia_answers_path}")

    # Read only the raw columns something declares it needs. The list is built
    # here, not in readers.py, so that readers.py does not depend on the registry.
    # filter_raw_columns=False reads every column instead.
    usecols = None
    if filter_raw_columns:
        header = pd.read_csv(ia_answers_path, nrows=0, engine="python").columns
        usecols = raw_whitelist(header)
        if verbose:
            print(
                f"  reading {len(usecols)} of {len(header)} raw columns "
                f"(whitelist; pass filter_raw_columns=False for all of them)"
            )

    df_answers = load_raw_answers_data(ia_answers_path, usecols=usecols)

    if remove_repeats:
        df_answers = df_answers[
            df_answers[C.REPEATED_TRIAL_COLUMN] == False
        ].copy()
    if remove_practice:
        df_answers = df_answers[
            df_answers[C.PRACTICE_TRIAL_COLUMN] == False
        ].copy()

    if verbose:
        print("\nResolving processing function lists…")

    base_funcs = resolve_base_functions(base_function_names)
    group_funcs = resolve_group_functions(group_function_names)

    # The fixation report is read once, here, and the same table is used for the
    # button presses, the pupil baseline, the fixation sequence, the area looked
    # at before each press and the reading times. Its participant and trial ids
    # are given the word table's types, so the two can be joined.
    if verbose:
        print(f"\nReading the fixation report: {fixations_path}")
    columns = FIXATION_COLUMNS + (MESSAGE_COLUMNS if button_clicks_fix_tsv_path is None else [])
    fixations_df = match_key_types(
        load_fixations(fixations_path, columns, verbose=verbose), df_answers
    )

    # The button presses are needed only for "the area looked at before each
    # press" (add_last), so they are built only then -- always from the raw
    # reports, never reused from an earlier run. A saved press table from an
    # earlier run can look perfectly valid and still be out of date.
    if add_last:
        if verbose:
            print("\nBuilding button-click data from raw…")
        run_trial_level_pipeline(
            fixations=fixations_df,
            fix_tsv_path=button_clicks_fix_tsv_path,
            output_csv_path=Path(button_clicks_path),
            msg_participant_col=button_clicks_msg_participant_col,
            all_answers_is_cumulative=all_answers_is_cumulative,
            verbose=verbose,
        )

    # Give the fixation table to the fixation-sequence feature.
    if any(func is create_fixation_sequence_tags for func, _ in group_funcs):
        group_funcs = [
            (func, {**kwargs, "fix_path": fixations_df})
            if func is create_fixation_sequence_tags
            else (func, kwargs)
            for func, kwargs in group_funcs
        ]

    # Work out each participant's typical pupil size on the answer screen, and give
    # it to the pupil z-scoring feature (only if that feature is being run). It is
    # always computed from this run's fixations, never loaded from a file.
    #
    # The paragraph screen gets its own baseline, in features/paragraph/spans.py:
    # the two screens differ in brightness and in task, so a person's typical
    # pupil size is not the same on both.
    if any(func is add_zscored_pupil_columns for func, _ in base_funcs):
        pupil_stats = get_participant_pupil_stats(
            fixations_path=pupil_fixations_path,
            fixations=fixations_df if pupil_fixations_path == fixations_path else None,
            verbose=verbose,
        )
        # Saved as a record of the baseline this run used.
        if save_auxiliary:
            _save(
                pupil_stats, pupil_stats_path,
                label="answer-screen pupil baseline", verbose=verbose,
            )
        base_funcs = [
            (func, {**kwargs, "pupil_stats": pupil_stats})
            if func is add_zscored_pupil_columns
            else (func, kwargs)
            for func, kwargs in base_funcs
        ]

    processed = _process(
        df_answers,
        base_funcs=base_funcs,
        group_funcs=group_funcs,
        add_last=add_last,
        add_rts=add_rts,
        last_labels_path=last_labels_path if save_auxiliary else None,
        rt_and_tfd_path=rt_and_tfd_path if save_auxiliary else None,
        fixations=fixations_df,
        button_clicks_path=button_clicks_path,
        label="all_participants",
        verbose=verbose,
    )

    # The single wide table (all_participants.csv) and its group subsets. Off by
    # default: it holds the same data as the three tables below, but with every
    # trial-level value copied onto each of ~39 word rows -- on Study 1 that is
    # 5.8 GB. Code that wants the wide frame rebuilds it from the three tables
    # with features/build.py::load_model_input.
    if write_broadcast_table:
        _save(processed, output_path, label="all_participants", verbose=verbose)

        if split_column is not None:
            if verbose:
                print(f"\nSaving splits by column: {split_column}")
            _save_splits(
                processed,
                split_column=split_column,
                base_output_path=output_path,
                split_output_paths=split_output_paths,
                verbose=verbose,
            )

    # The three tables. Only saved when the caller passed `dataset=`.
    if dataset is not None:
        if verbose:
            print("\nSplitting into grains…")
        save_grain_tables(
            processed, dataset, split_column=split_column, verbose=verbose
        )

    if verbose:
        print("\n✓ Done.\n")
