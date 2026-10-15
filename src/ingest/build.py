"""The prep pipeline: raw reports in, all_participants.csv out.

`main()` is the composition root for Stage 1. It reads the raw interest-area and
fixation reports, runs the base and group features from `ingest/registry.py` in
registry order, attaches the last-label and RT/TFD blocks, and writes the
IA-level table plus its auxiliary outputs.

Being the composition root, it imports from both layers -- that is its job, and
it is the only module here that does. The rule that matters is the one below it:
no `features/` module imports `ingest/build`, and no `ingest/` primitive
(`readers`, `geometry`, `clicks`) imports `features/`.
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


def _attach_last_label_features(
    df: pd.DataFrame,
    save_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute last-area-label features directly from the processed IA data and
    merge in:
    - C.LAST_LBL_BEFORE_SELECT
    - C.LAST_LBL_BEFORE_CONFIRM

    The features are derived on the fly (via
    src.features.last_visited.compute_last_area_labels) from the trial-level
    button-click table at `button_clicks_path`. If `save_path` is given, the
    computed (unmerged) last-label table is written there as an auxiliary artifact.
    """
    if verbose:
        print("Computing last-area-label features…")

    last_df = compute_last_area_labels(
        df, trial_level_path=button_clicks_path, verbose=verbose
    )

    if save_path is not None:
        _save(last_df, save_path, label="last-area labels", verbose=verbose)

    merge_cols = [C.PARTICIPANT_ID, C.TRIAL_ID]
    rename_map = {
        "area_label_before_last_select": C.LAST_LBL_BEFORE_SELECT,
        "area_label_before_confirm": C.LAST_LBL_BEFORE_CONFIRM,
    }

    last_df = (
        last_df[merge_cols + list(rename_map.keys())]
        .drop_duplicates(subset=merge_cols)
        .rename(columns=rename_map)
    )

    return df.merge(last_df, on=merge_cols, how="left")


def _attach_rt_and_tfd_features(
    df: pd.DataFrame,
    save_path: Path = None,
    fixations_path: Path = FIX_ANSWERS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute RT_pure_/RT_normalized_/TFD_pure_/TFD_normalized_ features directly
    from the processed IA data and merge them in at trial level on
    (participant_id, TRIAL_INDEX).

    The features are derived on the fly (via
    src.features.reading_times.build_rt_and_tfd), reading the run's own fixation
    report at `fixations_path` for the run-based RT. **This step does not touch
    the button-click table.** It used to: run-based RT came from the
    (timestamp, nearest-IA) list that `ingest/clicks.py` copied into that table,
    until `compute_run_based_rt` was deleted on 2026-10-06 in favour of reading
    the fixation report directly (stage D step 3.6). Answer regions only
    -- paragraph RT/TFD moved to `features/paragraph/spans.py` on 2026-09-23
    (`todo.md` T6.1), so this step no longer opens any paragraph report and the
    old `include_paragraph` flag is gone. If `save_path` is given, the computed
    RT/TFD table is written there as an auxiliary artifact.
    """
    if verbose:
        print("Computing RT/TFD features…")

    rt_df = build_rt_and_tfd(
        all_participants=df,
        fixations_path=fixations_path,
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
    """Partition `df` by the unique values of `split_column` and save each subset.

    The target path for a value is taken from `split_output_paths` (a
    {value: path} mapping) when provided; otherwise it is derived from
    `base_output_path` as "<stem>_<split_column>_<value><suffix>".
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
    """Split the broadcast interest-area frame into its three grains and write them.

    Step 4d of `docs/decisions/2026-10-13-feature-grain-proposal.md`. The split is
    driven by `registry.column_grain()` -- declarations, not observation -- and is
    checked before anything is written: keys present at every grain, no column
    lost. A failure here must stop the run rather than write a table that quietly
    drops a column.

    `areas_paragraph` is **not** written here. The paragraph screen is a separate
    pipeline with its own entry point (`features/paragraph/spans.py`), kept separate
    by T6.1, and this function only ever sees the answer screen.

    Group subsets are written in the **same call** as their parent, which is the
    point of doing it here rather than leaving it to a second pass: on 2026-10-13
    `model_ready.csv` was five days newer than its own hunters/gatherers copies,
    because those were a separate rebuild that nothing forced to happen.
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
            # Rename the facet to a name that cannot collide rather than relying
            # on merge suffixes: the split column is itself trial-grain, so it is
            # already present on `trials` and absent from the other two, and a
            # suffix rule has to be right in both cases. This one does not care.
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
    add_last: bool = True,
    add_rts: bool = True,
    last_labels_path: Path = None,
    rt_and_tfd_path: Path = None,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    fixations_path: Path = FIX_ANSWERS_PATH,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run base + group pipelines on `df`, optionally attach last-label and
    RT/TFD features (computed on the fly), and return the enriched DataFrame.

    The last-label features read the trial-level button clicks; RT reads the
    answer fixation report directly (since 2026-10-06, stage D step 6). Paragraph
    RT/TFD is not built here at all -- see `features/paragraph/spans.py` (T6.1).
    `last_labels_path` / `rt_and_tfd_path`, when given, are where the computed
    auxiliary tables are saved."""
    if verbose:
        print(f"\nProcessing {label} (row-level)…")
    out = add_base_features(df, base_funcs, verbose=verbose)

    if verbose:
        print(f"Applying group-level features for {label}…")
    out = generate_new_row_features(group_funcs, out)

    if add_last:
        out = _attach_last_label_features(
            out,
            save_path=last_labels_path,
            button_clicks_path=button_clicks_path,
            verbose=verbose,
        )

    if add_rts:
        out = _attach_rt_and_tfd_features(
            out,
            save_path=rt_and_tfd_path,
            fixations_path=fixations_path,
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
    button_clicks_fix_csv_path: Path = FIX_ANSWERS_PATH,
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
    """
    Full preprocessing pipeline. Runs end-to-end from the raw DATA_RAW reports —
    no previously-generated file is required.

    By default, all trials are processed together (after filtering repeated and
    practice trials) and saved as a single combined CSV to `output_path`
    (all_participants.csv).

    `fixations_path` is the single canonical fixations report for the run. It feeds
    both the participant pupil stats and the fixation-sequence group feature
    (`create_fixation_sequence_tags`); point it at the matching fixations report when
    processing a different raw dataset.

    **Nothing here is cached. Every intermediate is rebuilt on every run**
    (Diana, 2026-10-15). The auxiliary files below are a *record* of what a run
    produced, never an input to the next one. The previous behaviour — reuse
    `button_clicks_data.csv` whenever it happened to exist — is why L1's copy was
    three months older than every other file beside it, having survived the T3.18
    geometry fix and two full rebuilds without anything saying so.

    Intermediate ("auxiliary") artifacts are all derived on the fly and, when
    `save_auxiliary=True`, persisted under the dataset's `interim/aux/`:
    - the answer-screen pupil baseline — always computed from the fixation data at
      `pupil_fixations_path` (which defaults to `fixations_path`), per participant.
      Saved to `pupil_stats_path`. The paragraph screen has its own, separate
      baseline; see `features/paragraph/spans.py`.
    - button clicks — the one derived *input* the **last-label** step needs, so it is
      built only when `add_last` runs, and always rebuilt from raw (via
      clicks.run_trial_level_pipeline), reading from `button_clicks_fix_csv_path`
      + `button_clicks_fix_tsv_path`. `add_rts` does NOT need it — run-based RT has
      read the fixation report directly since 2026-10-06. For a self-contained report that holds both the
      message and fixation columns (e.g. the new experiment fixations), pass
      `button_clicks_fix_tsv_path=None` and set `button_clicks_msg_participant_col`
      (e.g. participant_id) and `all_answers_is_cumulative=False` to match it.
    - last-area labels (`add_last`) → `last_labels_path`.
    - RT/TFD features (`add_rts`) → `rt_and_tfd_path`.

    If `split_column` is provided (e.g. C.QUESTION_PREVIEW_COLUMN to recover the
    hunters/gatherers split), the processed DataFrame is *additionally*
    partitioned by the unique values of that column and each partition is saved
    as its own CSV. Use `split_output_paths` (a {value: path} mapping) to control
    the per-split filenames, e.g.::

        main(
            split_column=C.QUESTION_PREVIEW_COLUMN,
            split_output_paths={
                True: HUNTERS_PROCESSED_PATH,
                False: GATHERERS_PROCESSED_PATH,
            },
        )
    """
    # The canonical fixations report feeds both the pupil stats and the
    # fixation-sequence group feature. pupil_fixations_path can still override just
    # the pupil-stats source when set explicitly; otherwise it follows fixations_path.
    if pupil_fixations_path is None:
        pupil_fixations_path = fixations_path

    # Button clicks is the one derived input the RT/last-label steps need, so only
    # build it when those steps run. Always rebuilt from raw, routing to the
    # configured fixations source (default: the legacy CSV + separate TSV; pass
    # button_clicks_fix_tsv_path=None for a self-contained new-data report).
    #
    # This used to be reused whenever the file existed. It is the only place in the
    # prep pipeline that ever read its own previous output, and the staleness was
    # invisible: the table holds fixation ids and timestamps, which still *look*
    # right after an upstream change that invalidated them.
    # `add_last` only: run-based RT stopped reading this table on 2026-10-06. The
    # gate said `add_last or add_rts` until 2026-10-15, which was merely wasteful
    # while the table was cached and is a full raw rebuild now that it is not.
    needs_button_clicks = add_last
    if needs_button_clicks:
        if verbose:
            print("\nBuilding button-click data from raw…")
        run_trial_level_pipeline(
            fix_csv_path=button_clicks_fix_csv_path,
            fix_tsv_path=button_clicks_fix_tsv_path,
            output_csv_path=Path(button_clicks_path),
            msg_participant_col=button_clicks_msg_participant_col,
            all_answers_is_cumulative=all_answers_is_cumulative,
            verbose=verbose,
        )

    if verbose:
        print(f"\nLoading raw answers from: {ia_answers_path}")

    # The read-time column whitelist (stage D section 3.4; proved and switched on
    # 2026-10-13, step 2 of the grain proposal). Built here rather than in
    # `readers.py` so that the primitive stays free of the registry -- see
    # `load_raw_answers_data`.
    #
    # `filter_raw_columns=False` reads every column, which is how the proof is
    # re-run: one pass each way, compared byte for byte. Keep that escape hatch.
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

    # The fixations report has up to two consumers: the pupil-stats computation and
    # the fixation-sequence group feature. Load it once here and share the resulting
    # DataFrame with both, instead of letting each re-read the same large file.
    needs_pupil_fix = any(
        func is add_zscored_pupil_columns for func, _ in base_funcs
    )
    needs_seq_fix = any(
        func is create_fixation_sequence_tags for func, _ in group_funcs
    )

    fixations_df = None
    if needs_seq_fix or (needs_pupil_fix and pupil_fixations_path == fixations_path):
        if verbose:
            print(f"\nLoading fixations report once from: {fixations_path}")
        fixations_df = pd.read_csv(fixations_path)

    # Route the loaded fixations into the fixation-sequence group feature (it reads
    # raw fixation rows). Mirrors the pupil-stats injection for base funcs.
    if needs_seq_fix:
        group_funcs = [
            (func, {**kwargs, "fix_path": fixations_df})
            if func is create_fixation_sequence_tags
            else (func, kwargs)
            for func, kwargs in group_funcs
        ]

    # Resolve the ANSWER-SCREEN pupil baseline once and inject it into the pupil
    # z-scoring base feature (only if that feature is actually being run). Reuse the
    # shared fixations_df when the pupil source is the canonical report; otherwise
    # get_participant_pupil_stats reads the explicit pupil_fixations_path override.
    #
    # Always computed, never loaded (2026-10-15). The paragraph screen builds its
    # own baseline from its own fixations in `features/paragraph/spans.py` -- the
    # two screens differ in luminance and in task, so they are separate quantities
    # and are saved under separate names.
    if any(func is add_zscored_pupil_columns for func, _ in base_funcs):
        pupil_stats = get_participant_pupil_stats(
            fixations_path=pupil_fixations_path,
            fixations=fixations_df if pupil_fixations_path == fixations_path else None,
            verbose=verbose,
        )
        # Persist as an auxiliary artifact: a record of the baseline this run used.
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
        button_clicks_path=button_clicks_path,
        fixations_path=fixations_path,
        label="all_participants",
        verbose=verbose,
    )

    # The pre-split broadcast table: `all_participants.csv` and its group splits.
    #
    # **Off by default since 2026-10-13.** Every consumer now reads the grain
    # tables below, and on L1 these three files were 5.8 GB holding the same data
    # at ~39x redundancy -- 46% of their columns were constant within a trial.
    # Leaving this on would recreate that on the next prep run.
    #
    # The switch stays rather than the code going, because the frame itself does
    # not go away: `processed` is what the split is computed from, so persisting
    # it is a one-flag question rather than a rebuild.
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

    # The three grains (step 4d) -- the only thing written by default now.
    # `dataset=None` skips them, for a caller that wants the old output alone.
    if dataset is not None:
        if verbose:
            print("\nSplitting into grains…")
        save_grain_tables(
            processed, dataset, split_column=split_column, verbose=verbose
        )

    if verbose:
        print("\n✓ Done.\n")
