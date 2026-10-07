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
    PARTICIPANT_PUPILS_PATH,
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
    src.features.reading_times.build_rt_and_tfd), reading the trial's fixation
    sequence from `button_clicks_path` for the run-based RT. Answer regions only
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
    compute_pupil_stats: bool = True,
    pupil_fixations_path: Path = None,
    pupil_stats_path: Path = PARTICIPANT_PUPILS_PATH,
    rebuild_button_clicks: bool = False,
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

    Intermediate ("auxiliary") artifacts are all derived on the fly and, when
    `save_auxiliary=True`, persisted under the dataset's `interim/aux/`:
    - participant pupil stats — computed from raw fixation data at
      `pupil_fixations_path` (which defaults to `fixations_path`) when
      `compute_pupil_stats=True`, else loaded from `pupil_stats_path`. Saved to
      `pupil_stats_path` when freshly computed.
    - button clicks — the one derived *input* the RT/last-label steps need, so it is
      only built when `add_last`/`add_rts` run. Rebuilt from raw (via
      clicks.run_trial_level_pipeline) when `rebuild_button_clicks=True`
      or when `button_clicks_path` is missing, reading from `button_clicks_fix_csv_path`
      + `button_clicks_fix_tsv_path`. For a self-contained report that holds both the
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
    # build it when those steps run. Rebuild from raw when forced or absent, routing
    # to the configured fixations source (default: the legacy CSV + separate TSV;
    # pass button_clicks_fix_tsv_path=None for a self-contained new-data report).
    needs_button_clicks = add_last or add_rts
    if needs_button_clicks and (
        rebuild_button_clicks or not Path(button_clicks_path).exists()
    ):
        if verbose:
            reason = "forced" if rebuild_button_clicks else "missing"
            print(f"\nBuilding button-click data from raw ({reason})…")
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

    df_answers = load_raw_answers_data(ia_answers_path)

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
    needs_pupil_fix = compute_pupil_stats and any(
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

    # Resolve participant pupil stats once and inject them into the pupil
    # z-scoring base feature (only if that feature is actually being run). Reuse the
    # shared fixations_df when the pupil source is the canonical report; otherwise
    # get_participant_pupil_stats reads the explicit pupil_fixations_path override.
    if any(func is add_zscored_pupil_columns for func, _ in base_funcs):
        pupil_stats = get_participant_pupil_stats(
            stats_csv_path=pupil_stats_path,
            fixations_path=pupil_fixations_path,
            fixations=fixations_df if pupil_fixations_path == fixations_path else None,
            compute=compute_pupil_stats,
            verbose=verbose,
        )
        # Persist freshly-computed stats as an auxiliary artifact (no need to
        # re-save when they were just loaded from pupil_stats_path).
        if save_auxiliary and compute_pupil_stats:
            _save(pupil_stats, pupil_stats_path, label="participant pupils", verbose=verbose)
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

    _save(processed, output_path, label="all_participants", verbose=verbose)

    if split_column is not None:
        if verbose:
            print(f"\nSaving splits by column: {split_column}…")
        _save_splits(
            processed,
            split_column=split_column,
            base_output_path=output_path,
            split_output_paths=split_output_paths,
            verbose=verbose,
        )

    if verbose:
        print("\n✓ Done.\n")
