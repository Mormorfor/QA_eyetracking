from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional
from collections import Counter

import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness.run_model_bundles import run_full_features_correctness_bundle, _split_tag
from predictive_modeling.common.feature_selection import correlation_prune_features, aic_forward_select_logit

import matplotlib.pyplot as plt

from viz.plot_output import _answer_correctness_rel_dir, save_plot

COL_SAVE_PATH = "../reports/report_data/answer_correctness/feature_columns"


def save_feature_columns(
    columns: List[str],
    identifier: str,
    folder_path: str,
) -> Path:
    """
    Save feature columns with an identifier.

    - File name = {identifier}.json
    - File content includes both identifier and columns
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    filepath = folder / f"{identifier}.json"

    payload: Dict[str, Any] = {
        "identifier": identifier,
        "columns": columns,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return filepath


def load_feature_columns_from_json(path: str | Path) -> Dict[str, Any]:
    """
    Load one saved feature-column JSON file.

    Expected structure:
    {
      "identifier": "...",
      "columns": [...]
    }
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload


def _dedupe_keep_order(cols: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _threshold_to_tag(threshold: float) -> str:
    """
    0.5 -> '05'
    0.6 -> '06'
    0.75 -> '075'
    0.8 -> '08'
    """
    s = str(threshold).replace(".", "")
    return s


def _save_one(
    columns: Sequence[str],
    identifier: str,
    folder_path: str,
    saved_paths: Dict[str, Path],
) -> Path:
    cols = _dedupe_keep_order(columns)
    path = save_feature_columns(
        columns=cols,
        identifier=identifier,
        folder_path=folder_path,
    )
    saved_paths[identifier] = path
    return path


def generate_feature_column_files(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    target_col: str = Con.IS_CORRECT_COLUMN,
    corr_thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
    standardize_aic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate and save a full collection of feature-column JSON files.

    Saves:

    Baselines
    ---------
    1) baseline_last_ans
    2) baseline_last_confirm
    3) baseline_last_select

    No-optimization sets
    --------------------
    4) full_no_last
    5) full_last_all
    6) selected_no_last
    7) selected_last_all

    Correlation-pruned sets
    -----------------------
    8) full_no_last_pruned_t{thr}
    9) full_last_ans_pruned_t{thr}
       full_last_confirm_pruned_t{thr}
       full_last_select_pruned_t{thr}

    AIC-after-pruning sets
    ----------------------
    10) ..._aic versions for every pruned set above

    Returns a dictionary with:
    - saved_paths
    - prune_logs
    - aic_logs
    - aic_models
    """

    # ------------------------------------------------------------------
    # Last-based feature groups
    # ------------------------------------------------------------------
    LAST_ANS_VISITED_COLUMNS = [
        "last_visited_answer_A",
        "last_visited_answer_B",
        "last_visited_answer_C",
        "last_visited_answer_D",
    ]

    LAST_CONF = [
        "last_before_confirm_answer_A",
        "last_before_confirm_answer_B",
        "last_before_confirm_answer_C",
        "last_before_confirm_answer_D",
        "last_before_confirm_question",
    ]

    LAST_SELECT = [
        "last_before_select_answer_A",
        "last_before_select_answer_B",
        "last_before_select_answer_C",
        "last_before_select_answer_D",
        "last_before_select_question",
    ]

    LAST_ALL = _dedupe_keep_order(
        LAST_ANS_VISITED_COLUMNS + LAST_CONF + LAST_SELECT
    )

    # ------------------------------------------------------------------
    # Full feature set
    # ------------------------------------------------------------------
    METRIC_COLUMNS = [
        Con.MEAN_DWELL_TIME,
        Con.MEAN_FIXATIONS_COUNT,
        Con.MEAN_FIRST_FIXATION_DURATION,
        Con.SKIP_RATE,
        Con.AREA_DWELL_PROPORTION,
        Con.MEAN_AVG_FIX_PUPIL_SIZE_Z,
        Con.MEAN_MAX_FIX_PUPIL_SIZE_Z,
        Con.MEAN_MIN_FIX_PUPIL_SIZE_Z,
        Con.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z,
        Con.NUM_LABEL_VISITS,
    ]

    DERIVED_COLS = [
        "seq_len",
        "has_xyx",
        "has_xyxy",
        "trial_mean_dwell",
    ]

    AREA_COLS = (
        [f"{m}__correct" for m in METRIC_COLUMNS]
        + [f"{m}__wrong_mean" for m in METRIC_COLUMNS]
        + [f"{m}__contrast" for m in METRIC_COLUMNS]
        + [f"{m}__distance_furthest" for m in METRIC_COLUMNS]
        + [f"{m}__distance_closest" for m in METRIC_COLUMNS]
    )

    FULL_NO_LAST = _dedupe_keep_order(AREA_COLS + DERIVED_COLS)
    FULL_LAST_ALL = _dedupe_keep_order(FULL_NO_LAST + LAST_ALL)

    # ------------------------------------------------------------------
    # Manually selected set
    # ------------------------------------------------------------------
    SELECTED_METRIC_COLUMNS = [
        Con.SKIP_RATE,
        Con.AREA_DWELL_PROPORTION,
        Con.NUM_LABEL_VISITS,
    ]

    SELECTED_COLS = (
        [f"{m}__correct" for m in SELECTED_METRIC_COLUMNS]
        + [f"{m}__wrong_mean" for m in SELECTED_METRIC_COLUMNS]
        + [
            "seq_len",
            "has_xyx",
        ]
    )

    SELECTED_NO_LAST = _dedupe_keep_order(SELECTED_COLS)
    SELECTED_LAST_ALL = _dedupe_keep_order(SELECTED_NO_LAST + LAST_ALL)

    saved_paths: Dict[str, Path] = {}
    prune_logs: Dict[str, Any] = {}
    aic_logs: Dict[str, Any] = {}
    aic_models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1-7: raw sets
    # ------------------------------------------------------------------
    raw_sets: Dict[str, List[str]] = {
        "baseline_last_ans": LAST_ANS_VISITED_COLUMNS,
        "baseline_last_confirm": LAST_CONF,
        "baseline_last_select": LAST_SELECT,
        "full_no_last": FULL_NO_LAST,
        "full_last_all": FULL_LAST_ALL,
        "selected_no_last": SELECTED_NO_LAST,
        "selected_last_all": SELECTED_LAST_ALL,
    }

    for identifier, cols in raw_sets.items():
        _save_one(
            columns=cols,
            identifier=identifier,
            folder_path=folder_path,
            saved_paths=saved_paths,
        )
        if verbose:
            print(f"Saved: {identifier} ({len(cols)} cols)")

    # ------------------------------------------------------------------
    # 8-10: pruned sets and AIC versions
    # ------------------------------------------------------------------
    prune_bases: Dict[str, List[str]] = {
        "full_no_last": FULL_NO_LAST,
        "full_last_ans": _dedupe_keep_order(FULL_NO_LAST + LAST_ANS_VISITED_COLUMNS),
        "full_last_confirm": _dedupe_keep_order(FULL_NO_LAST + LAST_CONF),
        "full_last_select": _dedupe_keep_order(FULL_NO_LAST + LAST_SELECT),
    }

    for base_name, candidate_cols in prune_bases.items():
        for threshold in corr_thresholds:
            thr_tag = _threshold_to_tag(threshold)

            pruned_identifier = f"{base_name}_pruned_t{thr_tag}"

            kept_cols, dropped_cols, prune_log = correlation_prune_features(
                df=trial_df,
                feature_cols=candidate_cols,
                target_col=target_col,
                corr_threshold=threshold,
                verbose=False,
            )

            _save_one(
                columns=kept_cols,
                identifier=pruned_identifier,
                folder_path=folder_path,
                saved_paths=saved_paths,
            )

            prune_logs[pruned_identifier] = {
                "threshold": threshold,
                "base_name": base_name,
                "n_input": len(candidate_cols),
                "n_kept": len(kept_cols),
                "n_dropped": len(dropped_cols),
                "dropped_cols": dropped_cols,
                "prune_log": prune_log,
            }

            if verbose:
                print(
                    f"Saved: {pruned_identifier} "
                    f"({len(kept_cols)} kept / {len(candidate_cols)} input)"
                )

            aic_identifier = f"{pruned_identifier}_aic"

            aic_cols, aic_log, aic_model = aic_forward_select_logit(
                df=trial_df,
                feature_cols=kept_cols,
                target_col=target_col,
                standardize=standardize_aic,
                verbose=False,
            )

            _save_one(
                columns=aic_cols,
                identifier=aic_identifier,
                folder_path=folder_path,
                saved_paths=saved_paths,
            )

            aic_logs[aic_identifier] = {
                "threshold": threshold,
                "base_name": base_name,
                "n_input_to_aic": len(kept_cols),
                "n_selected_by_aic": len(aic_cols),
                "aic_log": aic_log,
            }
            aic_models[aic_identifier] = aic_model

            if verbose:
                print(
                    f"Saved: {aic_identifier} "
                    f"({len(aic_cols)} cols after AIC)"
                )

    return {
        "saved_paths": saved_paths,
        "prune_logs": prune_logs,
        "aic_logs": aic_logs,
        "aic_models": aic_models,
    }



def selected_with_last_variations():
    LAST_ANS = [
        "last_visited_answer_A",
        "last_visited_answer_B",
        "last_visited_answer_C",
        "last_visited_answer_D",
    ]

    LAST_CONF = [
        "last_before_confirm_answer_A",
        "last_before_confirm_answer_B",
        "last_before_confirm_answer_C",
        "last_before_confirm_answer_D",
        "last_before_confirm_question",
    ]

    LAST_SELECT = [
        "last_before_select_answer_A",
        "last_before_select_answer_B",
        "last_before_select_answer_C",
        "last_before_select_answer_D",
        "last_before_select_question",
    ]

    SELECTED_NO_LAST = [
        "skip_rate__correct",
        "skip_rate__wrong_mean",
        "area_dwell_proportion__correct",
        "area_dwell_proportion__wrong_mean",
        "num_label_visits__correct",
        "num_label_visits__wrong_mean",
        "seq_len",
        "has_xyx",
    ]

    # --------------------------------------
    # Save each combination
    # --------------------------------------
    _save_one(
        columns=SELECTED_NO_LAST + LAST_ANS,
        identifier="selected_last_ans_only",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    _save_one(
        columns=SELECTED_NO_LAST + LAST_CONF,
        identifier="selected_last_confirm_only",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    _save_one(
        columns=SELECTED_NO_LAST + LAST_SELECT,
        identifier="selected_last_select_only",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )


def second_manual_selected():
    SECOND_SELECTED = [
        "area_dwell_proportion__correct",
        "area_dwell_proportion__distance_closest",
        "num_label_visits__correct",
        "num_label_visits__distance_closest",
    ]

    LAST_ANS = [
        "last_visited_answer_A",
        "last_visited_answer_B",
        "last_visited_answer_C",
        "last_visited_answer_D",
    ]

    LAST_CONF = [
        "last_before_confirm_answer_A",
        "last_before_confirm_answer_B",
        "last_before_confirm_answer_C",
        "last_before_confirm_answer_D",
        "last_before_confirm_question",
    ]

    LAST_SELECT = [
        "last_before_select_answer_A",
        "last_before_select_answer_B",
        "last_before_select_answer_C",
        "last_before_select_answer_D",
        "last_before_select_question",
    ]

    LAST_ALL = LAST_ANS + LAST_CONF + LAST_SELECT

    # --------------------------------------
    # Save variants
    # --------------------------------------

    # 1) no last
    _save_one(
        columns=SECOND_SELECTED,
        identifier="second_selected_no_last",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    # 2) all last together
    _save_one(
        columns=SECOND_SELECTED + LAST_ALL,
        identifier="second_selected_last_all",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    # 3) last answer only
    _save_one(
        columns=SECOND_SELECTED + LAST_ANS,
        identifier="second_selected_last_ans",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    # 4) last confirm only
    _save_one(
        columns=SECOND_SELECTED + LAST_CONF,
        identifier="second_selected_last_confirm",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )

    # 5) last select only
    _save_one(
        columns=SECOND_SELECTED + LAST_SELECT,
        identifier="second_selected_last_select",
        folder_path=COL_SAVE_PATH,
        saved_paths={},
    )


def sequence_only():
    SEQ_ONLY = ["seq_len", "has_xyx", "has_xyxy", "trial_mean_dwell"]

    LAST_ANS = [
        "last_visited_answer_A",
        "last_visited_answer_B",
        "last_visited_answer_C",
        "last_visited_answer_D",
    ]

    _save_one(SEQ_ONLY, "sequence_only", COL_SAVE_PATH, {})
    _save_one(SEQ_ONLY + LAST_ANS, "sequence_last_ans", COL_SAVE_PATH, {})



def _dir_has_any_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())



def _correctness_output_exists(
    split_group_cols: Sequence[str],
    subdir: Optional[str] = None,
    results_base_dir: str | Path = "../reports/plots",
    verbose: bool = False,
) -> bool:
    base_rel_dir = Path(
        _answer_correctness_rel_dir(
            model_family="logreg",
            subdir=subdir,
            split_tag=_split_tag(split_group_cols),
        )
    )

    search_dir = Path(results_base_dir) / base_rel_dir

    if verbose:
        print("\n[CHECK] Looking for existing results in:")
        print(f"  - {search_dir} (exists={search_dir.exists()})")

    has_files = search_dir.exists() and any(p.is_file() for p in search_dir.rglob("*"))

    if verbose:
        print(f"    -> has_files={has_files}")

    return has_files


def run_correctness_bundle_for_saved_column_sets(
    df,
    columns_folder: str | Path,
    split_group_cols=None,
    paper_dirs=None,
    save: bool = False,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    recursive: bool = False,
    rerun: bool = True,
    results_base_dir: str | Path = "../reports/plots",
    verbose: bool = True,
):
    """
    Run `run_full_features_correctness_bundle` once for each saved
    feature-column JSON file in `columns_folder`.

    For each file:
    - feature_cols = loaded columns
    - subdir = identifier
    - run_identifier = identifier

    Parameters
    ----------
    rerun : bool
        If False, skip runs whose output directory already exists and is non-empty.

    Returns
    -------
    results_by_identifier : dict
        identifier -> model output
    metadata_df : pd.DataFrame
        simple run summary
    """
    if split_group_cols is None:
        split_group_cols = [Con.PARTICIPANT_ID]

    columns_folder = Path(columns_folder)

    if recursive:
        json_paths = sorted(columns_folder.rglob("*.json"))
    else:
        json_paths = sorted(columns_folder.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {columns_folder}")

    results_by_identifier = {}
    run_rows = []

    for json_path in json_paths:
        plt.close("all")
        payload = load_feature_columns_from_json(json_path)

        identifier = payload["identifier"]
        feature_cols = payload["columns"]

        already_exists = False
        if not rerun:
            already_exists = _correctness_output_exists(
                split_group_cols=split_group_cols,
                subdir=identifier,
                results_base_dir=results_base_dir,
                verbose=verbose,
            )

        if already_exists:
            if verbose:
                print(f"\nSkipping: {identifier}")
                print(f"  source: {json_path}")
                print("  reason: output already exists and rerun=False")

            run_rows.append({
                "identifier": identifier,
                "json_path": str(json_path),
                "n_features": len(feature_cols),
                "status": "skipped_existing",
            })
            continue

        if verbose:
            print(f"\nRunning: {identifier}")
            print(f"  source: {json_path}")
            print(f"  n_features: {len(feature_cols)}")

        out = run_full_features_correctness_bundle(
            df=df,
            feature_cols=feature_cols,
            split_group_cols=split_group_cols,
            paper_dirs=paper_dirs,
            subdir=identifier,
            run_identifier=identifier,
            save=save,
            coef_ci_method=coef_ci_method,
            coef_ci_cluster=coef_ci_cluster,
        )

        results_by_identifier[identifier] = out

        run_rows.append({
            "identifier": identifier,
            "json_path": str(json_path),
            "n_features": len(feature_cols),
            "status": "ran",
        })

    metadata_df = pd.DataFrame(run_rows).sort_values("identifier").reset_index(drop=True)
    return results_by_identifier, metadata_df


def plot_feature_frequency_from_full_jsons(
    columns_folder: str | Path,
    recursive: bool = False,
    case_sensitive: bool = False,
    sort_desc: bool = True,
    top_n: Optional[int] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    save: bool = False,
    save_path: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = False,
    verbose: bool = True,
):
    """
    Load all feature-column JSON files from `columns_folder` whose FILE NAME
    contains 'full', count how often each feature appears across files, and plot it.

    """
    columns_folder = Path(columns_folder)

    if recursive:
        json_paths = sorted(columns_folder.rglob("*.json"))
    else:
        json_paths = sorted(columns_folder.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {columns_folder}")

    def matches_full(path: Path) -> bool:
        name = path.name if case_sensitive else path.name.lower()
        needle = "full" if case_sensitive else "full"
        return needle in name

    matched_files = [p for p in json_paths if matches_full(p)]

    if not matched_files:
        raise FileNotFoundError(
            f"No JSON files with 'full' in the filename found in: {columns_folder}"
        )

    if verbose:
        print(f"Found {len(matched_files)} matching JSON files:")
        for p in matched_files:
            print(f"  - {p}")

    feature_counter = Counter()

    for json_path in matched_files:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "columns" not in payload:
            raise KeyError(f"Missing 'columns' key in file: {json_path}")

        feature_cols = payload["columns"]

        if not isinstance(feature_cols, list):
            raise TypeError(f"'columns' must be a list in file: {json_path}")

        feature_counter.update(feature_cols)

    if not feature_counter:
        raise ValueError("No features found in matched files.")

    freq_df = pd.DataFrame(
        [{"feature": feat, "count": cnt} for feat, cnt in feature_counter.items()]
    )

    freq_df = freq_df.sort_values(
        by=["count", "feature"],
        ascending=[not sort_desc, True]
    ).reset_index(drop=True)

    if top_n is not None:
        freq_df_plot = freq_df.head(top_n).copy()
    else:
        freq_df_plot = freq_df.copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(freq_df_plot["feature"], freq_df_plot["count"])
    ax.invert_yaxis()

    ax.set_xlabel("Frequency across matching files")
    ax.set_ylabel("Feature")
    ax.set_title(
        title or f"Feature appearance frequency across JSON files with 'full' in filename"
    )

    plt.tight_layout()

    if save:
        if save_path is None:
            save_path = columns_folder / "feature_frequency_full_files.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if verbose:
            print(f"Saved plot to: {save_path}")

    if close:
        plt.close(fig)

    return fig, freq_df, matched_files



def generate_k_most_frequent_feature_sets_from_full_files(
    columns_folder: str | Path,
    folder_path: str | Path,
    k: int,
    recursive: bool = False,
    case_sensitive: bool = False,
    save_plot_flag: bool = True,
    plot_rel_dir: str = "feature_selection",
    plot_filename: str = "k_most_frequent_feature_frequency",
    paper_dirs=None,
    verbose: bool = True,
):
    """
    Builds k-most-frequent feature sets from JSON files containing 'full'
    and saves:

    - k_most_frequent
    - k_most_frequent_last_ans
    - k_most_frequent_last_confirm
    - k_most_frequent_last_select
    - k_most_frequent_last_all

    Also plots feature frequencies.

    Returns
    -------
    saved_paths : dict
    freq_df : pd.DataFrame
    """

    columns_folder = Path(columns_folder)

    json_paths = (
        sorted(columns_folder.rglob("*.json"))
        if recursive
        else sorted(columns_folder.glob("*.json"))
    )

    def _matches_full(p: Path):
        name = p.name if case_sensitive else p.name.lower()
        return "full" in name

    matched_files = [p for p in json_paths if _matches_full(p)]

    if not matched_files:
        raise FileNotFoundError("No 'full' JSON files found")

    if verbose:
        print(f"Using {len(matched_files)} files with 'full'")

    # ------------------------------------------------------------------
    # Last groups (same as your generator)
    # ------------------------------------------------------------------
    LAST_ANS = [
        "last_visited_answer_A",
        "last_visited_answer_B",
        "last_visited_answer_C",
        "last_visited_answer_D",
    ]

    LAST_CONF = [
        "last_before_confirm_answer_A",
        "last_before_confirm_answer_B",
        "last_before_confirm_answer_C",
        "last_before_confirm_answer_D",
        "last_before_confirm_question",
    ]

    LAST_SELECT = [
        "last_before_select_answer_A",
        "last_before_select_answer_B",
        "last_before_select_answer_C",
        "last_before_select_answer_D",
        "last_before_select_question",
    ]

    LAST_ALL = _dedupe_keep_order(LAST_ANS + LAST_CONF + LAST_SELECT)

    # ------------------------------------------------------------------
    # Count frequencies
    # ------------------------------------------------------------------
    counter = Counter()

    for p in matched_files:
        payload = load_feature_columns_from_json(p)
        counter.update(payload["columns"])

    if not counter:
        raise ValueError("No features found")

    sorted_feats = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    k_base = _dedupe_keep_order([f for f, _ in sorted_feats[:k]])

    if verbose:
        print(f"Selected top {len(k_base)} features")

    # ------------------------------------------------------------------
    # Build sets
    # ------------------------------------------------------------------
    feature_sets = {
        f"{k}_most_frequent": k_base,
        f"{k}_most_frequent_last_ans": _dedupe_keep_order(k_base + LAST_ANS),
        f"{k}_most_frequent_last_confirm": _dedupe_keep_order(k_base + LAST_CONF),
        f"{k}_most_frequent_last_select": _dedupe_keep_order(k_base + LAST_SELECT),
        f"{k}_most_frequent_last_all": _dedupe_keep_order(k_base + LAST_ALL),
    }

    saved_paths: Dict[str, Path] = {}

    for identifier, cols in feature_sets.items():
        _save_one(
            columns=cols,
            identifier=identifier,
            folder_path=folder_path,
            saved_paths=saved_paths,
        )
        if verbose:
            print(f"Saved: {identifier} ({len(cols)} cols)")


    freq_df = pd.DataFrame(sorted_feats, columns=["feature", "count"])

    freq_df_k = freq_df[freq_df["feature"].isin(k_base)].copy()
    freq_df_k = freq_df_k.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(freq_df_k["feature"], freq_df_k["count"])

    ax.set_title(f"Top-{k} most frequent features")
    ax.set_xlabel("Frequency across 'full' feature sets")

    plt.tight_layout()

    if save_plot_flag:
        save_plot(
            fig=fig,
            rel_dir=plot_rel_dir,
            filename=plot_filename,
            paper_dirs=paper_dirs,
            close=True,
        )
    else:
        plt.show()

    return saved_paths, freq_df_k