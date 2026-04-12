from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness.run_model_bundles import run_full_features_correctness_bundle, _split_tag
from predictive_modeling.common.feature_selection import correlation_prune_features, aic_forward_select_logit

import matplotlib.pyplot as plt

from viz.plot_output import _answer_correctness_rel_dir

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