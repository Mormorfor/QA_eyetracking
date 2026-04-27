from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional
from collections import Counter

import pandas as pd

import src.constants as Con
from predictive_modeling.answer_correctness import feature_groups as fg
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


# ==========================================================================
# Feature-set generators
# ==========================================================================
#
# Each generator below builds and saves one logical group of feature-column
# JSONs. The generators are independent so any group can be rerun on its
# own without recomputing the others. The orchestrator
# `generate_all_feature_column_sets` runs every group.
#
# Naming scheme:
#   baseline_last_*       group 1: each LAST_* alone
#   baseline_last_all     group 2: LAST_ANSWER + LAST_CONFIRM + LAST_SELECT
#   all_no_last           group 3: ALL_FEATURES_NO_LAST
#   all_last_<x>          group 4: ALL_FEATURES_NO_LAST + LAST_<X>
#   all_last_all          group 5: full ALL_FEATURES
#   <base>_pruned_t<thr>  groups 6, 8: correlation-pruned at threshold
#   <base>_aic            groups 7, 8: AIC forward-selected
#   derived_with_num_selects  group 9
#   question_only         group 10: PER_QUESTION_COLS
#   area_only             group 11: AREA_COLS
#   select_1[_last_*]     group 12: SELECT_1_COLS curated subset (5 variants)

DEFAULT_CORR_THRESHOLDS: Tuple[float, ...] = (0.5, 0.7, 0.9)


def _save_set_collection(
    sets: Dict[str, Sequence[str]],
    folder_path: str,
    verbose: bool,
) -> Dict[str, Path]:
    """Save a {identifier: cols} mapping and return {identifier: path}."""
    saved_paths: Dict[str, Path] = {}
    for identifier, cols in sets.items():
        _save_one(
            columns=cols,
            identifier=identifier,
            folder_path=folder_path,
            saved_paths=saved_paths,
        )
        if verbose:
            print(f"Saved: {identifier} ({len(cols)} cols)")
    return saved_paths


def _generate_pruned_for_base(
    trial_df: pd.DataFrame,
    base_name: str,
    base_cols: Sequence[str],
    folder_path: str,
    target_col: str,
    corr_thresholds: Sequence[float],
    verbose: bool,
) -> Dict[str, Path]:
    """Correlation-prune `base_cols` at each threshold and save."""
    saved_paths: Dict[str, Path] = {}
    base_cols = _dedupe_keep_order(base_cols)
    for threshold in corr_thresholds:
        thr_tag = _threshold_to_tag(threshold)
        identifier = f"{base_name}_pruned_t{thr_tag}"
        kept_cols, _, _ = correlation_prune_features(
            df=trial_df,
            feature_cols=base_cols,
            target_col=target_col,
            corr_threshold=threshold,
            verbose=False,
        )
        _save_one(kept_cols, identifier, folder_path, saved_paths)
        if verbose:
            print(
                f"Saved: {identifier} "
                f"({len(kept_cols)} kept / {len(base_cols)} input)"
            )
    return saved_paths


def _generate_aic_for_base(
    trial_df: pd.DataFrame,
    base_name: str,
    base_cols: Sequence[str],
    folder_path: str,
    target_col: str,
    standardize: bool,
    verbose: bool,
) -> Dict[str, Path]:
    """AIC forward-select on `base_cols` and save."""
    saved_paths: Dict[str, Path] = {}
    base_cols = _dedupe_keep_order(base_cols)
    identifier = f"{base_name}_aic"
    aic_cols, _, _ = aic_forward_select_logit(
        df=trial_df,
        feature_cols=base_cols,
        target_col=target_col,
        standardize=standardize,
        verbose=False,
    )
    _save_one(aic_cols, identifier, folder_path, saved_paths)
    if verbose:
        print(
            f"Saved: {identifier} "
            f"({len(aic_cols)} after AIC / {len(base_cols)} input)"
        )
    return saved_paths


def _all_with_last(last_cols: Sequence[str]) -> List[str]:
    return _dedupe_keep_order(list(fg.ALL_FEATURES_NO_LAST) + list(last_cols))


def _base_with_last_variants(
    base_cols: Sequence[str],
    base_identifier: str,
) -> Dict[str, List[str]]:
    """
    {identifier: cols} for `base_cols` itself + each LAST_* variant.

    Use this when adding a new curated feature set that should be saved
    in five flavours (no_last / last_ans / last_confirm / last_select / last_all).
    """
    base = list(base_cols)
    return {
        base_identifier: _dedupe_keep_order(base),
        f"{base_identifier}_last_ans": _dedupe_keep_order(base + list(fg.LAST_ANSWER)),
        f"{base_identifier}_last_confirm": _dedupe_keep_order(base + list(fg.LAST_CONFIRM)),
        f"{base_identifier}_last_select": _dedupe_keep_order(base + list(fg.LAST_SELECT)),
        f"{base_identifier}_last_all": _dedupe_keep_order(base + list(fg.LAST_ALL)),
    }


def _last_with_last_groups() -> Dict[str, List[str]]:
    """{base_identifier: last_cols} for the four with-last variants."""
    return {
        "all_last_ans": list(fg.LAST_ANSWER),
        "all_last_confirm": list(fg.LAST_CONFIRM),
        "all_last_select": list(fg.LAST_SELECT),
        "all_last_all": list(fg.LAST_ALL),
    }


# --------------------------------------------------------------------------
# Group 1: each LAST_* alone
# --------------------------------------------------------------------------
def generate_baseline_last_only_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {
        "baseline_last_ans": fg.LAST_ANSWER,
        "baseline_last_confirm": fg.LAST_CONFIRM,
        "baseline_last_select": fg.LAST_SELECT,
    }
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 2: all three LAST_* together
# --------------------------------------------------------------------------
def generate_baseline_last_all_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"baseline_last_all": fg.LAST_ALL}
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 3: all features minus LAST_*
# --------------------------------------------------------------------------
def generate_all_no_last_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"all_no_last": fg.ALL_FEATURES_NO_LAST}
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 4: all features + each LAST_* separately
# --------------------------------------------------------------------------
def generate_all_with_each_last_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {
        "all_last_ans": _all_with_last(fg.LAST_ANSWER),
        "all_last_confirm": _all_with_last(fg.LAST_CONFIRM),
        "all_last_select": _all_with_last(fg.LAST_SELECT),
    }
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 5: all features + every LAST_*
# --------------------------------------------------------------------------
def generate_all_with_last_all_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"all_last_all": fg.ALL_FEATURES}
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 6: all features no last - correlation pruned at multiple thresholds
# --------------------------------------------------------------------------
def generate_pruned_no_last_sets(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    corr_thresholds: Sequence[float] = DEFAULT_CORR_THRESHOLDS,
    verbose: bool = True,
) -> Dict[str, Path]:
    return _generate_pruned_for_base(
        trial_df=trial_df,
        base_name="all_no_last",
        base_cols=fg.ALL_FEATURES_NO_LAST,
        folder_path=folder_path,
        target_col=target_col,
        corr_thresholds=corr_thresholds,
        verbose=verbose,
    )


# --------------------------------------------------------------------------
# Group 7: all features no last - AIC forward-selected
# --------------------------------------------------------------------------
def generate_aic_no_last_set(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    standardize: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    return _generate_aic_for_base(
        trial_df=trial_df,
        base_name="all_no_last",
        base_cols=fg.ALL_FEATURES_NO_LAST,
        folder_path=folder_path,
        target_col=target_col,
        standardize=standardize,
        verbose=verbose,
    )


# --------------------------------------------------------------------------
# Group 8: groups 6 and 7 across the four LAST_* inclusion versions
# --------------------------------------------------------------------------
def generate_pruned_with_last_sets(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    corr_thresholds: Sequence[float] = DEFAULT_CORR_THRESHOLDS,
    verbose: bool = True,
) -> Dict[str, Path]:
    saved_paths: Dict[str, Path] = {}
    for base_name, last_cols in _last_with_last_groups().items():
        saved_paths.update(
            _generate_pruned_for_base(
                trial_df=trial_df,
                base_name=base_name,
                base_cols=_all_with_last(last_cols),
                folder_path=folder_path,
                target_col=target_col,
                corr_thresholds=corr_thresholds,
                verbose=verbose,
            )
        )
    return saved_paths


def generate_aic_with_last_sets(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    standardize: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    saved_paths: Dict[str, Path] = {}
    for base_name, last_cols in _last_with_last_groups().items():
        saved_paths.update(
            _generate_aic_for_base(
                trial_df=trial_df,
                base_name=base_name,
                base_cols=_all_with_last(last_cols),
                folder_path=folder_path,
                target_col=target_col,
                standardize=standardize,
                verbose=verbose,
            )
        )
    return saved_paths


# --------------------------------------------------------------------------
# Group 9: derived columns + NUM_OF_SELECTS
# --------------------------------------------------------------------------
def generate_derived_with_num_selects_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {
        "derived_with_num_selects": fg.DERIVED_COLS + [Con.NUM_OF_SELECTS],
    }
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 10: question features only (per-metric '__question' columns)
# --------------------------------------------------------------------------
def generate_question_only_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"question_only": fg.PER_QUESTION_COLS}
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 11: AREA_COLS only (correct/wrong_mean/contrast/distance_*)
# --------------------------------------------------------------------------
def generate_area_only_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"area_only": fg.AREA_COLS}
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Group 12: SELECT_1 curated subset + each LAST_* variant
# --------------------------------------------------------------------------
def generate_select_1_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = _base_with_last_variants(
        fg.SELECT_1_COLS, "select_1"
    )
    return _save_set_collection(sets, folder_path, verbose)


# --------------------------------------------------------------------------
# Orchestrator: run every group
# --------------------------------------------------------------------------
def generate_all_feature_column_sets(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    corr_thresholds: Sequence[float] = DEFAULT_CORR_THRESHOLDS,
    standardize_aic: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Run every feature-set generator (groups 1-11). Returns a single
    {identifier: saved_path} dict. Individual generators can be called
    directly for partial reruns.
    """
    saved_paths: Dict[str, Path] = {}

    # 1, 2: pure last baselines
    saved_paths.update(generate_baseline_last_only_sets(folder_path, verbose=verbose))
    saved_paths.update(generate_baseline_last_all_set(folder_path, verbose=verbose))

    # 3, 4, 5: all features + last variations
    saved_paths.update(generate_all_no_last_set(folder_path, verbose=verbose))
    saved_paths.update(generate_all_with_each_last_sets(folder_path, verbose=verbose))
    saved_paths.update(generate_all_with_last_all_set(folder_path, verbose=verbose))

    # 6, 7: pruning + AIC on no-last
    saved_paths.update(
        generate_pruned_no_last_sets(
            trial_df, folder_path,
            target_col=target_col,
            corr_thresholds=corr_thresholds,
            verbose=verbose,
        )
    )
    saved_paths.update(
        generate_aic_no_last_set(
            trial_df, folder_path,
            target_col=target_col,
            standardize=standardize_aic,
            verbose=verbose,
        )
    )

    # 8: pruning + AIC across the four with-last bases
    saved_paths.update(
        generate_pruned_with_last_sets(
            trial_df, folder_path,
            target_col=target_col,
            corr_thresholds=corr_thresholds,
            verbose=verbose,
        )
    )
    saved_paths.update(
        generate_aic_with_last_sets(
            trial_df, folder_path,
            target_col=target_col,
            standardize=standardize_aic,
            verbose=verbose,
        )
    )

    # 9, 10, 11: focused subsets
    saved_paths.update(generate_derived_with_num_selects_set(folder_path, verbose=verbose))
    saved_paths.update(generate_question_only_set(folder_path, verbose=verbose))
    saved_paths.update(generate_area_only_set(folder_path, verbose=verbose))

    # 12: SELECT_1 curated subset + each LAST_* variant
    saved_paths.update(generate_select_1_sets(folder_path, verbose=verbose))

    return saved_paths



def _dir_has_any_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())



def _correctness_output_exists(
    test_regimes: Sequence[str],
    subdir: Optional[str] = None,
    results_base_dir: str | Path = "../reports/plots",
    verbose: bool = False,
) -> bool:
    base_rel_dir = Path(
        _answer_correctness_rel_dir(
            model_family="logreg",
            subdir=subdir,
            split_tag=_split_tag(test_regimes),
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
    test_regimes: Optional[Sequence[str]] = None,
    *,
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),
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
    if test_regimes is None:
        test_regimes = ["new_subject"]

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
                test_regimes=test_regimes,
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
            test_regimes=test_regimes,
            test_split=test_split,
            fold=fold,
            sources=sources,
            feature_cols=feature_cols,
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