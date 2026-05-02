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
from src.predictive_modeling.answer_correctness.feature_groups import (
    LAST_ALL,
    LAST_ANSWER,
    LAST_CONFIRM,
    LAST_SELECT,
)
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


def _should_skip(identifier: str, folder_path: str, rerun: bool) -> bool:
    """True if `rerun=False` and the identifier's JSON already exists on disk."""
    if rerun:
        return False
    return (Path(folder_path) / f"{identifier}.json").exists()


def _record_skip(
    identifier: str,
    folder_path: str,
    saved_paths: Dict[str, Path],
    *,
    verbose: bool,
    skip_msg: str = "Skipping",
) -> Path:
    path = Path(folder_path) / f"{identifier}.json"
    saved_paths[identifier] = path
    if verbose:
        print(f"{skip_msg}: {identifier} (file exists)")
    return path


# ==========================================================================
# Feature-set generators
# ==========================================================================
#
# Each generator below builds and saves one logical group of feature-column
# JSONs. Generators are independent so any group can be rerun on its own.
# The orchestrator `generate_all_feature_column_sets` runs every group.
#
# Naming scheme (groups 1-12):
#   1  general                            GENERAL_FEATURES (no last, no RT/TFD/TSO base or interactions)
#   2  complete                           ALL_FEATURES
#   3  general_plus_last_<x>              general + each LAST_* (ans/confirm/select/all)
#   4  general_plus_<rt|tfd|tso|rt_tfd_tso>  general + each base RT-family group
#   5  <group3|group4>_plus_<rt_int|tfd_int|rt_tfd_int>  add interaction terms
#   6  baseline_<last_*|rt|tfd|tso|rt_tfd_tso>  8 baselines, each set alone
#   7  select_1[_plus_<...>]              SELECT_1_COLS in place of general for groups 3+4
#   8  derived_with_num_selects           DERIVED_COLS + NUM_OF_SELECTS
#   9  question_only                      PER_QUESTION_COLS
#   10 area_only                          AREA_COLS
#   11 interactions_and_rt_tfd_tso        RT/TFD/TSO base + RT/TFD interactions
#   12 <general|complete>_<pruned_t**|aic|pruned_t**_aic>  feature selection (7 each)

DEFAULT_CORR_THRESHOLDS: Tuple[float, ...] = (0.5, 0.7, 0.9)


def _last_groups() -> Dict[str, List[str]]:
    return {
        "last_ans": list(fg.LAST_ANSWER),
        "last_confirm": list(fg.LAST_CONFIRM),
        "last_select": list(fg.LAST_SELECT),
        "last_all": list(fg.LAST_ALL),
    }


def _rt_family_groups() -> Dict[str, List[str]]:
    return {
        "rt": list(fg.RT_COLS),
        "tfd": list(fg.TFD_COLS),
        "tso": list(fg.TIME_SINCE_OFFSET_COLS),
        "rt_tfd_tso": list(fg.RT_TFD_OFFSET_COLS),
    }


def _interaction_groups() -> Dict[str, List[str]]:
    return {
        "rt_int": list(fg.RT_INTERACTION_COLS),
        "tfd_int": list(fg.TFD_INTERACTION_COLS),
        "rt_tfd_int": list(fg.RT_TFD_INTERACTION_COLS),
    }


def _save_set_collection(
    sets: Dict[str, Sequence[str]],
    folder_path: str,
    verbose: bool,
    *,
    rerun: bool = True,
) -> Dict[str, Path]:
    """Save a {identifier: cols} mapping and return {identifier: path}."""
    saved_paths: Dict[str, Path] = {}
    for identifier, cols in sets.items():
        if _should_skip(identifier, folder_path, rerun):
            _record_skip(identifier, folder_path, saved_paths, verbose=verbose)
            continue
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
    *,
    rerun: bool = True,
) -> Dict[str, Path]:
    """Correlation-prune `base_cols` at each threshold and save."""
    saved_paths: Dict[str, Path] = {}
    base_cols = _dedupe_keep_order(base_cols)
    for threshold in corr_thresholds:
        thr_tag = _threshold_to_tag(threshold)
        identifier = f"{base_name}_pruned_t{thr_tag}"
        if _should_skip(identifier, folder_path, rerun):
            _record_skip(identifier, folder_path, saved_paths, verbose=verbose)
            continue
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
    *,
    rerun: bool = True,
) -> Dict[str, Path]:
    """AIC forward-select on `base_cols` and save."""
    saved_paths: Dict[str, Path] = {}
    base_cols = _dedupe_keep_order(base_cols)
    identifier = f"{base_name}_aic"
    if _should_skip(identifier, folder_path, rerun):
        _record_skip(identifier, folder_path, saved_paths, verbose=verbose)
        return saved_paths
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


def _generate_prune_then_aic_for_base(
    trial_df: pd.DataFrame,
    base_name: str,
    base_cols: Sequence[str],
    folder_path: str,
    target_col: str,
    corr_thresholds: Sequence[float],
    standardize: bool,
    verbose: bool,
    *,
    rerun: bool = True,
) -> Dict[str, Path]:
    """Correlation-prune at each threshold, then AIC forward-select."""
    saved_paths: Dict[str, Path] = {}
    base_cols = _dedupe_keep_order(base_cols)
    for threshold in corr_thresholds:
        thr_tag = _threshold_to_tag(threshold)
        identifier = f"{base_name}_pruned_t{thr_tag}_aic"
        if _should_skip(identifier, folder_path, rerun):
            _record_skip(identifier, folder_path, saved_paths, verbose=verbose)
            continue
        kept_cols, _, _ = correlation_prune_features(
            df=trial_df,
            feature_cols=base_cols,
            target_col=target_col,
            corr_threshold=threshold,
            verbose=False,
        )
        aic_cols, _, _ = aic_forward_select_logit(
            df=trial_df,
            feature_cols=kept_cols,
            target_col=target_col,
            standardize=standardize,
            verbose=False,
        )
        _save_one(aic_cols, identifier, folder_path, saved_paths)
        if verbose:
            print(
                f"Saved: {identifier} "
                f"({len(aic_cols)} after AIC / {len(kept_cols)} after prune / {len(base_cols)} input)"
            )
    return saved_paths


# --------------------------------------------------------------------------
# Group 1: general (no last, no RT/TFD/TSO base or interactions)
# --------------------------------------------------------------------------
def generate_general_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"general": fg.GENERAL_FEATURES}
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 2: complete (everything = ALL_FEATURES)
# --------------------------------------------------------------------------
def generate_complete_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"complete": fg.ALL_FEATURES}
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 3: general + each last variant (4 sets)
# --------------------------------------------------------------------------
def generate_general_plus_last_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    base = list(fg.GENERAL_FEATURES)
    sets: Dict[str, Sequence[str]] = {
        f"general_plus_{name}": _dedupe_keep_order(base + cols)
        for name, cols in _last_groups().items()
    }
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 4: general + each RT-family variant (rt / tfd / tso / all three)
# --------------------------------------------------------------------------
def generate_general_plus_rt_family_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    base = list(fg.GENERAL_FEATURES)
    sets: Dict[str, Sequence[str]] = {
        f"general_plus_{name}": _dedupe_keep_order(base + cols)
        for name, cols in _rt_family_groups().items()
    }
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 5: groups 3 and 4 augmented with RT / TFD / both interaction terms
#   For each base in (last_ans, last_confirm, last_select, last_all,
#   rt, tfd, tso, rt_tfd_tso) we add each of (rt_int, tfd_int, rt_tfd_int)
#   on top of `general + base`. 8 * 3 = 24 sets.
# --------------------------------------------------------------------------
def generate_general_plus_addons_with_interactions_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    base = list(fg.GENERAL_FEATURES)
    sets: Dict[str, Sequence[str]] = {}
    addons = {**_last_groups(), **_rt_family_groups()}
    interactions = _interaction_groups()
    for addon_name, addon_cols in addons.items():
        for int_name, int_cols in interactions.items():
            identifier = f"general_plus_{addon_name}_plus_{int_name}"
            sets[identifier] = _dedupe_keep_order(base + addon_cols + int_cols)
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 6: 8 baselines — each last group alone, each RT-family group alone
# --------------------------------------------------------------------------
def generate_baseline_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {}
    for name, cols in {**_last_groups(), **_rt_family_groups()}.items():
        sets[f"baseline_{name}"] = list(cols)
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 7: select_1 in place of general for groups 3+4
#   select_1 alone, plus 4 last-variants, plus 4 RT-family variants (9 sets).
# --------------------------------------------------------------------------
def generate_select_1_sets(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    base = list(fg.SELECT_1_COLS)
    sets: Dict[str, Sequence[str]] = {"select_1": _dedupe_keep_order(base)}
    for name, cols in {**_last_groups(), **_rt_family_groups()}.items():
        sets[f"select_1_plus_{name}"] = _dedupe_keep_order(base + cols)
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 8: derived columns + NUM_OF_SELECTS
# --------------------------------------------------------------------------
def generate_derived_with_num_selects_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {
        "derived_with_num_selects": list(fg.DERIVED_COLS) + [Con.NUM_OF_SELECTS],
    }
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 9: question features only (per-metric '__question' columns)
# --------------------------------------------------------------------------
def generate_question_only_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"question_only": list(fg.PER_QUESTION_COLS)}
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 10: AREA_COLS only (correct/wrong_mean/contrast/distance_*)
# --------------------------------------------------------------------------
def generate_area_only_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {"area_only": list(fg.AREA_COLS)}
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 11: RT/TFD/TSO base columns + RT/TFD interaction terms (one set)
# --------------------------------------------------------------------------
def generate_interactions_and_rt_family_set(
    folder_path: str = COL_SAVE_PATH,
    *,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    sets: Dict[str, Sequence[str]] = {
        "interactions_and_rt_tfd_tso": _dedupe_keep_order(
            list(fg.RT_TFD_OFFSET_COLS) + list(fg.RT_TFD_INTERACTION_COLS)
        )
    }
    return _save_set_collection(sets, folder_path, verbose, rerun=rerun)


# --------------------------------------------------------------------------
# Group 12: feature selection on `general` and `complete`
#   For each base: pruned at thresholds, AIC, then prune-then-AIC.
#   7 sets per base, 14 total.
# --------------------------------------------------------------------------
def generate_feature_selection_sets(
    trial_df: pd.DataFrame,
    folder_path: str = COL_SAVE_PATH,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    corr_thresholds: Sequence[float] = DEFAULT_CORR_THRESHOLDS,
    standardize: bool = True,
    verbose: bool = True,
    rerun: bool = True,
) -> Dict[str, Path]:
    saved_paths: Dict[str, Path] = {}
    bases: List[Tuple[str, Sequence[str]]] = [
        ("general", fg.GENERAL_FEATURES),
        ("complete", fg.ALL_FEATURES),
    ]
    for base_name, base_cols in bases:
        saved_paths.update(
            _generate_pruned_for_base(
                trial_df=trial_df,
                base_name=base_name,
                base_cols=base_cols,
                folder_path=folder_path,
                target_col=target_col,
                corr_thresholds=corr_thresholds,
                verbose=verbose,
                rerun=rerun,
            )
        )
        saved_paths.update(
            _generate_aic_for_base(
                trial_df=trial_df,
                base_name=base_name,
                base_cols=base_cols,
                folder_path=folder_path,
                target_col=target_col,
                standardize=standardize,
                verbose=verbose,
                rerun=rerun,
            )
        )
        saved_paths.update(
            _generate_prune_then_aic_for_base(
                trial_df=trial_df,
                base_name=base_name,
                base_cols=base_cols,
                folder_path=folder_path,
                target_col=target_col,
                corr_thresholds=corr_thresholds,
                standardize=standardize,
                verbose=verbose,
                rerun=rerun,
            )
        )
    return saved_paths


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
    rerun: bool = True,
) -> Dict[str, Path]:
    """
    Run every feature-set generator (groups 1-12). Returns a single
    {identifier: saved_path} dict. Individual generators can be called
    directly for partial reruns.

    rerun : bool
        If False, skip identifiers whose JSON already exists in
        `folder_path` — including the (potentially expensive) pruning and
        AIC computations they would otherwise trigger. The skipped
        identifiers are still recorded in the returned dict pointing at
        their existing files.
    """
    saved_paths: Dict[str, Path] = {}

    # 1, 2: general and complete bases
    saved_paths.update(generate_general_set(folder_path, verbose=verbose, rerun=rerun))
    saved_paths.update(generate_complete_set(folder_path, verbose=verbose, rerun=rerun))

    # 3, 4: general + last / RT-family variants
    saved_paths.update(
        generate_general_plus_last_sets(folder_path, verbose=verbose, rerun=rerun)
    )
    saved_paths.update(
        generate_general_plus_rt_family_sets(folder_path, verbose=verbose, rerun=rerun)
    )

    # 5: groups 3+4 with RT / TFD / both interaction terms
    saved_paths.update(
        generate_general_plus_addons_with_interactions_sets(
            folder_path, verbose=verbose, rerun=rerun
        )
    )

    # 6: 8 baselines
    saved_paths.update(generate_baseline_sets(folder_path, verbose=verbose, rerun=rerun))

    # 7: select_1 + each last / RT-family addon
    saved_paths.update(generate_select_1_sets(folder_path, verbose=verbose, rerun=rerun))

    # 8, 9, 10, 11: focused subsets
    saved_paths.update(
        generate_derived_with_num_selects_set(folder_path, verbose=verbose, rerun=rerun)
    )
    saved_paths.update(
        generate_question_only_set(folder_path, verbose=verbose, rerun=rerun)
    )
    saved_paths.update(
        generate_area_only_set(folder_path, verbose=verbose, rerun=rerun)
    )
    saved_paths.update(
        generate_interactions_and_rt_family_set(folder_path, verbose=verbose, rerun=rerun)
    )

    # 12: feature selection on general and complete
    saved_paths.update(
        generate_feature_selection_sets(
            trial_df, folder_path,
            target_col=target_col,
            corr_thresholds=corr_thresholds,
            standardize=standardize_aic,
            verbose=verbose,
            rerun=rerun,
        )
    )

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
    contains 'pruned'or 'aic', count how often each feature appears across files, and plot it.

    """
    columns_folder = Path(columns_folder)

    if recursive:
        json_paths = sorted(columns_folder.rglob("*.json"))
    else:
        json_paths = sorted(columns_folder.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {columns_folder}")

    def matches(path: Path) -> bool:
        name = path.name if case_sensitive else path.name.lower()
        prune = "pruned" if case_sensitive else "pruned"
        aic = "aic" if case_sensitive else "aic"
        return (prune in name) or (aic in name)

    matched_files = [p for p in json_paths if matches(p)]

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
    name_must_contain: Optional[Sequence[str]] = None,
    name_any_of: Optional[Sequence[str]] = ("pruned", "aic"),
    save_plot_flag: bool = True,

    plot_rel_dir: str = "feature_selection",
    plot_filename: str = "k_most_frequent_feature_frequency",
    paper_dirs=None,
    verbose: bool = True,
):
    """
    Builds k-most-frequent feature sets from JSON files whose name matches
    the given substring filters and saves:

    - k_most_frequent
    - k_most_frequent_last_ans
    - k_most_frequent_last_confirm
    - k_most_frequent_last_select
    - k_most_frequent_last_all

    Filtering
    ---------
    name_must_contain : every substring in this sequence must appear in the
        file name (AND). Pass None/empty to skip this check.
    name_any_of : at least one substring in this sequence must appear in
        the file name (OR). Pass None/empty to skip this check.

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

    must = tuple(name_must_contain or ())
    any_of = tuple(name_any_of or ())
    if not case_sensitive:
        must = tuple(s.lower() for s in must)
        any_of = tuple(s.lower() for s in any_of)

    def _matches(p: Path):
        name = p.name if case_sensitive else p.name.lower()
        if must and not all(s in name for s in must):
            return False
        if any_of and not any(s in name for s in any_of):
            return False
        return True

    matched_files = [p for p in json_paths if _matches(p)]

    if not matched_files:
        raise FileNotFoundError(
            f"No JSON files matching must={must!r}, any_of={any_of!r} found in {columns_folder}"
        )

    if verbose:
        print(
            f"Using {len(matched_files)} files (must={list(must)}, any_of={list(any_of)})"
        )

    # ------------------------------------------------------------------
    # Last groups (same as your generator)
    # ------------------------------------------------------------------
    LAST_ALL

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
        f"{k}_most_frequent_last_ans": _dedupe_keep_order(k_base + LAST_ANSWER),
        f"{k}_most_frequent_last_confirm": _dedupe_keep_order(k_base + LAST_SELECT),
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