"""Which trials train a model and which are held out.

Two kinds of split:

* **Stored folds.** Ten predefined folds per group, one file each, in
  `data/cv_folds/HuntersRefolded/` and `GatherersRefolded/`. A fold file gives
  each participant x paragraph pair a *regime*: whether that person and that
  paragraph are in training, and if not, which held-out set they are in. The
  two groups hold out the same paragraphs in each fold.
  `group_vise_train_test_split` takes one fold; the loaders and
  `attach_fold_regimes` below feed cross-validation (`crossval.py`).
* **Leave one trial out**, for models fitted on a single participant.
"""

import warnings
from pathlib import Path
from typing import Sequence, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import HUNTERS_REFOLDED_DIR, GATHERERS_REFOLDED_DIR


# Short regime name -> how the fold files spell it (after "val_" / "test_").
_REGIME_SUFFIX = {
    "new_item": "_seen_subject_unseen_item",
    "new_subject": "_unseen_subject_seen_item",
    "both": "_unseen_subject_unseen_item",
    "new_item_and_subject": "_unseen_subject_unseen_item",  # alias for "both"
}
_FOLD_DIRS = {
    "hunters": HUNTERS_REFOLDED_DIR,
    "gatherers": GATHERERS_REFOLDED_DIR,
}

def group_vise_train_test_split(
    df: pd.DataFrame,
    *,
    test_regimes: Sequence[str],
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),
    n_folds: int = 10,
    random_state: Optional[int] = None,
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    fold_participant_col: str = "participant_id",
    fold_text_col: str = "unique_paragraph_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Take one stored fold and return its training and held-out trials.

    Picks one fold (at random unless `fold` is given). The training rows are
    that fold's `train_train` regime; the held-out rows are the requested
    regimes, in the requested split(s).

    The fold files of `sources` (both groups' refolded files, by default) are
    matched onto `df` by participant and paragraph (text_id). Both are turned into text, stripped and lower-cased
    first, and stay that way in the returned frames. Trials that are not in a
    fold file are dropped, without a message.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `df_participant_col` and `df_text_col`.
    test_regimes : Sequence[str]
        Any subset of {"new_item", "new_subject", "both"}. "new_item_and_subject"
        is accepted as an alias for "both".
    test_split : str
        "test", "val", or "both".
    fold : int, optional
        Fold index in [0, n_folds). If None, picked uniformly at random.
    sources : Sequence[str]
        Which fold-file directories to merge for the trial assignments. Any
        subset of {"hunters", "gatherers"}.

    Returns
    -------
    (train_df, test_df, info)
        info dict reports the chosen fold, regime labels, and row counts.
    """
    test_regimes = list(test_regimes)
    if not test_regimes:
        raise ValueError(
            "test_regimes must contain at least one of "
            f"{sorted(set(_REGIME_SUFFIX) - {'new_item_and_subject'})}"
        )
    bad = [r for r in test_regimes if r not in _REGIME_SUFFIX]
    if bad:
        raise ValueError(f"Unknown test_regime(s): {bad}")
    if test_split not in ("test", "val", "both"):
        raise ValueError("test_split must be one of 'test', 'val', 'both'")
    bad_src = [s for s in sources if s not in _FOLD_DIRS]
    if bad_src:
        raise ValueError(f"Unknown source(s): {bad_src}")

    rng = np.random.default_rng(random_state)
    if fold is None:
        fold = int(rng.integers(0, n_folds))

    splits = ("test", "val") if test_split == "both" else (test_split,)
    test_regime_labels = {
        f"{s}{_REGIME_SUFFIX[r]}" for s in splits for r in test_regimes
    }

    fold_dfs = []
    for src in sources:
        path = _FOLD_DIRS[src] / f"fold_{fold}_trial_ids_by_regime.csv"
        fold_dfs.append(
            pd.read_csv(path)[[fold_participant_col, fold_text_col, "regime"]]
        )
    fold_info = pd.concat(fold_dfs, ignore_index=True).drop_duplicates()

    out = df.copy()
    out[df_participant_col] = out[df_participant_col].astype(str).str.strip().str.lower()
    out[df_text_col] = out[df_text_col].astype(str).str.strip().str.lower()
    fold_info[fold_participant_col] = (
        fold_info[fold_participant_col].astype(str).str.strip().str.lower()
    )
    fold_info[fold_text_col] = (
        fold_info[fold_text_col].astype(str).str.strip().str.lower()
    )
    fold_info = fold_info.rename(
        columns={
            fold_participant_col: df_participant_col,
            fold_text_col: df_text_col,
        }
    )

    out = out.merge(fold_info, on=[df_participant_col, df_text_col], how="inner")

    train_df = out[out["regime"] == "train_train"].drop(columns=["regime"]).copy()
    test_df = out[out["regime"].isin(test_regime_labels)].drop(columns=["regime"]).copy()

    info = {
        "fold": fold,
        "sources": tuple(sources),
        "test_regimes": tuple(test_regimes),
        "test_split": test_split,
        "fold_test_regime_labels": tuple(sorted(test_regime_labels)),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    return train_df, test_df, info


def leave_one_trial_out_for_participant(
    df: pd.DataFrame,
    participant_id,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out one randomly chosen trial of one participant; their other trials
    are the training set.

    ``random_state`` fixes which trial is chosen (default None: a different
    one on each run).
    """

    df = df.copy()
    df_p = df[df[participant_col] == participant_id].copy()

    trials = df_p[trial_col].dropna().unique()

    rng = np.random.default_rng(random_state)
    test_trial = rng.choice(trials)

    test_df = df_p[df_p[trial_col] == test_trial].copy()
    train_df = df_p[df_p[trial_col] != test_trial].copy()

    return train_df, test_df


def iter_leave_one_trial_out_for_participant(
    df: pd.DataFrame,
    participant_id,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
):
    """
    Full leave-one-trial-out generator for a single participant.

    Yields ``(train_df, test_df, test_trial)`` once per trial of the
    participant: each trial is held out as the single-row test set exactly
    once while all the participant's remaining trials form the train set.
    Trials are visited in first-seen order, so the iteration is deterministic
    (no RNG involved).
    """
    df_p = df[df[participant_col] == participant_id].copy()
    trials = pd.unique(df_p[trial_col].dropna())

    for test_trial in trials:
        test_df = df_p[df_p[trial_col] == test_trial].copy()
        train_df = df_p[df_p[trial_col] != test_trial].copy()
        yield train_df, test_df, test_trial

# ===========================================================================
# The stored folds, and the names of their regimes
# ===========================================================================

# ---------------------------------------------------------------------
# Regime names
# ---------------------------------------------------------------------
# A regime name holds two separate things:
#
#   test_unseen_subject_unseen_item
#   ^^^^                               SPLIT   -- which held-out half
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^   NOVELTY -- what is new to the model
#
# NOVELTY is the scientific question -- a new paragraph, a new person, or
# both -- and is what the paper reports, under the short names below.
#
# SPLIT comes from how the folds were built: each held-out set was halved so
# that one half could be used for tuning. Nothing is tuned here (the logistic
# regression has no searched settings), so `val` and `test` are two
# interchangeable held-out samples. They are kept apart in case that changes;
# `eval_split` chooses which to use. Pooling them matters most for `both`, the
# smallest novelty (per fold and group, 78-120 validation trials and 234-360
# test trials; the test half also holds the pairs that cross the two halves).

NOVELTY_BY_CELL: Dict[str, str] = {
    "seen_subject_unseen_item": "new_item",
    "unseen_subject_seen_item": "new_subject",
    "unseen_subject_unseen_item": "both",
}

EVAL_SPLITS = ("test", "val", "both")


def parse_regime(regime: str) -> Tuple[str, str]:
    """``"test_unseen_subject_unseen_item"`` -> ``("test", "both")``."""
    for split in ("val", "test"):
        prefix = f"{split}_"
        if regime.startswith(prefix):
            cell = regime[len(prefix):]
            return split, NOVELTY_BY_CELL.get(cell, cell)
    return "train", NOVELTY_BY_CELL.get(regime, regime)


def eval_regimes_for_split(eval_split: str = "both") -> List[str]:
    """The evaluation regimes belonging to `eval_split`, in reporting order."""
    if eval_split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got {eval_split!r}")
    splits = ("val", "test") if eval_split == "both" else (eval_split,)
    return [f"{s}_{cell}" for cell in NOVELTY_BY_CELL for s in splits]

# ---------------------------------------------------------------------
# Reading fold files, and adding the regimes to a table
# ---------------------------------------------------------------------

def load_fold_assignment_csv(
    fold_csv_path: str | Path,
    *,
    participant_col_fold: str = "participant_id",
    text_col_fold: str = "unique_paragraph_id",
    trial_col_fold: str = "unique_trial_id",
    regime_col_fold: str = "regime",
) -> pd.DataFrame:
    """
    Read one fold file: one row per participant x paragraph, with its regime.

    The `*_col_fold` arguments are not used.
    """
    fold_df = pd.read_csv(fold_csv_path)

    return fold_df


def attach_fold_regimes(
    df: pd.DataFrame,
    fold_df: pd.DataFrame,
    *,
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    fold_participant_col: str = "participant_id",
    fold_text_col: str = "unique_paragraph_id",
    fold_regime_col: str = "regime",
) -> pd.DataFrame:
    """
    Add each trial's regime for this fold, matching on participant and
    paragraph (text_id) only.

    Both sides' ids are turned into text, stripped and lower-cased before
    matching, and stay that way in the result. Of the fold file, only the
    matching columns and the regime are kept.

    Trials that are not in the fold file are dropped, without a message. The
    refolded files give every trial of their group a regime, so with them
    nothing is dropped.
    """

    out = df.copy()
    fold_df = fold_df.copy()

    out[df_participant_col] = (
        out[df_participant_col].astype(str).str.strip().str.lower()
    )
    out[df_text_col] = (
        out[df_text_col].astype(str).str.strip().str.lower()
    )

    fold_df = fold_df[
        [fold_participant_col, fold_text_col, fold_regime_col]
    ].copy()

    fold_df[fold_participant_col] = (
        fold_df[fold_participant_col].astype(str).str.strip().str.lower()
    )
    fold_df[fold_text_col] = (
        fold_df[fold_text_col].astype(str).str.strip().str.lower()
    )
    fold_df[fold_regime_col] = (
        fold_df[fold_regime_col].astype(str).str.strip()
    )

    fold_df = fold_df.drop_duplicates()

    assign_df = fold_df.rename(
        columns={
            fold_participant_col: df_participant_col,
            fold_text_col: df_text_col,
        }
    )

    out = out.merge(
        assign_df,
        on=[df_participant_col, df_text_col],
        how="inner",
    )

    return out

# ---------------------------------------------------------------------
# Folds for a ready-made trial table
# ---------------------------------------------------------------------

REGIME_COL = "regime"


def attach_trial_level_fold_regimes(
    trial_df: pd.DataFrame,
    df_fold: pd.DataFrame,
    *,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Cut a ready-made trial table (one row per trial) down to one fold, adding
    each row's regime.

    ``df_fold`` is the word-level frame :func:`attach_fold_regimes` made, so
    the matching of trials to regimes is done in one place only. Trials that
    are not in the fold are dropped; any ``keep_cols`` the trial table lacks
    are taken from ``df_fold``.
    """
    keys = list(Con.TRIAL_ID_COLS)

    fold_regimes = (
        df_fold[keys + [REGIME_COL]]
        .drop_duplicates(subset=keys)
        .reset_index(drop=True)
    )

    base = trial_df.drop(columns=[REGIME_COL]) if REGIME_COL in trial_df.columns else trial_df
    out = base.merge(fold_regimes, on=keys, how="inner")

    missing = [c for c in (keep_cols or []) if c not in out.columns]
    if missing:
        extra = df_fold[keys + missing].drop_duplicates(subset=keys)
        out = out.merge(extra, on=keys, how="left")

    return out

def load_combined_fold_assignment_csv(
    fold_dirs: Sequence[str | Path],
    fold_idx: int,
    *,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
) -> pd.DataFrame:
    """
    Read fold ``fold_idx`` from each folder in ``fold_dirs`` and stack them.

    This makes an "all participants" fold, e.g. the hunters' fold 0 plus the
    gatherers' fold 0. No participant is in both groups, and the refolded
    files of both groups hold out the same paragraphs, so stacking the two
    files is enough.
    """
    frames = []
    for fold_dir in fold_dirs:
        fold_path = Path(fold_dir) / fold_filename_template.format(fold_idx=fold_idx)
        frames.append(load_fold_assignment_csv(fold_path))
    return pd.concat(frames, ignore_index=True)
