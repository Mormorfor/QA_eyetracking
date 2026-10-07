"""Train/test splits and predefined fold assignments.

Split out of `predictive_modeling/common/data_utils.py` in stage D step 7
(2026-10-07). That module held two unrelated things -- how you partition the
data, and how you put a confidence interval on a coefficient -- behind a name
("data utils") that described neither. They are now two modules, each named
after its job.

The counterpart is `modeling/inference.py`.

The second half of this module -- the regime vocabulary and the fold-assignment
loaders -- arrived from `answer_correctness/cross_validation.py` in the same step
(map §6.3). Both halves answer the same question, "which rows train and which
rows are held out", so they belong together: one builds a split, the other reads
a split someone already decided and stored.
"""

import warnings
from pathlib import Path
from typing import Sequence, Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import HUNTERS_FOLDS_DIR, GATHERERS_REFOLDED_DIR


# user-facing regime name -> fold-file regime suffix
_REGIME_SUFFIX = {
    "new_item": "_seen_subject_unseen_item",
    "new_subject": "_unseen_subject_seen_item",
    "both": "_unseen_subject_unseen_item",
    "new_item_and_subject": "_unseen_subject_unseen_item",  # alias for "both"
}
_FOLD_DIRS = {
    "hunters": HUNTERS_FOLDS_DIR,
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
    Fold-based train/test split.

    Selects one fold (random unless `fold` is provided), then for that fold:
      - train rows come from the `train_train` regime,
      - test rows come from the requested (test_split, test_regime) combinations.

    Trial-set membership is read from the precomputed fold files of the chosen
    `sources` (hunters fold + gatherers-refolded fold by default), and matched
    onto `df` via (participant_id, text_id).

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
    For a given participant:
    - randomly select one trial as test
    - all other trials are train

    ``random_state`` seeds the trial choice so the single held-out trial is
    reproducible across runs (default: None -> non-deterministic).
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
# Predefined fold assignments, and the regime vocabulary that names them
#
# From `answer_correctness/cross_validation.py`, stage D step 7. Nothing here is
# specific to answer correctness -- it reads fold CSVs and attaches their regime
# labels to a frame -- which is why it moved.
# ===========================================================================

# ---------------------------------------------------------------------
# Regime vocabulary
# ---------------------------------------------------------------------
# A fold regime name encodes two independent things, and reading it as one
# opaque string is what made the summary tables hard to interpret (T3.9):
#
#   test_unseen_subject_unseen_item
#   ^^^^                               SPLIT   -- which held-out half
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^   NOVELTY -- what kind of generalization
#
# NOVELTY is the scientific question -- a new item, a new subject, or both --
# and is the axis the paper reports on, under the short names below.
#
# SPLIT is an artifact of how the folds were built: each held-out novelty cell
# was halved so one half could be used for tuning. Nothing here tunes (the
# logreg has no searched hyperparameters), so `val` and `test` are two
# interchangeable held-out samples of the same population. They are kept
# separable rather than merged because that may stop being true; `eval_split`
# is the choice. Pooling matters most for the `both` cell, which is by far the
# smallest (~97 trials per fold) and is where the headline number comes from.

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
# Fold loading / assignment
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
    Load one fold CSV containing train/val/test regime assignments.
    """
    fold_df = pd.read_csv(fold_csv_path)


    #for col in [participant_col_fold, text_col_fold, trial_col_fold, regime_col_fold]:
    #        fold_df[col] = fold_df[col].astype(str).str.strip()

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
    Attach regime labels using (participant_id, text_id) join ONLY.
    Includes normalization (str, strip, lower).
    Keeps only the fold columns needed for the join + regime.
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
# Prebuilt trial-level features
# ---------------------------------------------------------------------

REGIME_COL = "regime"


def attach_trial_level_fold_regimes(
    trial_df: pd.DataFrame,
    df_fold: pd.DataFrame,
    *,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Restrict a prebuilt trial-level feature table to one fold, tagging each row
    with its regime.

    ``df_fold`` is the word-level frame :func:`attach_fold_regimes` produced, so
    the (participant, text) -> regime matching stays in one place and is only
    collapsed to one row per trial here. Trials missing from the fold assignment
    are dropped (inner join); any ``keep_cols`` the prebuilt table lacks are
    carried over from ``df_fold``.
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
    Load and vertically stack the fold-``fold_idx`` assignment CSVs from each
    directory in ``fold_dirs``.

    Used to build an "all participants" fold by combining, e.g., the hunters
    fold_0 and the (refolded) gatherers fold_0 into a single regime-assignment
    table. Participant ids are disjoint across the two groups, so stacking the
    assignments is sufficient.
    """
    frames = []
    for fold_dir in fold_dirs:
        fold_path = Path(fold_dir) / fold_filename_template.format(fold_idx=fold_idx)
        frames.append(load_fold_assignment_csv(fold_path))
    return pd.concat(frames, ignore_index=True)
