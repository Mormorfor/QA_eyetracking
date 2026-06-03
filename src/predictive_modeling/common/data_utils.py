# data_utils.py

from typing import Sequence, Tuple, List, Optional, Mapping, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import constants as Con
from src.data_paths import HUNTERS_FOLDS_DIR, GATHERERS_REFOLDED_DIR

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#--------------------------------
# Splits
#--------------------------------

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a given participant:
    - randomly select one trial as test
    - all other trials are train
    """

    df = df.copy()
    df_p = df[df[participant_col] == participant_id].copy()

    trials = df_p[trial_col].dropna().unique()

    rng = np.random.default_rng()
    test_trial = rng.choice(trials)

    test_df = df_p[df_p[trial_col] == test_trial].copy()
    train_df = df_p[df_p[trial_col] != test_trial].copy()

    return train_df, test_df




#--------------------------------
# Summary
#--------------------------------
def get_coef_summary(model: LogisticRegression,
                     feature_cols: List[str],
                     top_k: int = None):
    """
    Get a summary of coefficients from a fitted logistic regression model.
    """
    coef = np.asarray(model.coef_).reshape(-1)
    out = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "coef": coef,
            "odds_ratio": np.exp(coef),
            "abs_coef": np.abs(coef),
        }
    )
    sort_col = "abs_coef"
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)

    if top_k is not None:
        out = out.head(int(top_k)).reset_index(drop=True)

    return out

#--------------------------------
# Stats
#--------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm

# https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
def wald_logreg_coef_cis(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    ci: float = 0.95,

    include_intercept: bool = False,
    use_pinv: bool = True,
) -> pd.DataFrame:
    """
    Wald CIs for sklearn LogisticRegression coefficients.
    """
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float).reshape(-1)

    n, p = Xn.shape

    p_hat = model.predict_proba(X)[:, 1]
    w = p_hat * (1.0 - p_hat)

    X_design = np.hstack([np.ones((n, 1)), Xn])

    Xw = X_design * np.sqrt(w)[:, None]
    A = Xw.T @ Xw

    inv = np.linalg.pinv if use_pinv else np.linalg.inv
    A_inv = inv(A)

    theta = np.concatenate([model.intercept_.reshape(-1), model.coef_.reshape(-1)])

    cov = A_inv
    n_clusters = np.nan

    z = norm.ppf(1 - (1 - float(ci)) / 2)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    ci_low = theta - z * se
    ci_high = theta + z * se

    names = ["intercept"] + list(feature_names)
    out = pd.DataFrame({
        "feature": names,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "or_ci_low": np.exp(ci_low),
        "or_ci_high": np.exp(ci_high),
        "sig_ci": (ci_low > 0) | (ci_high < 0),
        "n_clusters": n_clusters,
    })

    if not include_intercept:
        out = out[out["feature"] != "intercept"].reset_index(drop=True)

    return out


def bootstrap_logreg_coef_cis(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_names: list[str],
    fit_kwargs: dict,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    cluster: np.ndarray = None,
) -> pd.DataFrame:
    """
    Bootstrap coefficient CIs for sklearn LogisticRegression.

    - If cluster is None: classic row bootstrap (resample rows).
    - If cluster is provided: cluster bootstrap (resample clusters with replacement,
      keep all rows for selected clusters).

    Returns a DF with:
      feature, ci_low, ci_high, or_ci_low, or_ci_high, n_boot_ok
    """
    rng = np.random.default_rng(seed)
    Xn = X.to_numpy()
    yn = y.astype(int).to_numpy()

    n, p = Xn.shape
    boot = np.full((n_boot, p), np.nan, dtype=float)

    if cluster is not None:
        cluster = np.asarray(cluster)
        uniq = pd.unique(cluster)
        idx_by_c = {c: np.flatnonzero(cluster == c) for c in uniq}

    ok = 0
    for b in range(n_boot):
        if cluster is None:
            idx = rng.integers(0, n, size=n)
        else:
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_c[c] for c in sampled], axis=0)

        Xb = Xn[idx]
        yb = yn[idx]

        # need both classes in the resample
        if np.unique(yb).size < 2:
            continue

        m = LogisticRegression(**fit_kwargs)
        m.fit(Xb, yb)

        boot[ok, :] = m.coef_.reshape(-1)
        ok += 1

        if ok == n_boot:
            break

    boot = boot[:ok, :]
    alpha = 1.0 - float(ci)
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    ci_low = np.percentile(boot, lo_q, axis=0)
    ci_high = np.percentile(boot, hi_q, axis=0)

    out = pd.DataFrame({
        "feature": feature_names,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "or_ci_low": np.exp(ci_low),
        "or_ci_high": np.exp(ci_high),
        "n_boot_ok": ok,
    })
    out["sig_ci"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


#--------------------------------
# Multicollinearity (VIF)
#--------------------------------

def compute_vif(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """
    Variance Inflation Factor for each feature, on any DataFrame + column set.

    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing feature j on all
    the other features (plus an intercept). It measures how much feature j is
    linearly explained by the rest.

    Reading it:
      * VIF ~ 1      uncorrelated with the others
      * VIF > 5      worth a look, > 10 serious multicollinearity
      * VIF == inf   exactly redundant -- e.g. a complete one-hot dummy set,
                     which is collinear with the intercept (the "dummy trap")

    """
    

    cols = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"compute_vif: skipping {len(missing)} column(s) not in df: {missing}")

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nonconst = [c for c in cols if float(X[c].std()) > 0.0]
    const_cols = [c for c in cols if c not in nonconst]

    Xv = X[nonconst]
    if add_intercept:
        Xv = add_constant(Xv, has_constant="add")  # const prepended as column 0
    arr = Xv.to_numpy(dtype=float)
    offset = 1 if add_intercept else 0

    vifs: dict[str, float] = {}
    notes: dict[str, str] = {}
    with np.errstate(divide="ignore"):  # 1/(1-1) -> inf for redundant columns
        for j, col in enumerate(nonconst):
            vifs[col] = float(variance_inflation_factor(arr, j + offset))
            notes[col] = ""

    for c in const_cols:
        vifs[c] = np.inf
        notes[c] = "constant"

    out = pd.DataFrame(
        {"VIF": pd.Series(vifs), "note": pd.Series(notes)}
    ).sort_values("VIF", ascending=False)
    return out


def vif_from_bundle(
    bundle: Mapping[str, Any],
    model_name: str = "trial_level_log_reg",
    *,
    data: str = "train",
    add_intercept: bool = True,
) -> pd.DataFrame:
    """
    VIF for the exact feature set a correctness bundle fit.

    ``bundle`` is the dict returned by ``run_full_features_correctness_bundle``.
    The feature list is read from the fitted model's ``coef_summary`` (so it is
    exactly what was modelled), and the design matrix is taken from the bundle's
    ``train_df`` (default), ``test_df``, or full ``trial_df`` via ``data=
    "train"|"test"|"all"``.
    """
    res = bundle["results"][model_name]
    if getattr(res, "coef_summary", None) is None:
        raise ValueError(f"Model '{model_name}' has no coef_summary to read features from.")
    feats = list(res.coef_summary["feature"])

    key = {"train": "train_df", "test": "test_df", "all": "trial_df"}.get(data)
    if key is None:
        raise ValueError(f"data must be 'train', 'test', or 'all' (got {data!r}).")

    return compute_vif(bundle[key], feats, add_intercept=add_intercept)


def summarize_random_effects(re_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in re_df.columns if c != re_df.columns[0]]
    out = []

    for c in value_cols:
        s = pd.to_numeric(re_df[c], errors="coerce")
        out.append({
            "term": c,
            "n_levels": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        })

    return pd.DataFrame(out)