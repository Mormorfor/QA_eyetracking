# data_utils.py

from typing import Sequence, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import constants as Con

#--------------------------------
# Splits
#--------------------------------
def group_vise_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_cols: List[str] = [Con.PARTICIPANT_ID, Con.TRIAL_ID],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-wise train/test split
    """
    df = df.copy()
    groups = (
        df[group_cols]
        .dropna()
        .drop_duplicates()
        .apply(tuple, axis=1)
        .to_numpy()
    )

    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_test = max(1, int(round(test_size * n_groups)))

    test_groups = set(groups[:n_test])
    train_groups = set(groups[n_test:])

    group_tuples = df[group_cols].apply(tuple, axis=1)

    train_mask = group_tuples.isin(train_groups)
    test_mask = group_tuples.isin(test_groups)

    return df[train_mask].copy(), df[test_mask].copy()



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
# Helpers
#--------------------------------

def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:

    df_cols = set(df.columns)
    present_cols = [c for c in feature_cols if c in df_cols]
    return df[present_cols].copy()



def build_area_metric_pivot(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    area_col: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse already-aggregated area-level metrics into one row per group,
    pivoting areas into feature columns.

    Input: word-level df with columns:
        group_cols + [area_col] + metric_cols
    Output: one row per group_cols, columns:
        <metric>__<area_label>
    """
    cols_needed = list(group_cols) + [area_col] + list(metric_cols)
    metrics_df = (
        df[cols_needed]
        .dropna(subset=[area_col])
        .groupby(list(group_cols) + [area_col], as_index=False)
        .agg({m: "first" for m in metric_cols})
    )

    metrics_pivot = metrics_df.pivot_table(
        index=list(group_cols),
        columns=area_col,
        values=metric_cols,
        aggfunc="first",
    )

    metrics_pivot.columns = [
        f"{metric}__{area_label}"
        for metric, area_label in metrics_pivot.columns
    ]
    metrics_pivot = metrics_pivot.reset_index()
    return metrics_pivot



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