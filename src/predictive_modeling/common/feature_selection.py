import pandas as pd
import numpy as np
import statsmodels.api as sm

from typing import Optional, Sequence, Any

from src import constants as Con



def correlation_prune_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: Optional[str] = None,
    corr_threshold: float = 0.80,
    verbose: bool = True,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Iteratively prune highly correlated features.

    Rule:
    - If two features have abs(corr) >= corr_threshold, drop one.
    - If target_col is provided, keep the feature with stronger absolute
      correlation to the target.
    - Otherwise, keep the one that appears first in feature_cols.

    Returns
    -------
    kept_cols : list[str]
        Features remaining after pruning.
    dropped_cols : list[str]
        Features removed by pruning.
    prune_log : pd.DataFrame
        Row-by-row record of pruning decisions.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    work_cols = feature_cols.copy()

    X = df[work_cols].apply(pd.to_numeric, errors="coerce")

    target_scores = {}
    if target_col is not None:
        y = pd.to_numeric(df[target_col], errors="coerce")
        for col in work_cols:
            valid = X[col].notna() & y.notna()
            if valid.sum() < 3:
                target_scores[col] = -np.inf
            else:
                target_scores[col] = abs(X.loc[valid, col].corr(y.loc[valid]))
                if pd.isna(target_scores[col]):
                    target_scores[col] = -np.inf

    prune_steps = []
    dropped = set()

    while True:
        current_cols = [c for c in work_cols if c not in dropped]
        if len(current_cols) <= 1:
            break

        corr_mat = X[current_cols].corr().abs()
        corr_values = corr_mat.to_numpy(copy=True)

        np.fill_diagonal(corr_values, np.nan)

        max_corr = np.nanmax(corr_values)
        if pd.isna(max_corr) or max_corr < corr_threshold:
            break

        i, j = np.where(corr_values == max_corr)
        f1 = current_cols[i[0]]
        f2 = current_cols[j[0]]

        if target_col is not None:
            s1 = target_scores.get(f1, -np.inf)
            s2 = target_scores.get(f2, -np.inf)

            if s1 > s2:
                keep, drop = f1, f2
                reason = "kept higher abs(feature-target corr)"
            elif s2 > s1:
                keep, drop = f2, f1
                reason = "kept higher abs(feature-target corr)"
            else:
                keep, drop = f1, f2
                reason = "tie on target score; kept first"
        else:
            keep, drop = f1, f2
            reason = "no target provided; kept first"

        dropped.add(drop)

        prune_steps.append(
            {
                "feature_1": f1,
                "feature_2": f2,
                "pair_abs_corr": float(max_corr),
                "kept": keep,
                "dropped": drop,
                "kept_target_score": target_scores.get(keep, np.nan) if target_col else np.nan,
                "dropped_target_score": target_scores.get(drop, np.nan) if target_col else np.nan,
                "reason": reason,
            }
        )

        if verbose:
            print(
                f"Dropping '{drop}' (corr={max_corr:.3f} with '{keep}') | reason: {reason}"
            )

    kept_cols = [c for c in work_cols if c not in dropped]
    dropped_cols = [c for c in work_cols if c in dropped]
    prune_log = pd.DataFrame(prune_steps)

    return kept_cols, dropped_cols, prune_log



def aic_forward_select_logit(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    standardize: bool = True,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame, Any]:
    """
    Simple forward AIC selection for logistic regression.

    Returns
    -------
    selected_cols : list[str]
    log_df : pd.DataFrame
    final_model : fitted statsmodels model
    """
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int).copy()

    if standardize:
        X = (X - X.mean()) / X.std(ddof=0)
        X = X.fillna(0.0)

    selected = []
    remaining = feature_cols.copy()
    log_rows = []

    def fit_aic(cols):
        X_model = sm.add_constant(X[cols], has_constant="add") if cols else sm.add_constant(
            pd.DataFrame(index=X.index), has_constant="add"
        )
        model = sm.Logit(y, X_model).fit(disp=False)
        return model.aic, model

    current_aic, current_model = fit_aic([])

    if verbose:
        print(f"Start AIC: {current_aic:.3f}")

    while remaining:
        best_feature = None
        best_aic = current_aic
        best_model = current_model

        for col in remaining:
            trial_cols = selected + [col]
            try:
                trial_aic, trial_model = fit_aic(trial_cols)
            except Exception:
                continue

            if trial_aic < best_aic:
                best_feature = col
                best_aic = trial_aic
                best_model = trial_model

        if best_feature is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)

        log_rows.append({
            "step": len(selected),
            "added": best_feature,
            "aic_before": current_aic,
            "aic_after": best_aic,
            "delta_aic": best_aic - current_aic,
            "n_features": len(selected),
        })

        if verbose:
            print(f"Step {len(selected)}: add '{best_feature}' | AIC {current_aic:.3f} -> {best_aic:.3f}")

        current_aic = best_aic
        current_model = best_model

    return selected, pd.DataFrame(log_rows), current_model



def elasticnet_select_logit(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    group_col: Optional[str] = Con.PARTICIPANT_ID,
    l1_ratios: Sequence[float] = (0.3, 0.5, 0.8, 1.0),
    n_Cs: int = 20,
    n_splits: int = 5,
    standardize: bool = True,
    class_weight: Optional[str] = "balanced",
    scoring: str = "balanced_accuracy",
    max_iter: int = 20000,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame, Any]:
    """
    Embedded feature selection via cross-validated elastic-net logistic regression.

    Fits an L1/L2-penalised logistic regression, tuning the penalty strength (C)
    and the L1/L2 mix (l1_ratio) by cross-validation. Features whose coefficient
    is shrunk exactly to zero are dropped; the rest are returned as `selected`.

    Robust to the collinearity that destabilises greedy/AIC selection: the L2
    component keeps correlated features together rather than arbitrarily zeroing
    one of them.

    Cross-validation is group-aware when `group_col` is present in `df` (folds
    split by participant, so selection reflects generalisation to new subjects);
    otherwise it falls back to stratified K-fold.

    Note: penalised coefficients are biased (shrunk) -- use this to SELECT, then
    refit a plain model (e.g. via the correctness bundle / TrialLevelLogRegModel)
    on `selected` for interpretation and CIs.

    Returns
    -------
    selected_cols : list[str]
        Features with non-zero coefficients in the refit best model.
    coef_log : pd.DataFrame
        One row per feature: coef, abs_coef, selected (sorted by abs_coef).
    model : sklearn LogisticRegressionCV
        The fitted estimator (best C in `model.C_`, best l1_ratio in
        `model.l1_ratio_`).
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int).copy()

    if standardize:
        X = (X - X.mean()) / X.std(ddof=0)
        X = X.fillna(0.0)

    # Group-aware CV when possible, so selection targets new-subject generalisation.
    if group_col is not None and group_col in df.columns:
        groups = df.loc[valid, group_col].to_numpy()
        n_groups = pd.unique(groups).size
        splits = min(n_splits, n_groups)
        cv = list(GroupKFold(n_splits=splits).split(X, y, groups=groups))
        cv_desc = f"GroupKFold(n_splits={splits}) by '{group_col}'"
    else:
        splits = min(n_splits, int(y.value_counts().min()))
        cv = list(
            StratifiedKFold(
                n_splits=splits, shuffle=True, random_state=random_state
            ).split(X, y)
        )
        cv_desc = f"StratifiedKFold(n_splits={splits})"

    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=list(l1_ratios),
        Cs=n_Cs,
        cv=cv,
        scoring=scoring,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    model.fit(X.to_numpy(dtype=float), y.to_numpy())

    coef = model.coef_.reshape(-1)
    coef_log = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "coef": coef,
                "abs_coef": np.abs(coef),
                "selected": coef != 0,
            }
        )
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    selected = coef_log.loc[coef_log["selected"], "feature"].tolist()

    if verbose:
        print(
            f"Elastic-net selection | {cv_desc} | scoring={scoring}\n"
            f"  best l1_ratio={float(model.l1_ratio_[0]):.2f}, "
            f"best C={float(model.C_[0]):.4g}\n"
            f"  selected {len(selected)} / {len(feature_cols)} features"
        )

    return selected, coef_log, model