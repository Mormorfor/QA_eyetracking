"""Coefficient confidence intervals and collinearity diagnostics.

Split out of `predictive_modeling/common/data_utils.py` in stage D step 7
(2026-10-07) -- see `modeling/folds.py` for why.

Three CI methods live here and they are not interchangeable; `docs/findings.md`
records when each was adopted and what it moved. The short version: the
**clustered bootstrap** is the default for the paper's figures because it refits
the actual estimator per resample (so the L2 penalty and class weights are
respected), and the **cluster-robust Wald sandwich** is ~560x faster and used
inside cross-validation. Wald warns when `n_clusters <= n_params`, which is the
condition that made it untrustworthy on KnowQA's six participants.
"""

import warnings
from typing import Sequence, List, Optional, Mapping, Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.config import columns as Con

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


# https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
def wald_logreg_coef_cis(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    ci: float = 0.95,

    include_intercept: bool = False,
    use_pinv: bool = True,
    cluster: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    to read on this: https://www.jakemanderson.com/courses/econ_104/chapters/15c-cluster-robust-se
        lmiratrix.github.io/MLM/cluster_demo.html
        https://economictheoryblog.com/2016/09/25/clustered-standard-errors/
        


    Wald CIs for sklearn LogisticRegression coefficients, as a sandwich estimator.

    ``cluster`` (e.g. ``participant_id``) gives cluster-robust CR1 intervals;
    ``None`` treats each row as its own cluster, which is the heteroskedasticity-
    robust (HC0) limit of the same formula.

    **Rewritten 2026-09-27 (`todo.md` T3.3).** The previous version inverted the
    plain information matrix ``pinv(X'WX)``, which was wrong three ways: it ignored
    the L2 penalty although the fit is penalised, it ignored
    ``class_weight="balanced"``, and it had no notion of clustering at all
    (``n_clusters`` was hardcoded NaN). All three enter here:

    * **penalty** -- sklearn minimises ``0.5 b'b + C * sum_i s_i * loss_i`` for
      l2, so the bread is ``H = P + C * X' S W X`` with ``P = diag(0, 1, ..., 1)``
      (the intercept is not penalised). Dropping ``P`` is what made the old SEs
      unpenalised-MLE SEs for penalised estimates.
    * **class weights** ``s_i`` enter both the bread and the score.
    * **clustering** is the meat: scores are summed *within* cluster before the
      outer product, so within-participant correlation stops being ignored.

    The penalty is carried in the bread only -- it is not a sum over observations,
    so it does not decompose per cluster. That is the conventional penalised-
    M-estimator sandwich, and it is an approximation.

    Validated against the clustered bootstrap on L1's 12-feature model: mean CI
    width 1.44x Wald-old for both, agreeing to ~2% per feature. Use the bootstrap
    where cost allows (it refits the real estimator and needs no approximation);
    this exists because cross-validation cannot afford 5,000 refits per fold.

    **Few clusters:** CR1's ``G/(G-1)`` correction is applied, but it does not
    rescue very small ``G``. The cluster-robust variance estimator has rank at
    most ``min(G, k)``, so with ``G <= k`` it is singular and ``pinv`` hides that
    -- see the warning below. Check ``n_clusters`` in the output; KnowQA has 6
    against 13 parameters, which is why it uses the bootstrap instead.

    Reading, in the order worth reading it:

    * Cameron & Miller (2015), "A Practitioner's Guide to Cluster-Robust
      Inference", *J. Human Resources* 50(2):317-372. The standard guide; §2-§3
      are the bread/meat derivation and the CR1 correction, §VI the few-clusters
      problem. Free PDF at cameron.econ.ucdavis.edu/research/papers.html
    * Zeileis (2006), "Object-Oriented Computation of Sandwich Estimators",
      *J. Statistical Software* 16(9). Where the bread/meat naming used here
      comes from, and the clearest short statement of the general form.
    * MacKinnon, Nielsen & Webb (2023), "Cluster-Robust Inference: A Guide to
      Empirical Practice", *J. Econometrics* 232(2):272-299 (arXiv:2205.03285).
      Modern; states the ``rank <= min(G, k)`` limit and what few clusters do.
    * Freedman (2006), "On the so-called Huber sandwich estimator and robust
      standard errors", *The American Statistician* 60(4):299-302. Four pages
      arguing the sandwich is often the wrong thing to reach for -- worth reading
      precisely because it is the counter-case to everything above.
    """
    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float).reshape(-1)

    n, p = Xn.shape

    p_hat = model.predict_proba(X)[:, 1]

    # Class weights, exactly as the fit used them.
    if getattr(model, "class_weight", None) is None:
        sw = np.ones(n, dtype=float)
    else:
        sw = compute_sample_weight(model.class_weight, yn)

    w = sw * p_hat * (1.0 - p_hat)

    X_design = np.hstack([np.ones((n, 1)), Xn])
    k = X_design.shape[1]

    # Bread: penalised, weighted Hessian of sklearn's objective.
    C_inv_scale = float(getattr(model, "C", 1.0))
    Xw = X_design * np.sqrt(w)[:, None]
    H = C_inv_scale * (Xw.T @ Xw)
    if getattr(model, "penalty", "l2") == "l2":
        P = np.eye(k)
        P[0, 0] = 0.0          # the intercept is not penalised
        H = H + P

    inv = np.linalg.pinv if use_pinv else np.linalg.inv
    H_inv = inv(H)

    theta = np.concatenate([model.intercept_.reshape(-1), model.coef_.reshape(-1)])

    # Meat: per-observation scores, summed within cluster.
    u = (C_inv_scale * sw * (yn - p_hat))[:, None] * X_design

    if cluster is None:
        n_clusters = np.nan
        meat = u.T @ u
        correction = 1.0
    else:
        cl = np.asarray(cluster).reshape(-1)
        uniq = pd.unique(cl)
        n_clusters = int(len(uniq))
        sums = np.vstack([u[cl == c].sum(axis=0) for c in uniq])
        meat = sums.T @ sums
        # CR1 small-sample correction.
        correction = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / max(n - k, 1))

        # The meat is a sum of G outer products, so its rank is at most G. With
        # G <= k it cannot support k parameters: pinv absorbs the deficiency
        # silently and returns intervals that look ordinary and are not -- on
        # KnowQA (G = 6, k = 13) some come out 3x NARROWER than the clustered
        # bootstrap. Warn rather than raise: the setting is deliberate for now
        # (Diana, 2026-09-27), but it must not be invisible. See todo.md T3.3.
        if n_clusters <= k:
            warnings.warn(
                f"Cluster-robust Wald CIs with {n_clusters} clusters for {k} "
                f"parameters: the meat matrix has rank <= {n_clusters} and cannot "
                f"support {k} parameters, so these intervals are not "
                f"trustworthy -- some will be far too narrow. Cluster-robust "
                f"inference needs more clusters than parameters (and in practice "
                f"well over ~30). Prefer ci_method='bootstrap', or do not report "
                f"coefficient significance at this n.",
                RuntimeWarning,
                stacklevel=2,
            )

    cov = correction * (H_inv @ meat @ H_inv)

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
