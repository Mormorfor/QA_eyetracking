"""Confidence intervals for logistic-regression coefficients, and a collinearity check.

Two ways to get an interval, each with or without clustering by participant:

* **Bootstrap** (`bootstrap_logreg_coef_cis`): refit the model on thousands of
  resamples -- of whole participants, when clustered. It needs no
  approximation, so it is the default for the paper's coefficients.
* **Wald** (`wald_logreg_coef_cis`): a sandwich formula, no refitting. Hundreds
  of times faster, so it is used inside cross-validation. It warns when there
  are no more participants than parameters, as on Study 2 (6 participants).

`compute_vif` measures how much each feature is explained by the others.
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
    Coefficient table for a fitted LogisticRegression: coefficient, odds ratio
    and absolute coefficient, largest first. ``top_k`` keeps only the largest.
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
    Wald confidence intervals for a fitted sklearn LogisticRegression, from a
    sandwich (robust) variance estimate.

    ``cluster`` (e.g. participant ids) gives cluster-robust (CR1) intervals;
    ``None`` treats each row as its own cluster (the HC0 robust form of the
    same formula).

    What the formula includes:

    * **the L2 penalty** -- sklearn minimises ``0.5 b'b + C * sum_i s_i *
      loss_i``, so the bread is ``H = P + C * X' S W X`` with
      ``P = diag(0, 1, ..., 1)`` (the intercept is not penalised);
    * **the class weights** ``s_i``, in both the bread and the scores;
    * **clustering**, in the meat: scores are summed within each cluster before
      the outer product, so trials of the same person are not treated as
      independent.

    The penalty is in the bread only -- it is not a sum over observations, so
    it does not split by cluster. That is the usual sandwich for penalised
    estimators, and it is an approximation. The bootstrap needs none, so use it
    where time allows; this exists because cross-validation cannot afford
    thousands of refits per fold.

    **Few clusters:** the CR1 correction ``G/(G-1)`` is applied, but it does not
    rescue a very small ``G``. The meat has rank at most ``min(G, k)``, so with
    ``G <= k`` it cannot support ``k`` parameters, and ``pinv`` hides that --
    see the warning in the code. Check ``n_clusters`` in the output: Study 2
    has only 6 participants, fewer than the parameters of the headline model
    (11).

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
    * Shorter introductions:
      jakemanderson.com/courses/econ_104/chapters/15c-cluster-robust-se,
      lmiratrix.github.io/MLM/cluster_demo.html,
      economictheoryblog.com/2016/09/25/clustered-standard-errors/
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
        # Study 2 (G = 6) some come out about 3x narrower than the clustered
        # bootstrap. This warns rather than stops: using it here is a
        # deliberate choice, but it must never be invisible.
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
    Bootstrap confidence intervals for LogisticRegression coefficients: refit
    on ``n_boot`` resamples and take the percentiles.

    - ``cluster`` None: resample rows.
    - ``cluster`` given (e.g. participant ids): resample whole clusters with
      replacement, keeping all rows of each.

    A resample that contains only one class cannot be fitted and is skipped;
    ``n_boot_ok`` says how many were used.

    Returns feature, ci_low, ci_high, or_ci_low, or_ci_high, n_boot_ok, sig_ci.
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

    Columns not in df are skipped, with a printed note. Values are made numeric
    and missing ones filled with 0, as the model does. A constant column gets
    VIF = inf, noted "constant".
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

    ``bundle`` is the dict returned by
    ``analyses/correctness_prediction/run.py::run_full_features_correctness_bundle``.
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
    """For a random-effects table (first column: the level, e.g. the
    participant), the number of levels, mean, SD, min and max of every other
    column. Used by the mixed-model exploration."""
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
