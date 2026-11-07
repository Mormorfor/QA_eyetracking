"""Mixed-effects logistic regression on one row per trial, fitted in Julia.

The paper's logistic regression (`logreg_model.py`) with a random intercept per
participant and per item added. It has the logistic regression's `fit`, `predict`,
`predict_proba` and `get_coef_summary`, so the cross-validation, the single-split and
new-data runs, the model comparison and the Study 2 analysis all take it through their
`model_builder` argument.

Importing this module starts Julia, so it needs the Julia packages that
`src/vendor/julia_integration/julia_test.ipynb` installs. Nothing else in `modeling/`
imports it, so the logistic regression runs without Julia.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Literal

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import columns as Con
from src.modeling.models.logreg_model import TrialLevelLogRegModel

# juliacall keeps its Julia packages in a project folder that it picks from the *activated*
# conda env (CONDA_PREFIX). Run this env's python without activating it, or from a Jupyter
# started in another env, and it picks a different folder: it then installs a fresh Julia
# there, which takes minutes, and fails with "Package DataFrames not found". Pinning the
# folder to the running interpreter's env makes the choice follow the Python actually in use.
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path(sys.prefix) / "julia_env"))
from juliacall import Main as jl  # noqa: E402  -- must come after the line above


# ==========================================================================
# Julia side: loaded once per process
# ==========================================================================

_JULIA_READY = False


def _setup_julia() -> None:
    """Load the Julia packages and define the few helpers the model calls."""
    global _JULIA_READY
    if _JULIA_READY:
        return

    jl.seval("using DataFrames, StatsModels, MixedModels, Distributions, StatsBase, PythonCall")

    # Columns are passed one at a time as plain Python lists. Handing Julia the pandas
    # frame itself does not work reliably (`vendor/julia_integration/README.md`).
    jl.seval("""
    function pycols_to_df(colnames, cols)
        d = DataFrame()
        for (nm, col) in zip(colnames, cols)
            d[!, Symbol(nm)] = pyconvert(Vector, col)
        end
        return d
    end
    """)

    jl.seval("""
    function fit_bernoulli_glmm(f, df, wts)
        w = pyconvert(Vector{Float64}, wts)   # empty = unweighted
        return fit(MixedModel, f, df, Bernoulli(); wts=w, progress=false)
    end
    """)

    # One {column name => values} dict per grouping factor: the level, then one column
    # per random term ("(Intercept)", and a feature name per random slope).
    jl.seval("""
    function ranef_columns(m)
        out = Dict{String, Any}()
        for (g, tbl) in pairs(raneftables(m))
            d = DataFrame(tbl)
            out[string(g)] = Dict(n => collect(d[!, n]) for n in names(d))
        end
        return out
    end
    """)

    jl.seval("""
    function variance_components(m)
        groups = String[]; terms = String[]; sds = Float64[]
        for (g, nt) in pairs(m.σs)
            for (t, s) in pairs(nt)
                push!(groups, string(g)); push!(terms, string(t)); push!(sds, s)
            end
        end
        return Dict("group" => groups, "term" => terms, "sd" => sds)
    end
    """)

    jl.seval("""
    function model_text(x)
        io = IOBuffer()
        show(io, MIME"text/plain"(), x)
        return String(take!(io))
    end
    """)

    _JULIA_READY = True


# NLopt's return codes that mean the optimiser stopped because it converged. Anything else
# (MAXEVAL_REACHED, MAXTIME_REACHED, FAILURE, ...) means it stopped for another reason.
_CONVERGED_RETURNVALUES = {"SUCCESS", "FTOL_REACHED", "XTOL_REACHED", "STOPVAL_REACHED"}

# A name pasted into `@formula(...)` must be a plain Julia identifier; anything else would
# be parsed as an expression rather than a column.
_JULIA_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class TrialLevelJuliaGLMERModel:
    """Mixed-effects logistic regression predicting `is_correct`, fitted in Julia.

    The paper's logistic regression (`TrialLevelLogRegModel`) with a random intercept per
    participant and per text added:

        is_correct ~ 1 + <features> + (1 | participant_id) + (1 | text_id_with_q)

    **What is the same as the logistic regression, by construction:**

    * the feature matrix is prepared by the logistic regression's own code
      (`TrialLevelLogRegModel._prepare_X`, called on an instance held in `prep_`): numeric
      columns, EVERY NaN filled with `fill_value` and counted (`imputed_counts_`,
      `imputed_mask_`), a warning for NaN outside the explained families, and a
      `StandardScaler` fitted on the training rows. So coefficients are per standard
      deviation of each feature, as there;
    * `class_weight="balanced"` gives the same per-trial weights sklearn uses
      (`compute_sample_weight`), so correct and wrong answers count equally;
    * `predict` is `predict_proba > 0.5`, which is sklearn's rule.

    **What is not:**

    * no L2 penalty. The logistic regression has sklearn's default (C = 1); on ~19,000
      trials and 12 standardised features that penalty is small, but it is not zero;
    * the class weights enter a mixed-model likelihood as prior weights. That shifts the
      intercept as intended, and also changes the estimated variance of the random
      intercepts and the model-based standard errors: neither is the likelihood estimate
      of the unweighted model. Fit with `class_weight=None` to see the unweighted values.

    **The text level** is `text_col`, by default `text_id_with_q`: a paragraph together
    with the question asked about it, the test item (L1: 972 items, about 20 trials each).

    **Prediction on held-out trials** (`predict_with`):

    * `"fixed"` -- fixed effects only: every random intercept set to 0, so a trial is
      scored as if its participant and its text were average. The same score for a seen
      and an unseen participant; the closest analogue of the logistic regression;
    * `"known_levels"` -- a participant's or text's estimated random effect where that
      level was in the training data, 0 where it was not. MixedModels'
      `new_re_levels=:population` rule, computed here from the fitted tables (MixedModels'
      own `predict` insists on a response column even for new data).

    With a new participant reading a new text the two are identical.

    Coefficient intervals are Wald intervals from the model's own standard errors: they
    account for trials being grouped by participant and by text through the random
    intercepts, not through a sandwich or a bootstrap as for the logistic regression.
    """

    name: str = "trial_level_julia_glmm"
    participant_col: str = Con.PARTICIPANT_ID
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN
    class_weight: Optional[str] = "balanced"
    # Put in place of EVERY NaN in the feature matrix, as in the logistic regression.
    fill_value: float = 0.0
    predict_with: Literal["fixed", "known_levels"] = "fixed"
    # "intercept": (1 | g). "slopes": an uncorrelated random slope for every feature as
    # well, zerocorr(1 + <features> | g). "none": no random effect for that grouping.
    # **"slopes" does not work in MixedModels (up to 5.9 at least).** Next to a second
    # grouping factor it builds part of the model's matrix without the weights, so with
    # the class weights the model cannot be set up at all (PosDefException), and without
    # them every fitting step goes wrong the same way and is very slow.
    participant_effects_mode: Literal["intercept", "slopes", "none"] = "intercept"
    text_effects_mode: Literal["intercept", "slopes", "none"] = "intercept"

    model: object = field(default=None, init=False)
    prep_: Optional[TrialLevelLogRegModel] = field(default=None, init=False)
    target_col_: Optional[str] = field(default=None, init=False)
    formula_: Optional[str] = field(default=None, init=False)
    intercept_: Optional[float] = field(default=None, init=False)
    coef_: Optional[pd.Series] = field(default=None, init=False)        # per feature
    stderr_: Optional[pd.Series] = field(default=None, init=False)      # incl. "(Intercept)"
    ranef_: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    optimizer_returnvalue_: Optional[str] = field(default=None, init=False)
    converged_: Optional[bool] = field(default=None, init=False)
    is_singular_: Optional[bool] = field(default=None, init=False)
    n_train_: Optional[int] = field(default=None, init=False)

    # The name of the folder its outputs are saved in, and of `model_family` in its tables
    # (`modeling/provenance.py`). The two scoring rules give different predictions from the
    # same fit, so each has its own: "glmm" for fixed effects only, "glmm_known" for known
    # random intercepts.
    @property
    def family(self) -> str:
        return {"fixed": "glmm", "known_levels": "glmm_known"}[self.predict_with]

    # The preparation's bookkeeping lives on `prep_`; these make it readable from here, under
    # the same names as on the logistic regression.
    @property
    def feature_cols_(self) -> List[str]:
        return [] if self.prep_ is None else list(self.prep_.feature_cols_)

    @property
    def scaler_(self) -> Optional[StandardScaler]:
        return None if self.prep_ is None else self.prep_.scaler_

    @property
    def imputed_counts_(self) -> Optional[pd.Series]:
        return None if self.prep_ is None else self.prep_.imputed_counts_

    @property
    def imputed_mask_(self) -> Optional[pd.DataFrame]:
        return None if self.prep_ is None else self.prep_.imputed_mask_

    @property
    def unexpected_imputed_(self) -> Optional[pd.Series]:
        return None if self.prep_ is None else self.prep_.unexpected_imputed_

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """The two grouping columns as text. A missing id stops the fit: as text it would
        become a level called "nan", pooling every trial with no id into one participant."""
        groups = df[[self.participant_col, self.text_col]]
        n_missing = groups.isna().sum()
        if n_missing.any():
            raise ValueError(
                f"Missing grouping ids: {n_missing[n_missing > 0].to_dict()}. Every trial "
                f"needs a participant and a text for the random intercepts."
            )
        return groups.astype(str)

    def _to_julia_df(self, y: np.ndarray, groups: pd.DataFrame, X: pd.DataFrame):
        names = [self.target_col_, self.participant_col, self.text_col] + list(X.columns)
        cols = (
            [y.astype(int).tolist()]
            + [groups[c].tolist() for c in (self.participant_col, self.text_col)]
            + [X[c].astype(float).tolist() for c in X.columns]
        )
        return jl.pycols_to_df(names, cols)

    # ------------------------------------------------------------------
    # Formula
    # ------------------------------------------------------------------
    def _random_term(self, mode: str, group: str, fixed: str) -> Optional[str]:
        if mode == "intercept":
            return f"(1 | {group})"
        if mode == "slopes":
            return f"zerocorr(1 + {fixed} | {group})"
        if mode == "none":
            return None
        raise ValueError(f"Unknown random-effects mode {mode!r}.")

    def _build_formula(self) -> str:
        names = [self.target_col_, self.participant_col, self.text_col] + self.feature_cols_
        bad = [n for n in names if not _JULIA_IDENTIFIER.match(n)]
        if bad:
            raise ValueError(f"Not usable as names in a Julia @formula: {bad}")

        fixed = " + ".join(self.feature_cols_)
        terms = [f"{self.target_col_} ~ 1 + {fixed}"]
        for mode, group in (
            (self.participant_effects_mode, self.participant_col),
            (self.text_effects_mode, self.text_col),
        ):
            term = self._random_term(mode, group, fixed)
            if term is not None:
                terms.append(term)
        return " + ".join(terms)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Sequence[str],
    ) -> None:
        _setup_julia()

        self.prep_ = TrialLevelLogRegModel(fill_value=self.fill_value)
        X = self.prep_._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = train_df[target_col].astype(int).to_numpy()
        groups = self._groups(train_df)
        self.target_col_ = target_col
        self.formula_ = self._build_formula()

        weights = (
            compute_sample_weight(self.class_weight, y).tolist()
            if self.class_weight is not None
            else []
        )

        j_df = self._to_julia_df(y, groups, X)
        j_formula = jl.seval(f"@formula({self.formula_})")
        self.model = jl.fit_bernoulli_glmm(j_formula, j_df, weights)
        self.n_train_ = len(y)

        self._read_fit()

    def _read_fit(self) -> None:
        """Copy what prediction and the summaries need out of the Julia model."""
        m = self.model
        names = [str(n) for n in jl.seval("m -> String.(fixefnames(m))")(m)]
        beta = np.asarray(jl.seval("m -> collect(fixef(m))")(m), dtype=float)
        se = np.asarray(jl.seval("m -> collect(stderror(m))")(m), dtype=float)

        # MixedModels drops a column it finds collinear with the others, and fixef then
        # leaves it out. The model would no longer be the one asked for, so stop.
        expected = ["(Intercept)"] + self.feature_cols_
        if names != expected:
            raise RuntimeError(
                f"Fixed effects came back as {names}, expected {expected}. A missing name "
                f"was dropped by MixedModels as collinear with the other columns."
            )

        self.intercept_ = float(beta[0])
        self.coef_ = pd.Series(beta[1:], index=self.feature_cols_)
        self.stderr_ = pd.Series(se, index=names)

        self.ranef_ = {}
        for group, cols in dict(jl.ranef_columns(m)).items():
            cols = dict(cols)
            frame = pd.DataFrame({
                str(k): (
                    [str(v) for v in cols[k]] if str(k) == group
                    else np.asarray(cols[k], dtype=float)
                )
                for k in cols
            })
            self.ranef_[str(group)] = frame.set_index(group)

        self.optimizer_returnvalue_ = str(jl.seval("m -> string(m.optsum.returnvalue)")(m))
        self.converged_ = self.optimizer_returnvalue_ in _CONVERGED_RETURNVALUES
        self.is_singular_ = bool(jl.seval("issingular")(m))

        if not self.converged_:
            warnings.warn(
                f"{self.name}: the optimiser stopped with {self.optimizer_returnvalue_}, "
                f"not a convergence code. The estimates may not be the maximum.",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.is_singular_:
            warnings.warn(
                f"{self.name}: singular fit -- at least one random-effect variance is "
                f"estimated at (or next to) 0. See get_variance_components().",
                RuntimeWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def _linear_predictor(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self.prep_._prepare_X(df, fit=False, feature_cols=feature_cols)
        eta = self.intercept_ + X[self.feature_cols_].to_numpy() @ self.coef_.to_numpy()

        if self.predict_with == "fixed":
            return eta
        if self.predict_with != "known_levels":
            raise ValueError(f"Unknown predict_with {self.predict_with!r}.")

        # Each grouping factor adds its estimated effects for levels it has seen and
        # nothing for new ones. "(Intercept)" multiplies 1; a random slope multiplies
        # its feature, on the same standardised scale as the fixed effect.
        groups = self._groups(df)
        design = X.assign(**{"(Intercept)": 1.0})
        for group, table in self.ranef_.items():
            effects = table.reindex(groups[group].to_numpy()).fillna(0.0)
            eta = eta + (effects.to_numpy() * design[list(table.columns)].to_numpy()).sum(axis=1)
        return eta

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        return expit(self._linear_predictor(df, feature_cols))

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        # Strictly greater, as sklearn's LogisticRegression.predict.
        return (self.predict_proba(df, feature_cols=feature_cols) > 0.5).astype(int)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def get_coef_summary(
        self,
        train_df: Optional[pd.DataFrame] = None,
        top_k: Optional[int] = None,
        ci_method: Literal["wald", "none"] = "wald",
        ci_cluster: Literal["cluster"] = "cluster",
        ci: float = 0.95,
        n_boot: int = 5000,
        seed: int = 42,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
    ) -> pd.DataFrame:
        """Fixed-effect coefficients with Wald intervals, in the logistic regression's
        columns (feature, coef, odds_ratio, abs_coef, se, ci_low, ci_high, or_ci_low,
        or_ci_high, sig_ci) plus `z` and `p_value`. Largest |coef| first, intercept left
        out as there.

        The signature is the logistic regression's, so `crossval.py` and `evaluate.py`
        can call either model. `train_df`, `feature_cols`, `target_col`, `n_boot` and
        `seed` are not needed and not used: the standard errors come with the fit.

        `ci_cluster` must be "cluster": the grouping of trials is in the model itself, as
        the random intercepts, and there is no version of these intervals that treats
        trials as independent. `ci_method="bootstrap"` is not available: MixedModels'
        parametric bootstrap is for linear mixed models only, and a participant bootstrap
        would mean thousands of mixed-model fits.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if ci_cluster != "cluster":
            raise ValueError(
                f"ci_cluster={ci_cluster!r}: a mixed model's intervals already account for "
                f"the grouping, through its random effects. Use 'cluster'."
            )
        if ci_method not in ("wald", "none"):
            raise NotImplementedError(
                f"ci_method={ci_method!r} is not available for the mixed model; use 'wald'."
            )

        out = pd.DataFrame({
            "feature": self.feature_cols_,
            "coef": self.coef_.to_numpy(),
        })
        out["odds_ratio"] = np.exp(out["coef"])
        out["abs_coef"] = out["coef"].abs()

        if ci_method == "wald":
            z = norm.ppf(1 - (1 - float(ci)) / 2)
            se = self.stderr_[self.feature_cols_].to_numpy()
            out["se"] = se
            out["z"] = out["coef"] / se
            out["p_value"] = 2 * norm.sf(np.abs(out["z"]))
            out["ci_low"] = out["coef"] - z * se
            out["ci_high"] = out["coef"] + z * se
            out["or_ci_low"] = np.exp(out["ci_low"])
            out["or_ci_high"] = np.exp(out["ci_high"])
            out["sig_ci"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)

        out = out.sort_values("abs_coef", ascending=False)
        if top_k is not None:
            out = out.head(int(top_k))
        return out.reset_index(drop=True)

    def get_formula(self) -> str:
        return self.formula_

    def get_random_effects(self) -> Dict[str, pd.DataFrame]:
        """Per grouping factor, one row per level: the level, then `random_intercept` (and
        a column per random slope). Keyed by the grouping column's name."""
        return {
            group: table.rename(columns={"(Intercept)": "random_intercept"}).reset_index()
            for group, table in self.ranef_.items()
        }

    def get_variance_components(self) -> pd.DataFrame:
        """The estimated standard deviation of each random effect, one row per group and term."""
        vc = dict(jl.variance_components(self.model))
        return pd.DataFrame({
            "group": [str(g) for g in vc["group"]],
            "term": [str(t) for t in vc["term"]],
            "sd": np.asarray(vc["sd"], dtype=float),
        }).assign(variance=lambda d: d["sd"] ** 2)

    def get_random_effect_variance_summary(self) -> str:
        """MixedModels' own printout of the variance components."""
        return str(jl.model_text(jl.seval("VarCorr")(self.model)))

    def summary_text(self) -> str:
        """MixedModels' full printout of the fitted model."""
        return str(jl.model_text(self.model))
