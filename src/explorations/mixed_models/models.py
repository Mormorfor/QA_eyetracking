"""Mixed-effects backends for the correctness outcome: Julia MixedModels and R lme4.

**Parked** -- `research-context.md` section 6: *"mixed effects have their own problems and
were judged not important enough to pursue for this paper."* The reported model is the
logistic regression in `modeling/models/logreg_model.py`.

Moved out of `modeling/models/` in stage E so the paper's run path imports no Julia.
`run_model_bundles` used to open with a warning that importing it *"pulls in the Julia
backend, so it needs a working juliacall toolchain even when only the logistic
regression is wanted"* -- that is no longer true."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from juliacall import Main as jl
import src.config.columns as Con
from typing import Optional, Sequence, Dict, List
import polars as pl
from pymer4.models import glmer
from src.config import columns as Con


# ==========================================================================
# from src/modeling/models/julia_model.py
# ==========================================================================

@dataclass
class TrialLevelJuliaGLMERModel:
    """
    Binomial mixed-effects model using Julia MixedModels.jl.

    Expected input:
      - a ready trial-level dataframe
      - explicit feature_cols passed from the outside

    Random intercepts/slopes:
      - participant
      - text
    """
    name: str = "full_features_correctness_julia_glmer"
    fill_value: float = 0.0

    model: object = field(default=None, init=False)
    scaler_: Optional[StandardScaler] = field(default=None, init=False)

    feature_cols_raw_: List[str] = field(default_factory=list, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    formula_: Optional[str] = field(default=None, init=False)

    rename_map_: Dict[str, str] = field(default_factory=dict, init=False)
    reverse_rename_map_: Dict[str, str] = field(default_factory=dict, init=False)

    target_col_model_: Optional[str] = field(default=None, init=False)
    participant_col_model_: Optional[str] = field(default=None, init=False)
    text_col_model_: Optional[str] = field(default=None, init=False)

    participant_effects_mode: str = "slopes"  # "intercept" | "slopes"
    text_effects_mode: str = "slopes"  # "intercept" | "slopes"

    _julia_ready: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Julia setup
    # ------------------------------------------------------------------
    def _setup_julia(self) -> None:
        if self._julia_ready:
            return

        jl.seval("using DataFrames")
        jl.seval("using StatsModels")
        jl.seval("using MixedModels")
        jl.seval("using Distributions")
        jl.seval("using PythonCall")
        jl.seval("using StatsBase")

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
        function table_to_py(x)
            PythonCall.Compat.pytable(x)
        end
        """)

        jl.seval("""
        function fit_glmm_binomial(formula_obj, df; wcol=nothing)
            if isnothing(wcol)
                return fit(MixedModel, formula_obj, df, Bernoulli(); progress=false)
            else
                w = Vector{Float64}(df[!, Symbol(wcol)])
                return fit(MixedModel, formula_obj, df, Bernoulli(), wts=w; progress=false)
            end
        end
        """)

        jl.seval("""
        function predict_glmm_prob(model, newdf; use_rfx=false)
            if use_rfx
                return Vector{Float64}(predict(model, newdf; type=:response))
            else
                return Vector{Float64}(predict(model, newdf; type=:response, new_re_levels=:population))
            end
        end
        """)

        jl.seval("""
        function fixef_table(model)
            return DataFrame(
                feature = String.(fixefnames(model)),
                coef = collect(fixef(model)),
            )
        end
        """)

        jl.seval("""
        function ranef_tables_dict(model)
            tabs = raneftables(model)
            out = Dict{String,Any}()
            for (k, v) in pairs(tabs)
                out[string(k)] = DataFrame(v)
            end
            return out
        end
        """)

        jl.seval("""
        function varcorr_text(model)
            io = IOBuffer()
            show(io, MIME"text/plain"(), VarCorr(model))
            return String(take!(io))
        end
        """)

        self._julia_ready = True

    # ------------------------------------------------------------------
    # Prepare model dataframe
    # ------------------------------------------------------------------
    def _prepare_model_df(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = False,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> pd.DataFrame:
        if feature_cols is None:
            raw_cols = list(self.feature_cols_raw_)
        else:
            raw_cols = list(feature_cols)

        model_df = df[[target_col, participant_col, text_col] + raw_cols].copy()

        for c in raw_cols:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
        model_df[raw_cols] = model_df[raw_cols].fillna(self.fill_value)

        model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce").fillna(0).astype(int)
        model_df[participant_col] = model_df[participant_col].astype(str)
        model_df[text_col] = model_df[text_col].astype(str)

        if fit:
            self.scaler_ = StandardScaler()
            model_df[raw_cols] = self.scaler_.fit_transform(model_df[raw_cols])
            self.feature_cols_raw_ = list(raw_cols)
        else:
            model_df[raw_cols] = self.scaler_.transform(model_df[raw_cols])

        self.rename_map_ = {c: c for c in [target_col, participant_col, text_col] + raw_cols}
        self.reverse_rename_map_ = dict(self.rename_map_)

        self.target_col_model_ = target_col
        self.participant_col_model_ = participant_col
        self.text_col_model_ = text_col

        self.feature_cols_ = list(raw_cols)

        return model_df

    # ------------------------------------------------------------------
    # Formula
    # ------------------------------------------------------------------
    def _build_formula(self) -> str:
        fixed = " + ".join(self.feature_cols_)

        terms = [f"{self.target_col_model_} ~ 1 + {fixed}"]

        if self.participant_effects_mode == "intercept":
            terms.append(f"(1 | {self.participant_col_model_})")
        elif self.participant_effects_mode == "slopes":
            terms.append(
                f"zerocorr(1 + {fixed} | {self.participant_col_model_})"
            )

        if self.text_effects_mode == "intercept":
            terms.append(f"(1 | {self.text_col_model_})")
        elif self.text_effects_mode == "slopes":
            terms.append(
                f"zerocorr(1 + {fixed} | {self.text_col_model_})"
            )

        formula = " + ".join(terms)

        self.formula_ = formula
        print(f"Built formula: {formula}")
        return formula

    # ------------------------------------------------------------------
    # Pandas -> Julia DataFrame
    # ------------------------------------------------------------------
    def _to_julia_df(self, df: pd.DataFrame):
        self._setup_julia()

        tmp = df.copy()

        for c in tmp.columns:
            if c in self.feature_cols_:
                tmp[c] = tmp[c].astype(float)
            elif c == self.target_col_model_:
                tmp[c] = tmp[c].astype(int)
            elif c == self.participant_col_model_:
                tmp[c] = tmp[c].astype(str)
            elif c == self.text_col_model_:
                tmp[c] = tmp[c].astype(str)
            elif c == "obs_weight":
                tmp[c] = tmp[c].astype(float)

        colnames = list(tmp.columns)
        cols = [tmp[c].tolist() for c in colnames]

        jl.colnames_py = colnames
        jl.cols_py = cols
        return jl.seval("pycols_to_df(colnames_py, cols_py)")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = Con.IS_CORRECT_COLUMN,
        feature_cols: Optional[Sequence[str]] = None,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> None:
        model_df = self._prepare_model_df(
            train_df,
            fit=True,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        formula = self._build_formula()

        counts = model_df[self.target_col_model_].value_counts()
        w0 = 1.0 / counts[0]
        w1 = 1.0 / counts[1]
        model_df["obs_weight"] = np.where(model_df[self.target_col_model_] == 0, w0, w1)
        model_df["obs_weight"] = model_df["obs_weight"] / model_df["obs_weight"].mean()

        j_df = self._to_julia_df(model_df)

        jl.j_df_train = j_df
        jl.seval(f"j_formula = @formula({formula})")

        self.model = jl.seval(
            'fit_glmm_binomial(j_formula, j_df_train; wcol="obs_weight")'
        )
        # self.model = jl.seval(
        #     'fit_glmm_binomial(j_formula, j_df_train; wcol=nothing)'
        # )

    # ------------------------------------------------------------------
    # Predict probabilities
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        df: pd.DataFrame,
        target_col: str = Con.IS_CORRECT_COLUMN,
        feature_cols: Optional[Sequence[str]] = None,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
        use_rfx: bool = False,
    ) -> np.ndarray:
        model_df = self._prepare_model_df(
            df,
            fit=False,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        j_new = self._to_julia_df(model_df)
        jl.model_py = self.model
        jl.j_new = j_new

        preds = jl.seval(
            f"predict_glmm_prob(model_py, j_new; use_rfx={str(use_rfx).lower()})"
        )
        return np.asarray(preds).reshape(-1).astype(float)

    # ------------------------------------------------------------------
    # Predict classes
    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        p = self.predict_proba(df, **kwargs)
        return (p >= threshold).astype(int)

    def get_coef_summary(
            self,
            train_df: Optional[pd.DataFrame] = None,
            top_k: Optional[int] = None,
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
            target_col: str = Con.IS_CORRECT_COLUMN,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        jl.model_py = self.model
        coef_tbl = jl.table_to_py(jl.seval("coeftable(model_py)"))
        out = pd.DataFrame(coef_tbl)

        if "Name" in out.columns:
            out = out.rename(columns={"Name": "feature"})
        elif "Coef." in out.columns:
            out = out.reset_index().rename(columns={"index": "feature"})
        else:
            out = out.reset_index().rename(columns={"index": "feature"})

        out["feature_raw"] = out["feature"]

        if "Coef." in out.columns:
            out["coef"] = pd.to_numeric(out["Coef."], errors="coerce")
        elif "Estimate" in out.columns:
            out["coef"] = pd.to_numeric(out["Estimate"], errors="coerce")
        else:
            out["coef"] = np.nan

        if "Std. Error" in out.columns:
            se = pd.to_numeric(out["Std. Error"], errors="coerce")
            out["se"] = se
            out["ci_low"] = out["coef"] - 1.96 * se
            out["ci_high"] = out["coef"] + 1.96 * se
        else:
            out["se"] = np.nan
            out["ci_low"] = np.nan
            out["ci_high"] = np.nan

        out["abs_coef"] = out["coef"].abs()
        out["or"] = np.exp(out["coef"])
        out["or_ci_low"] = np.exp(out["ci_low"])
        out["or_ci_high"] = np.exp(out["ci_high"])

        if "Pr(>|z|)" in out.columns:
            pvals = pd.to_numeric(out["Pr(>|z|)"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        elif "p" in out.columns:
            pvals = pd.to_numeric(out["p"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        else:
            out["sig_ci"] = (
                    pd.notna(out["ci_low"])
                    & pd.notna(out["ci_high"])
                    & ((out["ci_low"] > 0) | (out["ci_high"] < 0))
            )

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))

        return out.reset_index(drop=True)



    def get_formula(self) -> str:
        return self.formula_

    def get_fixef_table(self) -> pd.DataFrame:
        jl.model_py = self.model
        tbl = jl.table_to_py(jl.seval("fixef_table(model_py)"))
        out = pd.DataFrame(tbl)

        out["feature_raw"] = out["feature"]
        out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
        out["or"] = np.exp(out["coef"])
        out["abs_coef"] = out["coef"].abs()

        return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    def get_random_effects(self) -> Dict[str, pd.DataFrame]:
        jl.model_py = self.model
        out = jl.seval("ranef_tables_dict(model_py)")

        py_out: Dict[str, pd.DataFrame] = {}
        for group_name, table_obj in out.items():
            df = pd.DataFrame(jl.table_to_py(table_obj)).copy()

            df = df.rename(columns={
                "(Intercept)": "random_intercept",
                "Intercept": "random_intercept",
            })

            cols = list(df.columns)
            if cols:
                first = cols[0]
                other_cols = [c for c in cols if c != first]
                df = df[[first] + other_cols]

            py_out[str(group_name)] = df

        return py_out

    def get_random_effect_variance_summary(self) -> str:
        jl.model_py = self.model
        return str(jl.seval("varcorr_text(model_py)"))


# ==========================================================================
# from src/modeling/models/glmer_r_model.py
# ==========================================================================

@dataclass
class TrialLevelGLMERModel:
    """
    Binomial mixed-effects model on an already prepared trial-level dataframe.

    Assumptions:
    - df already contains the trial-level feature columns
    - target_col exists and is binary
    - participant_col and text_col exist
    - feature_cols are passed explicitly on first fit
    """
    name: str = "trial_level_glmer"
    fill_value: float = 0.0
    optimizer_control: str = (
        "glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000))"
    )

    model: object = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)

    raw_feature_cols_: List[str] = field(default_factory=list, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    formula_: Optional[str] = field(default=None, init=False)

    rename_map_: Dict[str, str] = field(default_factory=dict, init=False)
    reverse_rename_map_: Dict[str, str] = field(default_factory=dict, init=False)

    target_col_model_: Optional[str] = field(default=None, init=False)
    participant_col_model_: Optional[str] = field(default=None, init=False)
    text_col_model_: Optional[str] = field(default=None, init=False)

    def _validate_feature_cols(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> list[str]:
        cols = list(feature_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        return cols

    @staticmethod
    def _sanitize_colname(col: str) -> str:
        out = str(col)
        out = out.replace("__", "_")
        out = out.replace("-", "_")
        out = out.replace(" ", "_")
        out = out.replace("(", "_").replace(")", "_")
        out = out.replace("/", "_")
        out = out.replace("\\", "_")
        out = out.replace(".", "_")
        out = out.replace(":", "_")
        return out

    @staticmethod
    def _to_model_df(df: pd.DataFrame) -> pl.DataFrame:
        return pl.from_pandas(df.reset_index(drop=True))

    def _prepare_model_df(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> pd.DataFrame:
        if feature_cols is None:
            if not self.raw_feature_cols_:
                raise ValueError("feature_cols must be provided on first fit.")
            raw_cols = list(self.raw_feature_cols_)
        else:
            raw_cols = self._validate_feature_cols(df, feature_cols)

        needed = [target_col, participant_col, text_col] + list(raw_cols)
        model_df = df[needed].copy()

        for c in raw_cols:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
        model_df[raw_cols] = model_df[raw_cols].fillna(self.fill_value)

        model_df[target_col] = pd.to_numeric(
            model_df[target_col], errors="coerce"
        ).astype(int)
        model_df[participant_col] = model_df[participant_col].astype(str)
        model_df[text_col] = model_df[text_col].astype(str)

        if fit:
            self.raw_feature_cols_ = list(raw_cols)
            self.scaler_ = StandardScaler()
            model_df[raw_cols] = self.scaler_.fit_transform(model_df[raw_cols])
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            model_df[raw_cols] = self.scaler_.transform(model_df[raw_cols])

        rename_map = {
            c: self._sanitize_colname(c)
            for c in [target_col, participant_col, text_col] + list(raw_cols)
        }

        model_df = model_df.rename(columns=rename_map)

        self.rename_map_ = dict(rename_map)
        self.reverse_rename_map_ = {v: k for k, v in rename_map.items()}

        self.target_col_model_ = rename_map[target_col]
        self.participant_col_model_ = rename_map[participant_col]
        self.text_col_model_ = rename_map[text_col]
        self.feature_cols_ = [rename_map[c] for c in raw_cols]

        return model_df

    def _build_formula(self) -> str:
        if not self.feature_cols_:
            raise RuntimeError("No fitted feature columns available.")

        fixed = " + ".join(self.feature_cols_)

        formula = (
            f"{self.target_col_model_} ~ {fixed} "
            f"+ (1|{self.participant_col_model_}) "
            f"+ (1|{self.text_col_model_})"
        )
        self.formula_ = formula
        return formula

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = Con.IS_CORRECT_COLUMN,
        feature_cols: Optional[Sequence[str]] = None,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> None:
        model_df = self._prepare_model_df(
            train_df,
            fit=True,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        formula = self._build_formula()

        counts = model_df[self.target_col_model_].value_counts()
        if not {0, 1}.issubset(set(counts.index)):
            raise ValueError(
                "Target column must contain both classes 0 and 1 in training data."
            )

        w0 = 1.0 / counts[0]
        w1 = 1.0 / counts[1]
        model_df["obs_weight"] = np.where(
            model_df[self.target_col_model_] == 0,
            w0,
            w1,
        )
        model_df["obs_weight"] = model_df["obs_weight"] / model_df["obs_weight"].mean()

        self.model = glmer(
            formula=formula,
            data=self._to_model_df(model_df),
            family="binomial",
        )

        self.model.fit(
            exponentiate=False,
            summary=False,
            conf_method="wald",
            type_predict="response",
            control=self.optimizer_control,
            weights="obs_weight",
        )

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
        use_rfx: bool = False,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        model_df = self._prepare_model_df(
            df,
            fit=False,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        preds = self.model.predict(
            data=self._to_model_df(model_df),
            use_rfx=use_rfx,
            type_predict="response",
        )

        return np.asarray(preds).reshape(-1).astype(float)

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        p = self.predict_proba(df, feature_cols=feature_cols, **kwargs)
        return (p >= threshold).astype(int)

    def get_coef_summary(
            self,
            train_df: Optional[pd.DataFrame] = None,
            top_k: Optional[int] = None,
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
            target_col: str = Con.IS_CORRECT_COLUMN,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if hasattr(self.model, "result_fit") and self.model.result_fit is not None:
            out = self.model.result_fit.to_pandas()
        elif hasattr(self.model, "params") and self.model.params is not None:
            out = self.model.params.to_pandas()
        else:
            raise RuntimeError("Could not find coefficient table on fitted pymer4 model.")

        if "term" in out.columns:
            out = out.rename(columns={"term": "feature"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "feature"})
        else:
            out = out.reset_index().rename(columns={"index": "feature"})

        out["feature_raw"] = out["feature"].map(self.reverse_rename_map_).fillna(out["feature"])

        if "estimate" in out.columns:
            out["coef"] = pd.to_numeric(out["estimate"], errors="coerce")
        elif "Estimate" in out.columns:
            out["coef"] = pd.to_numeric(out["Estimate"], errors="coerce")
        else:
            out["coef"] = np.nan

        out["abs_coef"] = out["coef"].abs()

        if "conf_low" in out.columns and "conf_high" in out.columns:
            out["ci_low"] = pd.to_numeric(out["conf_low"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["conf_high"], errors="coerce")
        elif "lower_CL" in out.columns and "upper_CL" in out.columns:
            out["ci_low"] = pd.to_numeric(out["lower_CL"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["upper_CL"], errors="coerce")
        elif "2.5_ci" in out.columns and "97.5_ci" in out.columns:
            out["ci_low"] = pd.to_numeric(out["2.5_ci"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["97.5_ci"], errors="coerce")
        else:
            out["ci_low"] = np.nan
            out["ci_high"] = np.nan

        out["or"] = np.exp(out["coef"])
        out["or_ci_low"] = np.exp(out["ci_low"])
        out["or_ci_high"] = np.exp(out["ci_high"])

        if "p_value" in out.columns:
            pvals = pd.to_numeric(out["p_value"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        elif "P-val" in out.columns:
            pvals = pd.to_numeric(out["P-val"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        else:
            out["sig_ci"] = (
                    pd.notna(out["ci_low"])
                    & pd.notna(out["ci_high"])
                    & ((out["ci_low"] > 0) | (out["ci_high"] < 0))
            )

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))

        return out.reset_index(drop=True)

    def get_formula(self) -> str:
        if self.formula_ is None:
            raise RuntimeError("Model formula is not available before fitting.")
        return self.formula_
