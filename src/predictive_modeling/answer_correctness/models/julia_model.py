from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from juliacall import Main as jl

import src.constants as Con


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