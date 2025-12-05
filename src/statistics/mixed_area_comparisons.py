from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src import constants as Con



## Would be ideal to transition to LMER, but I struggle with pairwise comparisons there.
## So for now we keep using statsmodels MixedLM, even thought text ids are not treated exactly correctly
## (or maybe its even better this way?)
def mixed_area_analysis(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
    area_col: str = Con.AREA_LABEL_COLUMN,
    alpha: float = 0.05,
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Mixed-effects model: area effect on a given metric, with random intercepts
    for participants and texts.

    Model:
        stat_col ~ 0 + C(area_col)
        random intercept for participant_id
        variance component for text_id
        (≈ stat ~ 0 + area + (1|participant) + (1|text))

    Returns
    -------
    result : statsmodels MixedLMResults
    fe_table : DataFrame
        Per-area fixed effect estimates and CIs.
    pairwise : DataFrame
        Pairwise comparisons (Holm-adjusted p-values).
    """

    dedup = (
        df[list(trial_cols) + [area_col, stat_col]]
        .drop_duplicates()
        .copy()
    )

    area_order = [
        a for a in ["answer_A", "answer_B", "answer_C", "answer_D"]
        if a in dedup[area_col].unique()
    ]
    if not area_order:
        area_order = sorted(dedup[area_col].unique())

    dedup[area_col] = pd.Categorical(
        dedup[area_col],
        categories=area_order,
        ordered=True,
    )

    formula = f"{stat_col} ~ 0 + C({area_col})"
    model = smf.mixedlm(
        formula,
        data=dedup,
        groups=dedup[Con.PARTICIPANT_ID],
        re_formula="1",
        vc_formula={"text": "0 + C(text_id)"},
    )
    result = model.fit(method="lbfgs", reml=True)

    fe = result.fe_params.copy()
    ci = result.conf_int(alpha=alpha).loc[fe.index]
    fe_idx = fe.index.tolist()

    def pname(level: str) -> str:
        for nm in fe_idx:
            if nm.endswith(f"[{level}]") or nm.endswith(f"[T.{level}]"):
                return nm

    fe_table = (
        pd.DataFrame(
            {
                "term": fe.index,
                "estimate": fe.values,
                "ci_low": ci[0].values,
                "ci_high": ci[1].values,
            }
        )
        .assign(
            area=lambda d: d["term"].str.extract(
                r"\[(?:T\.)?([^\]]+)\]"
            )
        )
        .set_index("area")
        .loc[area_order]
        .reset_index()
    )

    # Pairwise comparisons (Holm-adjusted)
    pairs = []
    k_fe = len(fe_idx)

    for i in range(len(area_order)):
        for j in range(i + 1, len(area_order)):
            a, b = area_order[i], area_order[j]
            L = np.zeros((1, k_fe))
            L[0, fe_idx.index(pname(a))] = 1.0
            L[0, fe_idx.index(pname(b))] = -1.0

            t_res = result.t_test(L)
            eff = float(np.asarray(t_res.effect).ravel()[0])
            se = float(np.asarray(t_res.sd).ravel()[0])
            tval = float(np.asarray(t_res.tvalue).ravel()[0])
            pval = float(np.asarray(t_res.pvalue).ravel()[0])
            ci_lo, ci_hi = np.asarray(
                t_res.conf_int(alpha=alpha)
            ).ravel()

            pairs.append(
                {
                    "area_i": a,
                    "area_j": b,
                    "diff_i_minus_j": eff,
                    "se": se,
                    "t": tval,
                    "p_unc": pval,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                }
            )

    pairwise = pd.DataFrame(pairs)
    pairwise["p_adj_holm"] = multipletests(
        pairwise["p_unc"], method="holm"
    )[1]
    pairwise["sig"] = np.where(
        pairwise["p_adj_holm"] < alpha, "★", ""
    )

    return result, fe_table, pairwise