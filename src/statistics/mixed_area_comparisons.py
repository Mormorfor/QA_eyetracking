from typing import Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.viz.visualisations_area_significance_heatmaps import plot_pairwise_significance_heatmap

import os

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

    dedup[stat_col] = pd.to_numeric(dedup[stat_col], errors="coerce")
    dedup = dedup.dropna(subset=[stat_col, area_col, Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN]).copy()
    dedup = dedup.reset_index(drop=True)

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



# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def run_models_for_group(
    df: pd.DataFrame,
    group_name: str,
    metrics=None,
    alpha: float = 0.05,
    save_tables: bool = False,
    tables_root: str = "../reports/report_data/area_mixed_models",
    graphs_root="../reports/plots/area_significance_heatmaps",
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
    save_heatmaps: bool = True,
) -> Dict[str, Dict[str, dict]]:

    if metrics is None:
        metrics = Con.AREA_METRIC_COLUMNS_MODELING

    df_noq = df[df[Con.AREA_LABEL_COLUMN] != "question"].copy()

    results: Dict[str, Dict[str, dict]] = {}

    for metric in metrics:
        metric_results: Dict[str, dict] = {}

        available_labels = [
            lab for lab in ["A", "B", "C", "D"]
            if lab in df_noq[Con.SELECTED_ANSWER_LABEL_COLUMN].unique()
        ]

        print(f"\n=== {group_name.upper()} — metric: {metric} ===")

        for ans in available_labels:
            subset = df_noq[df_noq[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans].copy()

            if subset.empty:
                print(f"  Skipping answer {ans}: no trials.")
                continue

            print(f"\n--- {group_name.upper()}, selected = {ans} ---")
            model_res, fe_table, pairwise = mixed_area_analysis(
                subset,
                stat_col=metric,
                trial_cols=trial_cols,
                area_col=Con.AREA_LABEL_COLUMN,
                alpha=alpha,
            )


            if save_heatmaps:
                out_dir = os.path.join(graphs_root, metric, group_name)
                out_path = os.path.join(
                    out_dir,
                    "{}__{}__sig_heatmap.png".format(group_name, ans),
                )
                plot_pairwise_significance_heatmap(
                    pairwise=pairwise,
                    title="{} — {} — selected={}\n(-log10 Holm-adjusted p)".format(
                        group_name, metric, ans
                    ),
                    alpha=alpha,
                    save_path=out_path,
                    areas=("answer_A", "answer_B", "answer_C", "answer_D"),
                    show=False,
                )

            print("\nFixed effects (per area):")
            print(fe_table.to_string(index=False))

            print("\nPairwise comparisons (Holm-corrected):")
            print(pairwise.sort_values("p_adj_holm").to_string(index=False))

            if save_tables:
                base_dir = os.path.join(tables_root, metric, group_name)
                os.makedirs(base_dir, exist_ok=True)

                fe_path = os.path.join(base_dir, "{}__{}__fe.csv".format(group_name, ans))
                pw_path = os.path.join(base_dir, "{}__{}__pairwise.csv".format(group_name, ans))

                fe_table.to_csv(fe_path, index=False)
                pairwise.to_csv(pw_path, index=False)

                print(f"\nSaved fixed-effects to:  {fe_path}")
                print(f"Saved pairwise table to: {pw_path}")

            metric_results[ans] = {
                "model": model_res,
                "fe_table": fe_table,
                "pairwise": pairwise,
            }

        results[metric] = metric_results

    return results



def run_all_area_mixed_models(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metrics=None,
    alpha: float = 0.05,
    save_tables: bool = False,
    tables_root: str = "../reports/report_data/area_mixed_models",
    graphs_root: str = "../reports/plots/area_significance_heatmaps",
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """
    Convenience wrapper to run models for both hunters and gatherers.

    Returns:
      {
        "hunters":   <results dict from run_models_for_group>,
        "gatherers": <results dict from run_models_for_group>,
      }
    """
    hunters_res = run_models_for_group(
        hunters,
        group_name="hunters",
        metrics=metrics,
        alpha=alpha,
        save_tables=save_tables,
        tables_root=tables_root,
        graphs_root=graphs_root,
        trial_cols=trial_cols,
    )

    gatherers_res = run_models_for_group(
        gatherers,
        group_name="gatherers",
        metrics=metrics,
        alpha=alpha,
        save_tables=save_tables,
        tables_root=tables_root,
        graphs_root=graphs_root,
        trial_cols=trial_cols,
    )
    all_participants = pd.concat([hunters, gatherers], ignore_index=True)
    all_participants_res = run_models_for_group(
        all_participants,
        group_name="all participants",
        metrics=metrics,
        alpha=alpha,
        save_tables=save_tables,
        tables_root=tables_root,
        graphs_root=graphs_root,
        trial_cols=trial_cols,
    )

    return {
        "hunters": hunters_res,
        "gatherers": gatherers_res,
        "all participants": all_participants_res,
    }