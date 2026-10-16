"""Mixed models behind the attention-allocation significance heatmaps.

Among trials where the participant chose a given answer, do the four answers
differ on a per-area measurement (dwell time, skip rate, pupil size, ...)?
`mixed_area_analysis` answers this for one measurement and one set of trials: a
mixed model (a regression with random intercepts for participant and trial) gives
each answer's mean, and every pair of answers is tested, Holm-corrected.
`run_all_area_mixed_models` repeats it for every measurement, chosen answer and
group, and saves a significance heatmap (drawn in `plots.py`) with its tables.

This only compares the answers with each other. Predicting correctness with a
mixed model is a separate thing, kept in `explorations/mixed_models/`.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from src.lib.plotting.output import collect_tables, save_output
from src.analyses.attention_allocation.plots import plot_pairwise_significance_heatmap
from src.config import columns as Con
from typing import Dict
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# One measurement, one set of trials
# ---------------------------------------------------------------------------

def mixed_area_analysis(
    df: pd.DataFrame,
    stat_col: str = Con.MEAN_DWELL_TIME,
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
    area_col: str = Con.AREA_LABEL_COLUMN,
    alpha: float = 0.05,
) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """Fit one measurement on the four answer areas, and test every pair of answers.

    `df` holds the trials to compare (usually those where one answer was chosen).
    It may be the word-level table: it is reduced to one row per trial and area
    first, since the measurement is the same on every word of an area. The model:

        stat_col ~ 0 + area,  with a random intercept per participant
                              and one per text within each participant

    statsmodels cannot cross random effects, so the text intercept is estimated
    separately inside each participant. A participant reads each text once, so in
    effect it is an intercept per trial: it allows for a trial's four answers being
    alike, but it is not a text effect shared across participants.

    Only the four answers are compared: a question row, if present, is dropped, and
    so is any area with no value on `stat_col` (an answer nobody looked at has no
    first-fixation duration or pupil size), without a message.

    Returns (result, fe_table, pairwise):
      result    the fitted statsmodels model
      fe_table  each answer's estimate, with a (1 - alpha) confidence interval
      pairwise  one row per pair of answers: the difference (first minus second),
                its standard error, t and p, the Holm-adjusted p over this fit's
                pairs (`p_adj_holm`), and "★" in `sig` where that is below `alpha`
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

    # patsy cannot read pandas' string dtype: `C(area_label)` raises a TypeError
    # before any model is fitted. So the three columns the formula and the grouping
    # use are cast to `object`. This changes the dtype only, not the model.
    for _col in (area_col, Con.PARTICIPANT_ID, Con.TEXT_ID_COLUMN):
        dedup[_col] = dedup[_col].astype(object)

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
    save: Optional[bool] = None,
    to_paper=None,
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
) -> Dict[str, Dict[str, dict]]:
    """Fit `mixed_area_analysis` for every measurement and chosen answer in one
    group, and save a significance heatmap with its two tables for each fit.

    The question's rows are removed first, and each fit uses only the trials where
    that answer (A, B, C or D) was chosen. `metrics` defaults to the model's list
    of per-area measurements (pupil size z-scored). Each fit's tables are printed,
    and its heatmap is saved with `fixed_effects` and `pairwise` through
    `save_output`, under reports/attention_allocation/ in a subfolder per
    measurement.

    Returns results[metric][answer] = {"model", "fe_table", "pairwise"}.
    """

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


            fig = plot_pairwise_significance_heatmap(
                pairwise=pairwise,
                title="{} — {} — selected={}\n(-log10 Holm-adjusted p)".format(
                    group_name, metric, ans
                ),
                alpha=alpha,
                areas=("answer_A", "answer_B", "answer_C", "answer_D"),
                show=False,
            )

            print("\nFixed effects (per area):")
            print(fe_table.to_string(index=False))

            print("\nPairwise comparisons (Holm-corrected):")
            print(pairwise.sort_values("p_adj_holm").to_string(index=False))

            save_output(
                fig,
                analysis="attention_allocation",
                plot="area_pairwise_significance",
                tables={"fixed_effects": fe_table, "pairwise": pairwise},
                save=save,
                to_paper=to_paper,
                subdir=metric,
                group=group_name,
                metric=metric,
                selected=ans,
            )
            if fig is not None:
                plt.close(fig)

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
    save: Optional[bool] = None,
    to_paper=None,
    trial_cols=("participant_id", "text_id", "TRIAL_INDEX"),
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Run `run_models_for_group` for hunters, for gatherers, and for both together.

    Unlike the other runners, this takes the two groups' frames separately. The
    tables of all the fits are pooled into one long table each (fixed effects,
    pairwise), with the group, measurement and chosen answer as columns.

    Returns {"hunters": ..., "gatherers": ..., "all_participants": ...}, each as
    `run_models_for_group` returns it.
    """
    # Up to 120 figures: 3 groups x 10 measurements x 4 chosen answers.
    with collect_tables("attention_allocation", plots=["area_pairwise_significance"]):
        hunters_res = run_models_for_group(
            hunters,
            group_name="hunters",
            metrics=metrics,
            alpha=alpha,
            save=save,
            to_paper=to_paper,
            trial_cols=trial_cols,
        )

        gatherers_res = run_models_for_group(
            gatherers,
            group_name="gatherers",
            metrics=metrics,
            alpha=alpha,
            save=save,
            to_paper=to_paper,
            trial_cols=trial_cols,
        )
        all_participants = pd.concat([hunters, gatherers], ignore_index=True)
        all_participants_res = run_models_for_group(
            all_participants,
            group_name="all_participants",
            metrics=metrics,
            alpha=alpha,
            save=save,
            to_paper=to_paper,
            trial_cols=trial_cols,
        )

    return {
        "hunters": hunters_res,
        "gatherers": gatherers_res,
        "all_participants": all_participants_res,
    }
