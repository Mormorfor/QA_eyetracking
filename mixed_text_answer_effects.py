"""
mixed_text_answer_effects.py

Mixed-effects models linking text-level dwell times
(critical / distractor / outside spans) to answer-level dwell times.

Uses:
    - merged_hunters.csv
    - merged_gatherers.csv

These are produced by answers_paragraphs_csv.py and are expected to contain:
    - C.TRIAL_ID
    - C.PARTICIPANT_ID
    - C.TEXT_ID_WITH_Q_COLUMN
    - C.SELECTED_ANSWER_LABEL_COLUMN  ('A','B','C','D')
    - Answer dwell columns:           'answer_A', 'answer_B', ...
    - Text-span dwell columns:        'critical', 'distractor', 'outside'

Two model “families” are implemented:

1) No-slopes models:
    answer_X ~ critical + distractor + outside
               + (1 | participant_id)
               + (1 | text_id)

   fitted either:
   - on all trials (`*_all`)
   - on the subset where answer X was selected (`*_separated`)

2) Random-slope models:
    answer_X ~ critical + distractor + outside
               + (1 + critical + distractor + outside || participant_id)
               + (1 | text_id)

   and a variant with fully correlated slopes:
    answer_X ~ critical + distractor + outside
               + (1 + critical + distractor + outside | participant_id)
               + (1 | text_id)

The main goal:
    - estimate fixed effects of text spans on answer dwell times,
    - inspect participant-specific random effects (slopes).

All plotting and modelling utilities live here and can be called
from a notebook (mixed_text_answer_effects.ipynb).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pymer4.models import Lmer

import constants as C


# Silence specific FutureWarnings from pymer4
warnings.filterwarnings(
    "ignore",
    message=".*DataFrame.applymap.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Text-span predictors (columns in merged_* CSVs)
TERMS = ["critical", "distractor", "outside"]

# Colours per span
TERM_COLORS = {
    "critical": "#1f77b4",
    "distractor": "#ff7f0e",
    "outside": "#2ca02c",
}

# Markers per answer label
ANSWER_MARKERS = {
    "A": "o",
    "B": "s",
    "C": "^",
    "D": "D",
}

TEXT_GROUP_ID = C.TEXT_ID_WITH_Q_COLUMN
# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_merged_data(
    hunters_path: str = "output_data/merged_hunters.csv",
    gatherers_path: str = "output_data/merged_gatherers.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load merged hunters & gatherers CSVs produced by answers_paragraphs_csv.py.

    Ensures participant_id and text_id are strings (required by pymer4 / Lmer).

    Returns
    -------
    merged_h : DataFrame
        Hunters (question_preview == True).
    merged_g : DataFrame
        Gatherers (question_preview == False).
    """
    print(f"Loading hunters data from: {hunters_path}")
    merged_h = pd.read_csv(hunters_path)

    print(f"Loading gatherers data from: {gatherers_path}")
    merged_g = pd.read_csv(gatherers_path)


    return merged_h, merged_g


def _answer_col(label: str) -> str:
    """
    Build the answer dwell-time column name from an answer label ('A','B',...).
    """
    return f"{C.ANSWER_PREFIX}{label}"



# ---------------------------------------------------------------------------
# Mixed models without slopes
# ---------------------------------------------------------------------------

def calc_corr_mixed_answ_separated(
    df: pd.DataFrame,
    answer_label: str,
    terms: list[str] = TERMS,
) -> tuple[Lmer, pd.DataFrame]:
    """
    Fit a no-slopes mixed model for ONE answer (on trials where it was selected):

        answer_col ~ critical + distractor + outside
                     + (1 | participant_id)
                     + (1 | text_id)

    Parameters
    ----------
    df : DataFrame
        Merged hunters/gatherers data.
    answer_label : str
        'A', 'B', 'C', or 'D'.
    terms : list of str
        Column names of text-span predictors.

    Returns
    -------
    model : Lmer
    result : DataFrame
        pymer4 fit summary.
    """
    answer_col = _answer_col(answer_label)

    df_sub = df[df[C.SELECTED_ANSWER_LABEL_COLUMN] == answer_label].copy()

    predictors = " + ".join(terms)
    formula = (
        f"{answer_col} ~ {predictors} "
        f"+ (1|{C.PARTICIPANT_ID}) + (1|{TEXT_GROUP_ID})"
    )
    print(f"Fitting model (separated by answer selected) for answer {answer_label}: {formula}")
    model = Lmer(formula, data=df_sub)
    result = model.fit()
    return model, result


def calc_corr_mixed_all_answ(
    df: pd.DataFrame,
    answer_label: str,
    terms: list[str] = TERMS,
) -> tuple[Lmer, pd.DataFrame]:
    """
    Fit a no-slopes mixed model for ONE answer on ALL trials:

        answer_col ~ critical + distractor + outside
                     + (1 | participant_id)
                     + (1 | text_id)
    """
    answer_col = _answer_col(answer_label)

    predictors = " + ".join(terms)
    formula = (
        f"{answer_col} ~ {predictors} "
        f"+ (1|{C.PARTICIPANT_ID}) + (1|{TEXT_GROUP_ID})"
    )
    print(f"Fitting model (all) for answer {answer_label}: {formula}")
    model = Lmer(formula, data=df)
    result = model.fit()
    return model, result


def fit_no_slopes_for_all_answers(
    df: pd.DataFrame,
    terms: list[str] = TERMS,
    separated: bool = True,
) -> dict[str, Lmer]:
    """
    Fit no-slopes models for ALL answers (A–D) on the given DataFrame.

    Parameters
    ----------
    df : DataFrame
        merged_h or merged_g.
    terms : list of str
        Text-span columns.
    separated : bool
        If True, use only trials where answer was selected.
        If False, use all trials.

    Returns
    -------
    models : dict
        { "A": model_A, "B": model_B, ... }
    """
    models = {}
    for label in C.ANSWER_LABELS:
        if separated:
            m, _ = calc_corr_mixed_answ_separated(df, label, terms)
        else:
            m, _ = calc_corr_mixed_all_answ(df, label, terms)
        models[label] = m
    return models


def plot_fixed_effects(
    terms: list[str],
    models: dict[str, Lmer],
    h_or_g: str = "hunters",
    separated_label: str = "answ_separated",
    output_dir: str = "plots/texts_to_answers/fixed_effects",
):
    """
    Plot fixed effects (with 95% CIs) for each term and each answer model.

    Parameters
    ----------
    terms : list of str
        Text-span columns (e.g. ["critical","distractor","outside"]).
    models : dict[str, Lmer]
        {"A": model_A, ...}.
    h_or_g : {"hunters","gatherers"}
        Label for figure title & filename.
    separated_label : str
        Used in filename: 'answ_separated' | 'all_answ', etc.
    output_dir : str
        Base directory to save plots.
    """
    effects = pd.concat(
        [
            m.coefs.loc[terms, ["Estimate", "SE"]]
            .assign(
                answer=a,
                term=lambda d: d.index,
                ci_low=lambda d: d["Estimate"] - 1.96 * d["SE"],
                ci_high=lambda d: d["Estimate"] + 1.96 * d["SE"],
            )
            for a, m in models.items()
        ],
        ignore_index=True,
    )

    x = np.arange(len(terms))
    answers = C.ANSWER_LABELS
    offset = np.linspace(-0.3, 0.3, len(answers))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    for i, a in enumerate(answers):
        sub = effects[effects["answer"] == a].set_index("term").loc[terms]
        for j, t in enumerate(terms):
            est = sub.loc[t, "Estimate"]
            lo = sub.loc[t, "ci_low"]
            hi = sub.loc[t, "ci_high"]
            ax.errorbar(
                x[j] + offset[i],
                est,
                yerr=[[est - lo], [hi - est]],
                fmt=ANSWER_MARKERS[a],
                ms=6,
                mfc=TERM_COLORS.get(t, "gray"),
                mec="black",
                mew=0.6,
                ecolor=TERM_COLORS.get(t, "gray"),
                elinewidth=1,
                capsize=3,
            )

    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.capitalize() for t in terms]
    )
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Fixed effects by span — {h_or_g} ({separated_label})")

    answer_handles = [
        Line2D(
            [],
            [],
            marker=ANSWER_MARKERS[a],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label=a,
        )
        for a in answers
    ]
    term_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=TERM_COLORS.get(t, "gray"),
            markeredgecolor="black",
            label=t.capitalize(),
        )
        for t in terms
    ]

    leg1 = ax.legend(
        handles=answer_handles,
        title="Answers",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=term_handles,
        title="Text segment",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(
        output_dir,
        f"fixed_effects_by_span_{h_or_g}_{separated_label}.png",
    )
    plt.savefig(fname)
    plt.show()




# ---------------------------------------------------------------------------
# Mixed models with slopes
# ---------------------------------------------------------------------------

def build_model_with_slopes(
    df: pd.DataFrame,
    answer_label: str,
    terms: list[str] = TERMS,
    separated: bool = True,
    correlated_slopes: bool = False,
) -> Lmer:
    """
    Build a random-slope model for ONE answer.

    If separated=True:
        filter to trials where that answer was selected.
    If correlated_slopes=True:
        (1 + terms | participant_id)
    else:
        (1 + terms || participant_id)

    Formula:
        answer_col ~ critical + distractor + outside
                     + (1 + ... || participant_id)
                     + (1 | text_id)
    """
    answer_col = _answer_col(answer_label)

    if separated:
        df = df[df[C.SELECTED_ANSWER_LABEL_COLUMN] == answer_label].copy()

    predictors = " + ".join(terms)
    slope_part = " + ".join(terms)
    slope_bar = "|" if correlated_slopes else "||"

    formula = (
        f"{answer_col} ~ {predictors} "
        f"+ (1 + {slope_part} {slope_bar} {C.PARTICIPANT_ID}) "
        f"+ (1 | {TEXT_GROUP_ID})"
    )
    print(f"Fitting random-slope model for answer {answer_label}: {formula}")
    m = Lmer(formula, data=df)
    m.fit()
    return m




def participant_effects(model: Lmer, terms: list[str] = TERMS) -> pd.DataFrame:
    """
    Extract participant-specific random intercepts and slopes,
    and add fixed effects to get participant-level total effects.

    Returns
    -------
    eff : DataFrame
        Columns:
            participant_id
            rand_intercept
            <term>_dev  (random slope deviation)
            <term>      (random + fixed)
    """

    re = model.ranef[1].copy()
    if "(Intercept)" in re.columns:
        re = re.rename(columns={"(Intercept)": "Intercept"})
    re = re.reset_index().rename(columns={"index": C.PARTICIPANT_ID})

    fix = model.coefs.loc[terms, "Estimate"]
    out = re[[C.PARTICIPANT_ID, "Intercept"]].rename(
        columns={"Intercept": "rand_intercept"}
    )
    for t in terms:
        out[f"{t}_dev"] = re[t]
        out[t] = re[t] + fix[t]
    return out



def _signif_code(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    elif p < 1e-1:
        return "."
    else:
        return ""



def plot_participant_effects(
    model: Lmer,
    answer_label: str,
    terms: list[str] = TERMS,
    h_or_g: str = "hunters",
    separated: bool = True,
    correlated_slopes: bool = False,
    output_dir: str = "plots/texts_to_answers/participant_effects",
    data_output_dir: str = "output_data/slopes",
):
    """
    Plot participant-specific random intercepts and slopes for one answer model,
    and save a fixed-effects summary table.

    Folder structure:

        plots:  {output_dir}/{sep_label}__{corr_label}/...
        tables: {data_output_dir}/{sep_label}__{corr_label}/...

    where:
        sep_label  = 'separated' or 'all_trials'
        corr_label = 'correlated' or 'uncorrelated'
    """
    # ------------------------------------------------------------------
    # Resolve condition labels / subfolders
    # ------------------------------------------------------------------
    sep_label = "separated" if separated else "all_trials"
    corr_label = "correlated" if correlated_slopes else "uncorrelated"

    plot_dir = os.path.join(output_dir, f"{sep_label}__{corr_label}")
    table_dir = os.path.join(data_output_dir, f"{sep_label}__{corr_label}")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Fixed effects summary
    # ------------------------------------------------------------------
    eff = participant_effects(model, terms)
    coefs = model.coefs.loc[terms, ["Estimate", "SE", "P-val"]].copy()

    # Add CI and significance markers, and keep p-values
    coefs["CI_low"] = coefs["Estimate"] - 1.96 * coefs["SE"]
    coefs["CI_high"] = coefs["Estimate"] + 1.96 * coefs["SE"]

    def _stars(p):
        if p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 5e-2:
            return "*"
        elif p < 1e-1:
            return "."
        else:
            return ""

    coefs["sig"] = coefs["P-val"].apply(_stars)
    coefs = coefs.reset_index().rename(columns={"index": "term"})

    print(f"\n=== {answer_label}: fixed effects ({sep_label}, {corr_label}) ===")
    for _, row in coefs.iterrows():
        t = row["term"]
        p = row["P-val"]
        print(
            f"{t:12s} β={row['Estimate']:.3f} ± {1.96*row['SE']:.3f}  "
            f"p={p:.4g} {row['sig']}"
        )

    # Save fixed-effects table as CSV
    fe_path = os.path.join(
        table_dir,
        f"fixed_effects_{answer_label}_{h_or_g}.csv",
    )
    coefs.to_csv(fe_path, index=False)
    print(f"Saved fixed-effects table to: {fe_path}")

    # ------------------------------------------------------------------
    # Plot random intercepts
    # ------------------------------------------------------------------
    d = eff["rand_intercept"].sort_values().reset_index(drop=True)
    x = np.arange(len(d))
    plt.figure(figsize=(12, 3), dpi=150)
    plt.errorbar(x, d, yerr=0, fmt="none")
    plt.vlines(x, 0, d, color="black", alpha=0.5, lw=1)
    plt.scatter(x, d, color="black", s=10)
    plt.axhline(0, color="black", lw=1)
    plt.title(
        f"Random intercepts — {answer_label} "
        f"({h_or_g}, {sep_label}, {corr_label})"
    )
    plt.xlabel("Participant (sorted)")
    plt.ylabel("Random intercept")
    fname_int = os.path.join(
        plot_dir,
        f"{answer_label}_intercept_{h_or_g}.png",
    )
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(fname_int)
    print(f"Saved intercept plot to: {fname_int}")
    plt.show()

    # ------------------------------------------------------------------
    # Plot participant-specific slopes per term
    # ------------------------------------------------------------------
    for t in terms:
        sd = eff[f"{t}_dev"].std(ddof=1)
        d = eff[t].sort_values().reset_index(drop=True)
        x = np.arange(len(d))
        yerr = np.full_like(d.values, 1.96 * sd, dtype=float)

        plt.figure(figsize=(12, 3), dpi=150)
        plt.errorbar(
            x,
            d,
            yerr=yerr,
            fmt="none",
            ecolor=TERM_COLORS.get(t, "gray"),
            alpha=0.7,
            capsize=2,
            lw=1,
            zorder=1,
        )
        plt.vlines(
            x,
            0,
            d,
            color=TERM_COLORS.get(t, "gray"),
            alpha=0.4,
            lw=1,
            zorder=2,
        )
        plt.scatter(
            x,
            d,
            color=TERM_COLORS.get(t, "gray"),
            s=12,
            zorder=3,
        )

        plt.axhline(0, color="black", lw=1)
        plt.axhline(d.mean(), color="gray", ls="--", lw=1, label="Mean")
        plt.axhline(
            np.median(d),
            color="gray",
            ls="-.",
            lw=1,
            label="Median",
        )
        plt.title(
            f"Participant-specific effect of {t.capitalize()} — {answer_label} "
            f"({h_or_g}, {sep_label}, {corr_label})"
        )
        plt.xlabel("Participant (sorted)")
        plt.ylabel("Effect (ms per unit)")
        plt.grid(axis="y", alpha=0.2)
        plt.legend(loc="upper left")
        plt.tight_layout()

        fname_term = os.path.join(
            plot_dir,
            f"{answer_label}_{t}_{h_or_g}.png",
        )
        plt.savefig(fname_term)
        print(f"Saved slope plot for {t} to: {fname_term}")
        plt.show()




# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all(
    hunters_path: str = "output_data/merged_hunters.csv",
    gatherers_path: str = "output_data/merged_gatherers.csv",
):
    """
    Convenience entry point that:
        1) loads merged hunters/gatherers,
        2) fits no-slope models (separated & all),
        3) plots fixed effects,
        4) fits random-slope models (separated & all),
        5) plots participant-specific effects (separated & all).

    """
    merged_h, merged_g = load_merged_data(hunters_path, gatherers_path)

    # --- No slopes: separated ---
    models_h_sep = fit_no_slopes_for_all_answers(merged_h, TERMS, separated=True)
    models_g_sep = fit_no_slopes_for_all_answers(merged_g, TERMS, separated=True)

    plot_fixed_effects(TERMS, models_h_sep, "hunters", "answ_separated")
    plot_fixed_effects(TERMS, models_g_sep, "gatherers", "answ_separated")

    # --- No slopes: all trials ---
    models_h_all = fit_no_slopes_for_all_answers(merged_h, TERMS, separated=False)
    models_g_all = fit_no_slopes_for_all_answers(merged_g, TERMS, separated=False)

    plot_fixed_effects(TERMS, models_h_all, "hunters", "all_answ")
    plot_fixed_effects(TERMS, models_g_all, "gatherers", "all_answ")

    # --- Slopes: separated ---
    for label in C.ANSWER_LABELS:
        m_h = build_model_with_slopes(
            merged_h, label, TERMS, separated=True, correlated_slopes=False
        )
        plot_participant_effects(m_h, f"Answer {label}", TERMS, "hunters")

        m_g = build_model_with_slopes(
            merged_g, label, TERMS, separated=True, correlated_slopes=False
        )
        plot_participant_effects(m_g, f"Answer {label}", TERMS, "gatherers")

    # --- Slopes: all trials (correlated) ---
    for label in C.ANSWER_LABELS:
        m_h_all = build_model_with_slopes(
            merged_h, label, TERMS, separated=False, correlated_slopes=True
        )
        plot_participant_effects(
            m_h_all, f"Answer {label} (all)", TERMS, "hunters"
        )

        m_g_all = build_model_with_slopes(
            merged_g, label, TERMS, separated=False, correlated_slopes=True
        )
        plot_participant_effects(
            m_g_all, f"Answer {label} (all)", TERMS, "gatherers"
        )


