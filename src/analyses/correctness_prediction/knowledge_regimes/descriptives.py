"""Study 2 (KnowQA) described by knowledge regime, and the L1-trained correctness model
applied to it. The functions behind `notebooks/drivers/knowqa_data_showcase.ipynb`.

* **The data.** Who took part (`cohort_summary`), how often they were right
  (`correctness_by_regime`), how fast they went (`reading_speed_by_regime`) and how
  confident they said they were (`confidence_by_regime`), per regime; then the items,
  the sittings and the participants' English proficiency (LexTALE) behind those numbers.
* **The model.** The correctness model is fitted once on every L1 trial (`fit_l1_model`)
  and its prediction added to every KnowQA trial (`attach_predictions`). The rest are
  group-bys over those per-trial probabilities: how well the model does in each regime,
  what it says about lucky guesses, and whether it tells right from wrong better than the
  confidence rating. (`comparison_runs` instead scores the model on each regime as a
  separate run.)

Start from `load_showcase_frame`: the KnowQA feature table, with `regime`, `session_id`
and `confidence_rating` taken from the cleaned IA report.

Two things about the transfer that the numbers do not show on their own:

* **The scaler is L1's.** Every feature is standardised with L1's mean and SD, so a
  feature that runs higher in KnowQA shows up as shifted rather than rescaled.
* **Pupil baselines.** Pupil size is z-scored against each person's own baseline in both
  studies, but in KnowQA that baseline pools the person's three sessions. One of the
  model's features, `mean_max_fix_pupil_size_z__correct`, is such a z-score.

Each plot function returns its figure and, when `save` is on, also writes it, with the
table behind it, to `reports/correctness_prediction/knowledge_regimes/`. `plot_lextale`
only returns its figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.config.datasets import KNOW_QA_FEATURES_PATH, KNOW_QA_IA_ANSWERS_PATH, READY_ALL_FEATURES_PATH
from src.lib.stats.proportions import wilson_ci
from src.features.build import load_model_ready
from src.modeling.models.logreg_model import (
    TrialLevelLogRegModel,
    imputed_cell_counts,
)
import src.modeling.feature_sets as fg

# The regimes in order of increasing knowledge. Written out rather than read from the
# data, so every figure and table uses the same order.
REGIME_ORDER: tuple[str, ...] = ("no knowledge", "partial knowledge", "full knowledge")

REGIME_COL = "regime"
SESSION_COL = Con.SESSION_ID
CONFIDENCE_COL = "confidence_rating"

# Trial-level columns taken from the cleaned IA report. The feature table has `regime`
# and `session_id` but not the confidence rating; all three are read from the one place.
_META_COLS = (REGIME_COL, SESSION_COL, CONFIDENCE_COL)

# The paper's best cross-validated feature set: the ten hand-picked features of
# `SELECT_1_COLS`, plus two last-look features -- whether the last area looked at before
# the confirming press was the correct answer, and whether it was the question.
SHOWCASE_FEATURE_COLS: List[str] = list(fg.SELECT_1_COLS) + list(fg.LAST_CONFIRM_COMPACT)

PROB_COL = "pred_prob"
PRED_COL = "pred_is_correct"

DataFrameOrPath = Union[pd.DataFrame, str, Path]


def _resolve(frame_or_path: DataFrameOrPath) -> pd.DataFrame:
    """`frame_or_path` itself if it is a DataFrame, else the model-ready table at that path."""
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path
    return load_model_ready(frame_or_path)


def load_showcase_frame(
    features: DataFrameOrPath = KNOW_QA_FEATURES_PATH,
    ia_source: DataFrameOrPath = KNOW_QA_IA_ANSWERS_PATH,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """The KnowQA feature table, with `regime`, `session_id` and `confidence_rating` from
    the IA report.

    The three are matched on `(participant_id, TRIAL_INDEX)`, and any copies already in
    the feature table are replaced. The IA report also holds the practice trials; the
    feature table does not, so they drop out of the join.

    Stops if a trial gets no regime: it would otherwise drop out of every per-regime
    table while still counting in the pooled ones.
    """
    feats = _resolve(features)

    if isinstance(ia_source, pd.DataFrame):
        ia = ia_source[[Con.PARTICIPANT_ID, Con.TRIAL_ID, *_META_COLS]].copy()
    else:
        ia = pd.read_csv(
            ia_source,
            low_memory=False,
            usecols=[Con.PARTICIPANT_ID, Con.TRIAL_ID, *_META_COLS],
            dtype={Con.PARTICIPANT_ID: str},
        )

    # Stripped and lower-cased as in `comparison_runs.attach_regime`, so the two modules'
    # regime labels match.
    ia[Con.PARTICIPANT_ID] = ia[Con.PARTICIPANT_ID].astype(str).str.strip().str.lower()
    ia[REGIME_COL] = ia[REGIME_COL].astype(str).str.strip().str.lower()

    meta = ia.drop_duplicates(subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID])

    # Drop the feature table's own copies first: merging a column that is already there
    # would leave two copies, `_x` and `_y`, and no column under the plain name.
    feats = feats.drop(columns=list(_META_COLS), errors="ignore")

    out = feats.merge(meta, on=[Con.PARTICIPANT_ID, Con.TRIAL_ID], how="left")

    assert len(out) == len(feats), (
        f"regime merge changed the row count: {len(feats)} -> {len(out)}; "
        "the IA metadata is not unique per (participant_id, TRIAL_INDEX)"
    )
    unmatched = int(out[REGIME_COL].isna().sum())
    assert unmatched == 0, (
        f"{unmatched} feature-table trial(s) got no regime from {ia_source}. "
        "Every trial must carry one; do not drop or impute them."
    )

    if verbose:
        print(f"KnowQA showcase frame: {len(out)} trials x {out.shape[1]} columns")
        print(f"  participants : {out[Con.PARTICIPANT_ID].nunique()}")
        print(f"  sessions     : {out[SESSION_COL].nunique()}")
        print(f"  regimes      : {out[REGIME_COL].value_counts().to_dict()}")

    return out


def _ordered_regimes(df: pd.DataFrame) -> List[str]:
    """Regimes present in `df`, in `REGIME_ORDER`, then any other label alphabetically."""
    present = list(df[REGIME_COL].dropna().unique())
    known = [r for r in REGIME_ORDER if r in present]
    return known + sorted(r for r in present if r not in REGIME_ORDER)


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: number of sessions, number of trials, and trials per regime."""
    counts = (
        df.pivot_table(
            index=Con.PARTICIPANT_ID,
            columns=REGIME_COL,
            values=Con.IS_CORRECT_COLUMN,
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=_ordered_regimes(df), fill_value=0)
    )
    base = df.groupby(Con.PARTICIPANT_ID).agg(
        sessions=(SESSION_COL, "nunique"),
        trials=(Con.IS_CORRECT_COLUMN, "size"),
    )
    return base.join(counts).reset_index()


def _proportion_rows(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    """Per group: n, k (trials where `value_col` is 1), the proportion and its 95% Wilson
    interval."""
    rows = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(sub)
        k = int(sub[value_col].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({**dict(zip(group_cols, keys)), "n": n, "k": k,
                     "proportion": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def correctness_by_regime(
    df: pd.DataFrame,
    *,
    by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Participant accuracy with 95% Wilson intervals: a pooled `all` row, then each regime.

    `by` splits every row further (e.g. `["participant_id"]`); the `all` rows are split
    the same way.
    """
    group_cols = [REGIME_COL] + list(by or [])

    pooled = _proportion_rows(df.assign(**{REGIME_COL: "all"}), group_cols, Con.IS_CORRECT_COLUMN)
    per_regime = _proportion_rows(df, group_cols, Con.IS_CORRECT_COLUMN)

    order = {r: i for i, r in enumerate(["all"] + _ordered_regimes(df))}
    out = pd.concat([pooled, per_regime], ignore_index=True)
    return (
        out.assign(_o=out[REGIME_COL].map(order))
        .sort_values(["_o"] + list(by or []))
        .drop(columns="_o")
        .reset_index(drop=True)
        .rename(columns={"proportion": "accuracy", "k": "n_correct"})
    )


# Pace measures, with the labels the tables and figures use. KnowQA has no paragraph
# screen, so all of them are about the answer screen: the question and the four answers.
SPEED_COLS: Dict[str, str] = {
    "total_answering_RT": "total answering time (ms)",
    "total_answering_RT_normalized": "ms per word on screen",
    "RT_normalized_question": "ms per word, question",
    "RT_normalized_correct": "ms per word, correct answer",
    "RT_normalized_wrong_mean": "ms per word, wrong answers (mean)",
    "seq_len": "area transitions in the trial",
    Con.NUM_OF_SELECTS: "selection presses before confirming",
}


def reading_speed_by_regime(
    df: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean, SD and median of each pace measure, for a pooled `all` row and each regime.

    `cols` defaults to the measures in `SPEED_COLS`; any not in `df` are left out.
    """
    use = [c for c in (cols or SPEED_COLS) if c in df.columns]

    frames = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        stats = sub[use].agg(["mean", "std", "median"]).T
        stats.insert(0, "n", len(sub))
        stats.insert(0, REGIME_COL, label)
        frames.append(stats.rename_axis("measure").reset_index())

    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["measure"].map(SPEED_COLS)
    return out


def confidence_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Mean confidence (1-5) for a pooled `all` row and each regime, overall and split by
    whether the answer was correct."""
    frames = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        row: Dict[str, Any] = {REGIME_COL: label, "n": len(sub),
                               "mean_confidence": sub[CONFIDENCE_COL].mean()}
        for is_corr, tag in [(1, "correct"), (0, "wrong")]:
            part = sub[sub[Con.IS_CORRECT_COLUMN] == is_corr]
            row[f"n_{tag}"] = len(part)
            row[f"mean_confidence_{tag}"] = part[CONFIDENCE_COL].mean()
        frames.append(row)
    return pd.DataFrame(frames)


def confidence_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Share of trials at each confidence rating, one row per regime (each row sums to 1)."""
    tab = pd.crosstab(df[REGIME_COL], df[CONFIDENCE_COL], normalize="index")
    return tab.reindex(_ordered_regimes(df)).reset_index()


# ---------------------------------------------------------------------------
# Transfer: fit on L1 once, predict every KnowQA trial
# ---------------------------------------------------------------------------


def fit_l1_model(
    train_df: DataFrameOrPath = READY_ALL_FEATURES_PATH,
    feature_cols: Sequence[str] = SHOWCASE_FEATURE_COLS,
    *,
    verbose: bool = True,
) -> TrialLevelLogRegModel:
    """Fit the correctness model on every L1 trial, with nothing held out: the model that
    is then applied to Study 2."""
    l1 = _resolve(train_df)
    model = TrialLevelLogRegModel()
    model.fit(l1, target_col=Con.IS_CORRECT_COLUMN, feature_cols=list(feature_cols))

    if verbose:
        print(f"fitted {model.name} on {len(l1)} L1 trials, {len(feature_cols)} features")
        print(f"  L1 base rate: {l1[Con.IS_CORRECT_COLUMN].mean():.4f}")

    return model


def attach_predictions(
    df: pd.DataFrame,
    model: TrialLevelLogRegModel,
    feature_cols: Sequence[str] = SHOWCASE_FEATURE_COLS,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add the model's predicted probability (`pred_prob`) and predicted label
    (`pred_is_correct`) to a KnowQA frame.

    With `verbose`, first prints how many feature cells the model fills with
    `fill_value`, in total and per column. The fill is a modelling choice, not a
    measurement, so its size is reported beside the predictions it went into.
    """
    cols = list(feature_cols)

    if verbose:
        # Counted with `imputed_cell_counts`, the function the model counts its own
        # fills with, so the two counts are made the same way.
        na = imputed_cell_counts(df, cols)
        total = int(na.sum())
        print(
            f"imputed with fill_value={model.fill_value}: {total} cell(s) "
            f"= {100 * total / (len(df) * len(cols)):.3f}% of the KnowQA feature matrix"
        )
        for c, n in na.items():
            print(f"    {c}: {n} ({100 * n / len(df):.2f}% of trials)")

    out = df.copy()
    out[PROB_COL] = model.predict_proba(out, feature_cols=cols)
    out[PRED_COL] = model.predict(out, feature_cols=cols)
    return out


def _confusion_row(sub: pd.DataFrame) -> Dict[str, Any]:
    """Confusion counts for one group of trials, and the rates built from them."""
    y, p = sub[Con.IS_CORRECT_COLUMN].astype(int), sub[PRED_COL].astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    # Balanced accuracy is the mean of sensitivity and specificity, so it needs both
    # outcomes present. With only one, the other half is undefined, and averaging the
    # half that exists would report it under a name that means something else. NaN.
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    balanced = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan
    return {
        "n": len(sub), "n_correct": int(y.sum()), "n_wrong": int((y == 0).sum()),
        "accuracy": (tp + tn) / len(sub) if len(sub) else np.nan,
        "balanced_accuracy": balanced,
        "sensitivity": sens, "specificity": spec,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "FN_over_FP": fn / fp if fp else np.nan,
    }


def prediction_metrics_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """How well the model's predicted labels match the outcomes, for a pooled `all` row
    and each regime: accuracy, balanced accuracy, sensitivity, specificity, the four
    confusion counts and FN / FP."""
    rows = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        rows.append({REGIME_COL: label, **_confusion_row(sub)})
    return pd.DataFrame(rows)


def probability_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted P(correct) for a pooled `all` row and each regime: mean, median and SD,
    and the mean on correct and on wrong answers separately.

    The key number is `mean_prob_correct` under **no knowledge**. Those answers were
    right, but there was nothing to answer from, so a model that reads behaviour rather
    than outcome should not be confident about them.
    """
    rows = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        row: Dict[str, Any] = {
            REGIME_COL: label, "n": len(sub),
            "mean_prob": sub[PROB_COL].mean(),
            "median_prob": sub[PROB_COL].median(),
            "sd_prob": sub[PROB_COL].std(),
        }
        for is_corr, tag in [(1, "correct"), (0, "wrong")]:
            part = sub[sub[Con.IS_CORRECT_COLUMN] == is_corr]
            row[f"n_{tag}"] = len(part)
            row[f"mean_prob_{tag}"] = part[PROB_COL].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def decompose_probability_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Split each regime's mean predicted probability into a part due to its mix of right
    and wrong answers and a part due to how the model scores them.

    A regime can have a lower mean P(correct) because the model scores its trials lower
    (a *level* effect), or simply because it has more wrong answers, which the model
    scores low in every regime (a *composition* effect). `counterfactual_mean` mixes the
    regime's own mean probabilities on correct and on wrong answers at the **pooled**
    accuracy instead of its own. `explained_by_composition` (observed minus
    counterfactual) is the part due to the mix; what remains between
    `counterfactual_mean` and the pooled mean is the level effect.
    """
    pooled_acc = df[Con.IS_CORRECT_COLUMN].mean()
    rows = []
    for r in _ordered_regimes(df):
        sub = df[df[REGIME_COL] == r]
        acc = sub[Con.IS_CORRECT_COLUMN].mean()
        m_corr = sub.loc[sub[Con.IS_CORRECT_COLUMN] == 1, PROB_COL].mean()
        m_wrong = sub.loc[sub[Con.IS_CORRECT_COLUMN] == 0, PROB_COL].mean()
        rows.append({
            REGIME_COL: r, "n": len(sub), "accuracy": acc,
            "mean_prob": sub[PROB_COL].mean(),
            "mean_prob_correct": m_corr, "mean_prob_wrong": m_wrong,
            "counterfactual_mean": pooled_acc * m_corr + (1 - pooled_acc) * m_wrong,
        })
    out = pd.DataFrame(rows)
    out["explained_by_composition"] = out["mean_prob"] - out["counterfactual_mean"]
    return out


# ---------------------------------------------------------------------------
# Figures. Each returns its Figure; `save` decides whether it is also written to disk.
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

from src.lib.plotting.output import save_output  # noqa: E402  (kept with the plotting section)

_REGIME_COLOURS = {
    "no knowledge": "#c44e52",
    "partial knowledge": "#dd8452",
    "full knowledge": "#4c72b0",
    "all": "#8c8c8c",
}


def _colours(labels: Sequence[str]) -> List[str]:
    return [_REGIME_COLOURS.get(l, "#8c8c8c") for l in labels]


def plot_accuracy_by_regime(summary: pd.DataFrame, *, figsize=(7, 4.2), save: Optional[bool] = None, to_paper=None):
    """Participant accuracy per regime: bars with 95% Wilson intervals, and a dotted line
    at chance (1 in 4).

    Takes the output of `correctness_by_regime`; its pooled `all` row is not drawn.
    "Participant accuracy" is how often the people answered correctly, not how well the
    model predicted it; this module's figures show both, so the names keep them apart.
    """
    s = summary[summary[REGIME_COL] != "all"] if "all" in set(summary[REGIME_COL]) else summary
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(s))
    err = [s["accuracy"] - s["ci_low"], s["ci_high"] - s["accuracy"]]
    ax.bar(x, s["accuracy"], color=_colours(s[REGIME_COL]), yerr=err, capsize=5)
    ax.axhline(0.25, ls=":", c="grey", lw=1)
    # Left edge, above the line: at the right edge the label would hit the tallest bar's
    # interval and run off the axes.
    ax.text(-0.45, 0.27, "chance (1 of 4)", fontsize=8, color="grey", ha="left")
    for i, (_, r) in enumerate(s.iterrows()):
        # Above the whisker, not just the bar, so the interval does not run through the label.
        ax.text(i, r["ci_high"] + 0.025, f"{r['accuracy']:.3f}\nn={r['n']}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(s[REGIME_COL])
    ax.set_ylim(0, 1.18); ax.set_ylabel("participant accuracy")
    ax.set_title("KnowQA participant accuracy by knowledge regime (Wilson 95% CI)")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="accuracy_by_regime",
        tables={"data": summary},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_pace_by_regime(df: pd.DataFrame, measure: str = "total_answering_RT",
                        *, figsize=(7, 4.2), save: Optional[bool] = None, to_paper=None):
    """Box plot of one pace measure (a key of `SPEED_COLS`) per regime, outliers not drawn."""
    regimes = _ordered_regimes(df)
    data = [df.loc[df[REGIME_COL] == r, measure].dropna() for r in regimes]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, labels=regimes, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], _colours(regimes)):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel(SPEED_COLS.get(measure, measure))
    ax.set_title(f"{SPEED_COLS.get(measure, measure)} by regime")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="pace_by_regime",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_confidence_distribution(df: pd.DataFrame, *, figsize=(7, 4.2), save: Optional[bool] = None, to_paper=None):
    """Grouped bars: the share of trials at each confidence rating, one colour per regime."""
    tab = confidence_distribution(df).set_index(REGIME_COL)
    ratings = list(tab.columns)
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(tab)
    for i, (regime, row) in enumerate(tab.iterrows()):
        ax.bar([r + i * width for r in range(len(ratings))], row.values, width,
               label=regime, color=_REGIME_COLOURS.get(regime, "#8c8c8c"))
    ax.set_xticks([r + 0.4 - width / 2 for r in range(len(ratings))])
    ax.set_xticklabels(ratings)
    ax.set_xlabel("self-reported confidence"); ax.set_ylabel("share of trials")
    ax.set_title("Confidence by knowledge regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="confidence_distribution",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_probability_by_regime(df: pd.DataFrame, *, split_by_outcome: bool = True,
                               figsize=(8, 4.5), save: Optional[bool] = None, to_paper=None):
    """Box plots of predicted P(correct) per regime, by default split into correct and
    wrong answers.

    Split, the comparison to make is between the correct boxes: if the model reads
    knowledge rather than outcome, correct answers under no knowledge (lucky guesses)
    should sit lower than correct answers under full knowledge.
    """
    regimes = _ordered_regimes(df)
    fig, ax = plt.subplots(figsize=figsize)

    if not split_by_outcome:
        ax.boxplot([df.loc[df[REGIME_COL] == r, PROB_COL] for r in regimes],
                   labels=regimes, showfliers=False)
    else:
        positions, labels, data, colours = [], [], [], []
        for i, r in enumerate(regimes):
            for j, (is_corr, tag) in enumerate([(1, "correct"), (0, "wrong")]):
                sub = df[(df[REGIME_COL] == r) & (df[Con.IS_CORRECT_COLUMN] == is_corr)]
                positions.append(i * 2.6 + j * 0.9)
                labels.append(f"{r.split()[0]}\n{tag}\nn={len(sub)}")
                data.append(sub[PROB_COL].values)
                colours.append(_REGIME_COLOURS[r] if is_corr else "#cccccc")
        bp = ax.boxplot(data, positions=positions, widths=0.75,
                        patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)

    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_ylabel("predicted P(correct)")
    ax.set_title("L1-trained model: predicted probability on KnowQA, by regime")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="probability_by_regime",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


# ---------------------------------------------------------------------------
# Item structure — why two participants can post identical accuracies
# ---------------------------------------------------------------------------


def item_correctness(df: pd.DataFrame, *, regime: Optional[str] = None) -> pd.DataFrame:
    """Per item, hardest first: trials, accuracy, mean confidence and, if predictions are
    attached, the mean predicted probability.

    An item is a paragraph and one of its questions (`text_id_with_q`). With six
    participants an item is seen by a handful of people at most, so read every rate
    beside its `n`. `regime` limits the table to one regime.
    """
    sub = df if regime is None else df[df[REGIME_COL] == regime]
    agg = sub.groupby(Con.TEXT_ID_WITH_Q_COLUMN).agg(
        n=(Con.IS_CORRECT_COLUMN, "size"),
        n_correct=(Con.IS_CORRECT_COLUMN, "sum"),
        accuracy=(Con.IS_CORRECT_COLUMN, "mean"),
        mean_confidence=(CONFIDENCE_COL, "mean"),
    )
    if PROB_COL in sub.columns:
        agg["mean_pred_prob"] = sub.groupby(Con.TEXT_ID_WITH_Q_COLUMN)[PROB_COL].mean()
    return agg.sort_values("accuracy").reset_index()


def shared_item_structure(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """For each pair of participants in one regime: how many items they share, and on
    how many of those both were right or both were wrong.

    The counterbalancing assigns whole item sets to regimes, so two participants can get
    *identical* item sets. Their accuracies are then not independent: the same items,
    of the same difficulty, which makes matching totals much less surprising than they
    look. `same_outcome` tells this apart from duplicated data: different people agree
    on most shared items, but not on all of them.
    """
    sub = df[df[REGIME_COL] == regime]
    by_pid = {
        pid: s.set_index(Con.TEXT_ID_WITH_Q_COLUMN)[Con.IS_CORRECT_COLUMN]
        for pid, s in sub.groupby(Con.PARTICIPANT_ID)
    }

    rows = []
    pids = sorted(by_pid)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            A, B = by_pid[a], by_pid[b]
            shared = A.index.intersection(B.index)
            agree = int((A.loc[shared] == B.loc[shared]).sum()) if len(shared) else 0
            rows.append({
                "participant_a": a, "participant_b": b,
                "n_a": len(A), "n_b": len(B), "n_shared": len(shared),
                "same_outcome": agree,
                "agreement": agree / len(shared) if len(shared) else np.nan,
                "acc_a": A.mean(), "acc_b": B.mean(),
            })
    return pd.DataFrame(rows).sort_values("n_shared", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Confidence — bridge to confidence_correlation, plus discrimination
# ---------------------------------------------------------------------------


def to_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the showcase frame into the per-(run, trial) frame that
    `confidence_correlation` reads, so its tables and figures all work on it.

    One block of rows per regime, after a pooled `all` block that repeats them all: that
    module treats the pooled run as a group of its own.

    `comparison_runs.run_regime_split_comparison` gives the same predictions when run
    with the same features (`feature_cols=SHOWCASE_FEATURE_COLS`): each of its runs fits
    the same model on the same L1 data, and the model never sees the regime. With its
    default, L1's full feature set, the predictions differ.
    """
    cols = [Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.IS_CORRECT_COLUMN,
            CONFIDENCE_COL, PROB_COL, PRED_COL, SESSION_COL, REGIME_COL]
    base = df[cols].copy()

    blocks = [base.assign(run="all")]
    blocks += [base[base[REGIME_COL] == r].assign(run=r) for r in _ordered_regimes(df)]

    out = pd.concat(blocks, ignore_index=True)
    out["y_true"] = out[Con.IS_CORRECT_COLUMN].astype(int)
    out["y_pred"] = out[PRED_COL].astype(int)
    return out


def _auc(score: pd.Series, target: pd.Series) -> float:
    """ROC AUC over complete pairs; NaN when only one outcome occurs (undefined, not 0.5)."""
    from sklearn.metrics import roc_auc_score

    ok = score.notna() & target.notna()
    y = target[ok]
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, score[ok]))


def discrimination_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Does the model tell correct from wrong answers apart better than the confidence
    rating does?

    The paper's claim is that P(correct) can stand in for an unreliable confidence
    rating. That is a claim about telling the two apart, so both are scored the same way:
    the AUC for predicting `is_correct`, for a pooled `all` row and each regime.
    `auc_combined` scores the average of the two ranks, which says whether they carry the
    same information or add to each other.
    """
    rows = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        combined = (sub[PROB_COL].rank(pct=True) + sub[CONFIDENCE_COL].rank(pct=True)) / 2
        rows.append({
            REGIME_COL: label, "n": len(sub),
            "n_correct": int(sub[Con.IS_CORRECT_COLUMN].sum()),
            "auc_model": _auc(sub[PROB_COL], sub[Con.IS_CORRECT_COLUMN]),
            "auc_confidence": _auc(sub[CONFIDENCE_COL], sub[Con.IS_CORRECT_COLUMN]),
            "auc_combined": _auc(combined, sub[Con.IS_CORRECT_COLUMN]),
        })
    out = pd.DataFrame(rows)
    out["model_minus_confidence"] = out["auc_model"] - out["auc_confidence"]
    return out


# ---------------------------------------------------------------------------
# Per-participant profiles
# ---------------------------------------------------------------------------


def participant_profile(
    df: pd.DataFrame,
    *,
    by: Sequence[str] = (REGIME_COL,),
) -> pd.DataFrame:
    """One row per participant and regime, or per participant and session with
    `by=(SESSION_COL,)`.

    The participant's side: trials, wrong answers, `participant_accuracy` (how often
    they were right), mean confidence, median answering time and mean `seq_len`. With
    predictions attached, the model's side too: `model_accuracy` is how often its
    predicted label matched the outcome, `mean_pred_prob` what it believed on average.
    They answer different questions and can move in opposite directions.

    Compare cells on `model_balanced_accuracy`, not `model_accuracy`: participant
    accuracy runs from about 0.3 to 1.0 across these cells, so plain accuracy mostly
    reports how often the participant was right rather than how well the model read
    them. Balanced accuracy needs both outcomes, so it is NaN where a participant made
    no errors. `n_wrong` is there because under full knowledge it is 0 to 3 trials, and
    a rate resting on three trials should be read as such.
    """
    keys = [Con.PARTICIPANT_ID, *by]
    rows = []
    for vals, sub in df.groupby(keys, sort=True):
        vals = vals if isinstance(vals, tuple) else (vals,)
        row = dict(zip(keys, vals))
        row.update({
            "n": len(sub),
            "n_wrong": int((sub[Con.IS_CORRECT_COLUMN] == 0).sum()),
            "participant_accuracy": sub[Con.IS_CORRECT_COLUMN].mean(),
            "mean_confidence": sub[CONFIDENCE_COL].mean(),
            "median_RT_ms": sub["total_answering_RT"].median(),
            "mean_seq_len": sub["seq_len"].mean(),
        })
        if PROB_COL in sub.columns:
            conf = _confusion_row(sub)
            row.update({
                "mean_pred_prob": sub[PROB_COL].mean(),
                "model_accuracy": conf["accuracy"],
                "model_balanced_accuracy": conf["balanced_accuracy"],
                "model_sensitivity": conf["sensitivity"],
                "model_specificity": conf["specificity"],
                "auc_model": _auc(sub[PROB_COL], sub[Con.IS_CORRECT_COLUMN]),
                "auc_confidence": _auc(sub[CONFIDENCE_COL], sub[Con.IS_CORRECT_COLUMN]),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion_by_regime(df: pd.DataFrame, *, normalize: str = "true",
                             figsize=(13, 4.2), save: Optional[bool] = None, to_paper=None):
    """One confusion matrix per regime, with the counts printed in each cell.

    `normalize="true"` (the default) shades each row by its share of the actual outcome,
    which makes the panels comparable: participant accuracy runs from 0.42 to 0.97
    across the regimes, so raw counts would say more about the manipulation than about
    the model. `normalize="all"` shades by the share of all trials.

    With rows normalised, the two diagonal shares are the specificity (top left) and the
    sensitivity (bottom right), so each title gives their mean, the balanced accuracy,
    beside plain accuracy and the base rate.
    """
    regimes = _ordered_regimes(df)
    fig, axes = plt.subplots(1, len(regimes), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, regimes):
        sub = df[df[REGIME_COL] == r]
        y, p = sub[Con.IS_CORRECT_COLUMN].astype(int), sub[PRED_COL].astype(int)
        cm = np.array([[int(((y == a) & (p == b)).sum()) for b in (0, 1)] for a in (0, 1)],
                      dtype=float)

        shown = cm.copy()
        if normalize == "true":
            with np.errstate(invalid="ignore"):
                shown = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "all":
            shown = cm / cm.sum()

        ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        for i in range(2):
            for j in range(2):
                frac = "" if not normalize else f"\n{shown[i, j]:.2f}"
                ax.text(j, i, f"{int(cm[i, j])}{frac}", ha="center", va="center",
                        fontsize=11,
                        color="white" if shown[i, j] > 0.55 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred wrong", "pred correct"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["was wrong", "was correct"])

        # Worked out from this panel's own cells, so the title always matches its matrix.
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        acc = (tp + tn) / len(sub) if len(sub) else np.nan
        # Both halves required — see _confusion_row.
        bal = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan
        ax.set_title(
            f"{r}\nn={len(sub)} · base {y.mean():.2f}\n"
            f"acc {acc:.2f} · balanced acc {bal:.2f}",
            fontsize=9.5,
        )

    fig.suptitle("L1 model on KnowQA — confusion by regime "
                 f"({'row-normalised' if normalize == 'true' else normalize})",
                 fontsize=11)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="confusion_by_regime",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_probability_histograms(df: pd.DataFrame, *, bins: int = 20,
                                split_by_outcome: bool = True, figsize=(13, 3.8), save: Optional[bool] = None, to_paper=None):
    """Histogram of predicted P(correct) per regime, by default split into correct and
    wrong answers; the dashed line is the regime's mean.

    Split, each panel asks whether the two humps sit in different places: whether the
    model separates correct from wrong answers *within* that regime, rather than only
    following the regime's base rate.
    """
    regimes = _ordered_regimes(df)
    edges = np.linspace(0, 1, bins + 1)
    fig, axes = plt.subplots(1, len(regimes), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, regimes):
        sub = df[df[REGIME_COL] == r]
        if split_by_outcome:
            for is_corr, colour, tag in [(0, "#c44e52", "was wrong"),
                                         (1, "#4c72b0", "was correct")]:
                vals = sub.loc[sub[Con.IS_CORRECT_COLUMN] == is_corr, PROB_COL]
                ax.hist(vals, bins=edges, alpha=0.65, color=colour,
                        label=f"{tag} (n={len(vals)})")
        else:
            ax.hist(sub[PROB_COL], bins=edges, color=_REGIME_COLOURS.get(r, "#8c8c8c"))

        ax.axvline(sub[PROB_COL].mean(), ls="--", c="k", lw=1.2)
        ax.set_title(f"{r}\nmean P = {sub[PROB_COL].mean():.3f}", fontsize=10)
        ax.set_xlabel("predicted P(correct)")
        ax.legend(fontsize=7, frameon=False)

    axes[0].set_ylabel("trials")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="probability_histograms",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_participant_profiles(df: pd.DataFrame, *,
                              measures: Sequence[str] = ("participant_accuracy", "mean_confidence",
                                                         "mean_pred_prob", "median_RT_ms"),
                              figsize=(12, 7), save: Optional[bool] = None, to_paper=None):
    """One panel per measure of `participant_profile`: participants along x, one line
    per regime.

    Six participants is too few to test anything, so this is for seeing *shape*:
    whether a person is low everywhere or in one regime only, and whether their
    confidence and the model's probability move together.
    """
    prof = participant_profile(df)
    regimes = _ordered_regimes(df)
    use = [m for m in measures if m in prof.columns]

    ncols = 2
    nrows = int(np.ceil(len(use) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, measure in zip(axes.ravel(), use):
        for r in regimes:
            s = prof[prof[REGIME_COL] == r].sort_values(Con.PARTICIPANT_ID)
            ax.plot(s[Con.PARTICIPANT_ID], s[measure], marker="o",
                    color=_REGIME_COLOURS.get(r, "#8c8c8c"), label=r)
        ax.set_title(measure, fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(use):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Per-participant profile by regime", fontsize=11)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="participant_profiles",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


# ---------------------------------------------------------------------------
# Figures for the correlation / comparison tables
# ---------------------------------------------------------------------------

from src.lib.stats.proportions import p_to_stars  # noqa: E402


def _run_order_local(runs: Sequence[str]) -> List[str]:
    """The pooled `all` run first, then `REGIME_ORDER`, then any other label alphabetically."""
    out = [r for r in runs if r == "all"]
    out += [r for r in REGIME_ORDER if r in runs]
    return out + sorted(r for r in runs if r != "all" and r not in REGIME_ORDER)


def plot_correlations_by_run(corr_df: pd.DataFrame, *, method: str = "spearman",
                             figsize=(9, 4.6), save: Optional[bool] = None, to_paper=None):
    """Grouped bars of `confidence_correlation.correlations_by_run`: one cluster per run,
    one bar per comparison, with significance stars.

    Read the two comparisons together. `is_correct ~ confidence` is the reference, how
    well the rating tracks correctness in that run, and `pred_prob ~ confidence` only
    means something against it. Where both are near zero, the rating carried no signal
    there; that is not a failure of the model.
    """
    r_col, p_col = f"{method}_r", f"{method}_p"
    runs = _run_order_local(corr_df["run"].unique().tolist())
    comps = list(dict.fromkeys(corr_df["comparison"]))

    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(comps)
    palette = ["#4c72b0", "#dd8452", "#55a868"]

    for j, comp in enumerate(comps):
        xs, ys, ps = [], [], []
        for i, run in enumerate(runs):
            row = corr_df[(corr_df["run"] == run) & (corr_df["comparison"] == comp)]
            xs.append(i + j * width)
            ys.append(float(row[r_col].iloc[0]) if len(row) else np.nan)
            ps.append(float(row[p_col].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(xs, ys, width, label=comp, color=palette[j % len(palette)])
        for b, y, p in zip(bars, ys, ps):
            if np.isfinite(y):
                ax.text(b.get_x() + b.get_width() / 2,
                        y + (0.02 if y >= 0 else -0.05),
                        p_to_stars(p), ha="center", fontsize=8)

    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(runs))])
    ax.set_xticklabels(runs)
    ax.set_ylabel(f"{method} r")
    ax.set_title("Confidence correlations by regime")
    # Headroom for the significance stars, which sit above each bar.
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0), hi * 1.12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="correlations_by_run",
        tables={"data": corr_df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_participant_correlation_forest(
    by_model: pd.DataFrame,
    by_correct: Optional[pd.DataFrame] = None,
    *,
    method: str = "spearman",
    figsize=(8, 4.6),
    save: Optional[bool] = None,
    to_paper=None,
):
    """Within-participant correlations as a dot plot, one row per participant.

    `by_model` and `by_correct` come from `confidence_correlation.correlations_by_participant`:
    `pred_prob ~ confidence`, and the same with `prob_col="is_correct"`. Confidence
    scales are personal, so the pooled correlation mixes the effect within each person
    with how differently people use the scale; one row per person keeps only the first.
    Drawing both series shows whether a participant whose rating tracks their own
    correctness is also one the model agrees with. Only `by_model` is saved as the
    figure's table.
    """
    r_col, p_col = f"{method}_r", f"{method}_p"
    pids = list(by_model[Con.PARTICIPANT_ID])
    y = np.arange(len(pids))

    model_r = by_model.set_index(Con.PARTICIPANT_ID).reindex(pids)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(model_r[r_col], y, s=70, color="#4c72b0", zorder=3,
               label="pred_prob ~ confidence")

    if by_correct is not None:
        correct_r = by_correct.set_index(Con.PARTICIPANT_ID).reindex(pids)
        for yi in y:
            a, b = model_r[r_col].iloc[yi], correct_r[r_col].iloc[yi]
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [yi, yi], c="grey", lw=1, zorder=1)
        ax.scatter(correct_r[r_col], y, s=70, marker="D", color="#dd8452", zorder=3,
                   label="is_correct ~ confidence")
    else:
        correct_r = None

    # Stars on both series, offset in opposite directions: the two markers can sit almost
    # on top of each other (participant 4004: 0.60 and 0.58), and starring only one would
    # make the other look non-significant when it is not.
    for frame, colour, dy, va in [(model_r, "#4c72b0", 0.20, "bottom"),
                                  (correct_r, "#dd8452", -0.20, "top")]:
        if frame is None:
            continue
        for yi, r, p in zip(y, frame[r_col], frame[p_col]):
            if np.isfinite(r):
                ax.text(r, yi + dy, p_to_stars(p), ha="center", va=va,
                        fontsize=7, color=colour)

    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pids)
    ax.set_ylim(-0.75, len(pids) - 0.15)   # room for the stars above and below
    ax.set_xlabel(f"{method} r")
    ax.set_ylabel("participant")
    ax.set_title("Within-participant confidence correlations")
    ax.grid(axis="x", alpha=0.25)
    # Anchored outside the axes: loc="best" lands on the points when the dots cluster.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="participant_correlation_forest",
        tables={"data": by_model},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_discrimination_roc(df: pd.DataFrame, *, figsize=(13, 4.2), save: Optional[bool] = None, to_paper=None):
    """ROC curves per regime for predicting a correct answer: the model's P(correct)
    against the confidence rating.

    The AUC table (`discrimination_comparison`) says which separates correct from wrong
    answers better; the curves show where. Near the origin are the highest scores: a
    rating whose curve follows the diagonal there gives its top value as often to wrong
    answers as to correct ones.
    """
    from sklearn.metrics import roc_curve

    regimes = _ordered_regimes(df)
    fig, axes = plt.subplots(1, len(regimes), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, r in zip(axes, regimes):
        sub = df[df[REGIME_COL] == r]
        y = sub[Con.IS_CORRECT_COLUMN].astype(int)
        for score, colour, name in [(sub[PROB_COL], "#4c72b0", "model P(correct)"),
                                    (sub[CONFIDENCE_COL], "#dd8452", "self-report")]:
            if y.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y, score)
            ax.plot(fpr, tpr, color=colour, lw=2,
                    label=f"{name} (AUC {_auc(score, y):.3f})")
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
        ax.set_title(f"{r}\nn={len(sub)}, {int(y.sum())} correct", fontsize=10)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")

    axes[0].set_ylabel("true positive rate")
    fig.suptitle("Discriminating correct from wrong: model vs self-report", fontsize=11)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="discrimination_roc",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_transfer_metrics(metrics: pd.DataFrame, *,
                          measures: Sequence[str] = ("accuracy", "balanced_accuracy",
                                                     "sensitivity", "specificity"),
                          figsize=(9.5, 4.4), save: Optional[bool] = None, to_paper=None):
    """Grouped bars of the transfer metrics from `prediction_metrics_by_regime`, with
    each regime's base rate marked.

    The dashed line is the run's own participant accuracy (`n_correct / n`): the score
    of always predicting "correct". Plain accuracy cannot be read without it. Base rates
    run from 0.42 to 0.97, and the model is trained with balanced class weights, so it
    gives up accuracy on the common outcome to catch the rare one: in the high-base-rate
    regimes its accuracy sits *below* the line while its specificity stays near 0.91.
    Read balanced accuracy; the line is there so the accuracy bar is not read as
    performance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    runs = list(metrics[REGIME_COL])
    width = 0.8 / len(measures)
    palette = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]

    for j, m in enumerate(measures):
        xs = [i + j * width for i in range(len(runs))]
        ax.bar(xs, metrics[m], width, label=m, color=palette[j % len(palette)])

    for i, row in metrics.reset_index(drop=True).iterrows():
        base = row["n_correct"] / row["n"]
        ax.plot([i - 0.1, i + 0.8], [base, base], c="k", ls="--", lw=1.2)
        ax.text(i + 0.82, base, f"always-correct\nbaseline {base:.2f}",
                fontsize=6.5, va="center")

    ax.axhline(0.5, c="grey", ls=":", lw=1)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(runs))])
    ax.set_xticklabels(runs)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("L1 model transferred to KnowQA, by regime\n"
                 "(dashed = always-predict-correct baseline)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="transfer_metrics",
        tables={"data": metrics},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_probability_decomposition(dec: pd.DataFrame, *, figsize=(8.5, 4.2), save: Optional[bool] = None, to_paper=None):
    """Per regime, two dots from `decompose_probability_by_regime`: the observed mean
    P(correct), and the same regime re-mixed at the pooled accuracy.

    The gap between the dots is the part of the regime's mean that its mix of right and
    wrong answers explains. Where they coincide, any difference from the other regimes
    comes from how the model scores that regime's trials.
    """
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(dec))

    for yi, (_, row) in zip(y, dec.iterrows()):
        ax.plot([row["counterfactual_mean"], row["mean_prob"]], [yi, yi],
                c="grey", lw=2, zorder=1)
    ax.scatter(dec["counterfactual_mean"], y, s=90, color="#8c8c8c", zorder=3,
               label="re-mixed at pooled accuracy")
    ax.scatter(dec["mean_prob"], y, s=90,
               color=_colours(dec[REGIME_COL]), zorder=3, label="observed")

    for yi, (_, row) in zip(y, dec.iterrows()):
        ax.text(max(row["mean_prob"], row["counterfactual_mean"]) + 0.015, yi,
                f"{row['explained_by_composition']:+.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(dec[REGIME_COL])
    ax.set_xlabel("mean predicted P(correct)")
    ax.set_title("How much of the regime ordering is composition?")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="probability_decomposition",
        tables={"data": dec},
        save=save,
        to_paper=to_paper,
    )
    return fig


def _annotated_heatmap(ax, mat: pd.DataFrame, *, fmt: str = "{:.2f}",
                       cmap: str = "RdYlBu", vmin=None, vmax=None, cbar_label="",
                       marks: Optional[pd.DataFrame] = None):
    """Draw `mat` as a heatmap with every value printed, and label both axes. Missing
    cells show "-".

    `marks` is an optional frame of the same shape holding a suffix per cell (e.g.
    "*"), to flag a value that is not the panel's main measure; the suffix is printed
    with the number, so the two stay together. Each label is white or black according
    to how dark its cell is actually drawn, so it stays readable on any colour map,
    including ones that are near-white at one end.
    """
    im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="grey")
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b   # ITU-R BT.601
            suffix = "" if marks is None else str(marks.values[i, j] or "")
            ax.text(j, i, fmt.format(v) + suffix, ha="center", va="center", fontsize=8,
                    color="white" if luminance < 0.55 else "black")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return im


def plot_participant_regime_heatmap(
    df: pd.DataFrame,
    *,
    measures: Sequence[str] = ("participant_accuracy", "mean_confidence", "mean_pred_prob",
                               "model_balanced_accuracy"),
    side_column: Optional[str] = None,
    figsize=(14, 3.8),
    save: Optional[bool] = None,
    to_paper=None,
):
    """Participant x regime heatmaps, one panel per measure of `participant_profile`.

    The shading shows at a glance a participant who is out of line in one regime only,
    rather than everywhere, which is hard to see in the table.

    The model panel shows **balanced** accuracy, because participant accuracy (the
    first panel) runs from 0.32 to 1.00 across these cells, and plain accuracy would
    largely repeat it. Where a participant made no errors in a regime, balanced accuracy
    is undefined (no wrong answers, so no specificity); that cell shows the model's plain
    accuracy instead, marked with a star and explained under the figure.

    `side_column` adds a participant-level column (`"lextale"`, after `attach_lextale`)
    as a narrow panel on the left, with its **own** colour scale spanning only the
    observed range: LexTALE is 0-100 where the other measures are 0-1, and the six scores
    (75-95) would otherwise all look alike. Its colours rank these participants against
    each other only; the printed numbers are the real scores.
    """
    prof = participant_profile(df)
    use = [m for m in measures if m in prof.columns]
    regimes = _ordered_regimes(df)

    if side_column is not None:
        assert side_column in df.columns, (
            f"{side_column!r} is not in the frame — call the matching attach_* first "
            f"(e.g. attach_lextale for {LEXTALE_COL!r})."
        )
        side = df.groupby(Con.PARTICIPANT_ID)[side_column].first().to_frame(side_column)
        widths = [0.42] + [1.0] * len(use)
        fig, all_axes = plt.subplots(1, len(use) + 1, figsize=figsize,
                                     gridspec_kw={"width_ratios": widths})
        all_axes = np.atleast_1d(all_axes)
        side_ax, axes = all_axes[0], all_axes[1:]
        _annotated_heatmap(side_ax, side.reindex(prof[Con.PARTICIPANT_ID].unique()),
                           fmt="{:g}", cmap="PuBuGn",
                           vmin=float(side[side_column].min()),
                           vmax=float(side[side_column].max()))
        side_ax.set_title(f"{side_column}\n(own scale)", fontsize=9)
        side_ax.set_xticks([])
    else:
        fig, axes = plt.subplots(1, len(use), figsize=figsize)
        axes = np.atleast_1d(axes)

    starred = False
    for ax, m in zip(axes, use):
        mat = (prof.pivot(index=Con.PARTICIPANT_ID, columns=REGIME_COL, values=m)
                   .reindex(columns=regimes))
        marks = None

        # Balanced accuracy is undefined where a participant made no errors in a regime:
        # there are no wrong answers to measure specificity on. A blank cell would read
        # as "no result for the model", yet its plain accuracy is well defined, so the
        # cell shows that instead, starred because it is a different measure. Only the
        # figure changes; participant_profile keeps the NaN.
        if m == "model_balanced_accuracy" and "model_accuracy" in prof.columns:
            fallback = (prof.pivot(index=Con.PARTICIPANT_ID, columns=REGIME_COL,
                                   values="model_accuracy")
                            .reindex(columns=regimes))
            gaps = mat.isna() & fallback.notna()
            if gaps.to_numpy().any():
                starred = True
                marks = gaps.map(lambda flag: "*" if flag else "")
                mat = mat.fillna(fallback)

        # Confidence is a 1-5 scale; everything else here is a 0-1 rate.
        vmin, vmax = (1, 5) if m == "mean_confidence" else (0, 1)
        _annotated_heatmap(ax, mat, vmin=vmin, vmax=vmax, marks=marks)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("")
        # Participant ids once, on whichever panel is leftmost.
        if side_column is not None or ax is not axes[0]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    fig.suptitle("Per participant, per regime", fontsize=11)
    fig.tight_layout()
    if starred:
        fig.text(0.005, -0.02,
                 "* the model's plain accuracy, not its balanced accuracy: the "
                 "PARTICIPANT answered every trial in that regime correctly, so there "
                 "are no wrong trials to measure specificity on and balanced accuracy "
                 "is undefined.",
                 fontsize=7.5, ha="left", va="top", wrap=True)
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="participant_regime_heatmap",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_shared_item_structure(pairs: pd.DataFrame, *, figsize=(11, 4.4), save: Optional[bool] = None, to_paper=None):
    """Two heatmaps over pairs of participants, from `shared_item_structure`: how many
    items they shared, and how often they answered the shared ones the same way.

    The left panel shows the design (the counterbalancing gives some pairs identical
    item sets); the right is the check that they are still different people. A pair
    agreeing on every shared item would mean duplicated data, not a shared list.
    """
    pids = sorted(set(pairs["participant_a"]) | set(pairs["participant_b"]))
    shared = pd.DataFrame(np.nan, index=pids, columns=pids, dtype=float)
    agree = pd.DataFrame(np.nan, index=pids, columns=pids, dtype=float)

    for _, r in pairs.iterrows():
        a, b = r["participant_a"], r["participant_b"]
        shared.loc[a, b] = shared.loc[b, a] = r["n_shared"]
        agree.loc[a, b] = agree.loc[b, a] = r["agreement"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _annotated_heatmap(axes[0], shared, fmt="{:.0f}", cmap="Blues",
                       vmin=0, cbar_label="items in common")
    axes[0].set_title("Shared item sets", fontsize=10)
    _annotated_heatmap(axes[1], agree, fmt="{:.2f}", cmap="Greens",
                       vmin=0, vmax=1, cbar_label="same outcome")
    axes[1].set_title("Agreement on shared items", fontsize=10)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="shared_item_structure",
        tables={"data": pairs},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_item_difficulty(items: pd.DataFrame, *, figsize=(11, 4.4), save: Optional[bool] = None, to_paper=None):
    """Item accuracy, from `item_correctness`: its distribution, and whether the model
    knew which items were hard.

    The scatter, drawn only when predictions were attached, is the panel to read: one dot
    per item, sized by its number of trials and coloured by mean confidence. Items on the
    diagonal are ones the model read the way the participants performed; low-accuracy,
    high-probability items are where it was confidently wrong, and are worth reading one
    by one.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(items["accuracy"], bins=np.linspace(0, 1, 11),
                 color="#4c72b0", edgecolor="white")
    axes[0].set_xlabel("item accuracy")
    axes[0].set_ylabel("items")
    axes[0].set_title(f"Item difficulty ({len(items)} items)", fontsize=10)

    if "mean_pred_prob" in items.columns:
        sizes = 18 * items["n"].clip(lower=1)
        sc = axes[1].scatter(items["accuracy"], items["mean_pred_prob"], s=sizes,
                             c=items["mean_confidence"], cmap="viridis",
                             alpha=0.75, edgecolor="k", linewidth=0.3)
        axes[1].plot([0, 1], [0, 1], ls=":", c="grey")
        axes[1].set_xlabel("item accuracy (what happened)")
        axes[1].set_ylabel("mean predicted P(correct)")
        axes[1].set_title("Did the model know which items were hard?", fontsize=10)
        cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label("mean confidence", fontsize=8)

    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="item_difficulty",
        tables={"data": items},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_session_trajectories(
    df: pd.DataFrame,
    *,
    measures: Sequence[str] = ("participant_accuracy", "mean_confidence", "median_RT_ms",
                               "mean_pred_prob"),
    figsize=(12, 7),
    save: Optional[bool] = None,
    to_paper=None,
):
    """One line per participant across their sittings, one panel per measure of
    `participant_profile`.

    Practice and fatigue affect every pace measure: a participant whose time falls
    steadily across sessions is learning the interface, not responding to the regimes.
    Sessions are put in order by their id, which gives the sitting order (see
    `add_session_order`).
    """
    prof = participant_profile(df, by=(SESSION_COL,))
    use = [m for m in measures if m in prof.columns]

    nrows = int(np.ceil(len(use) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False)

    for ax, m in zip(axes.ravel(), use):
        for pid, sub in prof.groupby(Con.PARTICIPANT_ID):
            sub = sub.sort_values(SESSION_COL)
            ax.plot(range(1, len(sub) + 1), sub[m], marker="o", label=pid)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("session (in order)")
        ax.set_xticks(range(1, 1 + prof.groupby(Con.PARTICIPANT_ID).size().max()))
        ax.grid(alpha=0.25)

    for ax in axes.ravel()[len(use):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2, title="participant")
    fig.suptitle("Across the three sittings", fontsize=11)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="session_trajectories",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_pace_panel(df: pd.DataFrame, *, figsize=(12, 6.5), save: Optional[bool] = None, to_paper=None):
    """Every pace measure in `SPEED_COLS`, one panel each, regimes along x: means with
    standard-error whiskers.

    The measures are on very different scales (milliseconds against counts), hence one
    panel each rather than one chart.
    """
    regimes = _ordered_regimes(df)
    use = [c for c in SPEED_COLS if c in df.columns]

    ncols = 3
    nrows = int(np.ceil(len(use) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for ax, measure in zip(axes.ravel(), use):
        means, errs = [], []
        for r in regimes:
            vals = df.loc[df[REGIME_COL] == r, measure].dropna()
            means.append(vals.mean())
            errs.append(vals.std() / np.sqrt(len(vals)) if len(vals) else np.nan)
        ax.bar(range(len(regimes)), means, yerr=errs, capsize=4,
               color=_colours(regimes))
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels([r.split()[0] for r in regimes], fontsize=8)
        ax.set_title(SPEED_COLS.get(measure, measure), fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(use):]:
        ax.set_visible(False)
    fig.suptitle("Pace measures by regime (mean +/- SE)", fontsize=11)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="pace_panel",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


def plot_confidence_by_regime(conf: pd.DataFrame, *, figsize=(8.5, 4.2), save: Optional[bool] = None, to_paper=None):
    """Mean confidence per regime, side by side for correct and wrong answers, from
    `confidence_by_regime`.

    The gap between a regime's two bars is how well the participants could tell, in
    that regime, when they were wrong: those who can show two clearly separated bars.
    """
    rows = conf[conf[REGIME_COL] != "all"] if "all" in set(conf[REGIME_COL]) else conf
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(rows))
    width = 0.35

    ax.bar(x - width / 2, rows["mean_confidence_correct"], width,
           label="answer was correct", color="#4c72b0")
    ax.bar(x + width / 2, rows["mean_confidence_wrong"], width,
           label="answer was wrong", color="#c44e52")

    for xi, (_, r) in zip(x, rows.iterrows()):
        for off, col, n in [(-width / 2, "mean_confidence_correct", "n_correct"),
                            (width / 2, "mean_confidence_wrong", "n_wrong")]:
            if np.isfinite(r[col]):
                ax.text(xi + off, r[col] + 0.06, f"{r[col]:.2f}\nn={int(r[n])}",
                        ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(rows[REGIME_COL])
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("mean confidence (1-5)")
    ax.set_title("Self-reported confidence by regime and outcome")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="confidence_by_regime",
        tables={"data": conf},
        save=save,
        to_paper=to_paper,
    )
    return fig


# ---------------------------------------------------------------------------
# Session order: is there a drift across the three sittings?
# ---------------------------------------------------------------------------
#
# READ THIS BEFORE INTERPRETING ANYTHING BELOW.
#
# Session order is **perfectly confounded with article batch**: every participant took
# batch 1 in their first sitting, batch 2 in their second and batch 3 in their third,
# and the three batches share no items. So a decline across sittings has two
# explanations that cannot be told apart -- the participants tiring, or the later
# batches being harder -- and more participants will not separate them while the batch
# order stays fixed. Only counterbalancing the batch order would.
#
# What can be done is to ask how hard the same items are for people with no sittings at
# all. Every KnowQA item also appears in L1, where it was seen under a different design,
# so L1's accuracy on an item is a difficulty baseline with no session effect in it.
# `attach_item_difficulty` adds it, and the tests below control for it.

SESS_ORDER_COL = "sess_order"
ITEM_DIFFICULTY_COL = "l1_item_acc"


def add_session_order(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sess_order`: 1, 2, 3 for each participant's sittings, in order.

    It ranks each participant's `session_id`s (`4000_21` is batch 2, list 1). The batch
    digit comes first, so this is the batch order, which is also the order of the
    sittings because every participant took the batches in order 1, 2, 3 (see the note
    above).
    """
    out = df.copy()
    out[SESS_ORDER_COL] = (
        out.groupby(Con.PARTICIPANT_ID)[SESSION_COL].rank(method="dense").astype(int)
    )
    return out


def attach_item_difficulty(
    df: pd.DataFrame,
    l1_source: DataFrameOrPath = READY_ALL_FEATURES_PATH,
) -> pd.DataFrame:
    """Add each item's **L1** accuracy (`l1_item_acc`): a difficulty baseline with no
    session effect in it.

    Stops if a KnowQA item has no L1 counterpart: its trials would get NaN and then drop
    out of any model that adjusts for difficulty, quietly changing which trials the
    session estimate rests on.

    Safe to call twice: an existing `l1_item_acc` is dropped and recomputed rather than
    merged again (which would leave two copies, `_x` and `_y`), so a changed `l1_source`
    takes effect.
    """
    l1 = _resolve(l1_source)
    diff = l1.groupby(Con.TEXT_ID_WITH_Q_COLUMN)[Con.IS_CORRECT_COLUMN].mean()

    base = df.drop(columns=[ITEM_DIFFICULTY_COL], errors="ignore")
    out = base.merge(diff.rename(ITEM_DIFFICULTY_COL), left_on=Con.TEXT_ID_WITH_Q_COLUMN,
                     right_index=True, how="left")

    assert len(out) == len(df), (
        f"item-difficulty merge changed the row count: {len(df)} -> {len(out)}; "
        "the L1 difficulty index is not unique per item"
    )
    missing = int(out[ITEM_DIFFICULTY_COL].isna().sum())
    assert missing == 0, (
        f"{missing} KnowQA trial(s) have no L1 counterpart item, so no difficulty "
        "baseline. Do not drop them silently — decide what they should be."
    )
    return out


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sitting: what the participants did and, if predictions are attached,
    what the model made of it.

    `l1_item_acc` is how the *same items* were answered in L1, a baseline with no
    sittings in it. Read participant accuracy against it: a drop the L1 column shares is
    the items getting harder, not the participants. Adds `sess_order` and `l1_item_acc`
    first if they are missing.
    """
    work = df if SESS_ORDER_COL in df.columns else add_session_order(df)
    if ITEM_DIFFICULTY_COL not in work.columns:
        work = attach_item_difficulty(work)

    rows = []
    for order, sub in work.groupby(SESS_ORDER_COL):
        y = sub[Con.IS_CORRECT_COLUMN].astype(int)
        row: Dict[str, Any] = {
            "session": order, "n": len(sub),
            "n_items": sub[Con.TEXT_ID_WITH_Q_COLUMN].nunique(),
            "participant_accuracy": y.mean(),
            ITEM_DIFFICULTY_COL: sub[ITEM_DIFFICULTY_COL].mean(),
            "mean_confidence": sub[CONFIDENCE_COL].mean(),
        }
        if PROB_COL in sub.columns:
            conf = _confusion_row(sub)
            row.update({
                "model_balanced_accuracy": conf["balanced_accuracy"],
                "model_auc": _auc(sub[PROB_COL], y),
                "mean_pred_prob": sub[PROB_COL].mean(),
                "confidence_auc": _auc(sub[CONFIDENCE_COL], y),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def session_trend_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Is the decline across sittings real? Three tests, none of which can be conclusive
    with six participants.

    All three respect that the trials are grouped within six people: a test treating the
    870 trials as independent would give far too small p-values.

    * **Wilcoxon, first vs last sitting**, on the six per-participant accuracies. With
      six pairs the smallest possible p is 0.031, so this row cannot reach 0.01 however
      large the effect.
    * **Per-participant trend**: one Spearman rho per person between sitting and
      outcome, then a one-sample t-test of the six values against 0.
    * **Logistic regression with standard errors clustered by participant**, on sitting
      and regime, then again with L1 item difficulty added. The pair is the point: if
      the sitting coefficient shrinks once difficulty is in, the decline was mostly the
      items. `statistic` is the odds ratio per sitting.
    """
    from scipy import stats
    import statsmodels.api as sm

    work = df if SESS_ORDER_COL in df.columns else add_session_order(df)
    if ITEM_DIFFICULTY_COL not in work.columns:
        work = attach_item_difficulty(work)

    rows: List[Dict[str, Any]] = []

    per_part = work.pivot_table(index=Con.PARTICIPANT_ID, columns=SESS_ORDER_COL,
                                values=Con.IS_CORRECT_COLUMN, aggfunc="mean")
    first, last = per_part.columns.min(), per_part.columns.max()
    w = stats.wilcoxon(per_part[first], per_part[last])
    rows.append({"test": f"Wilcoxon, sitting {first} vs {last}", "n_units": len(per_part),
                 "statistic": float(w.statistic), "p_value": float(w.pvalue),
                 "effect": per_part[last].mean() - per_part[first].mean(),
                 "effect_is": "mean accuracy difference"})

    rhos = [stats.spearmanr(s[SESS_ORDER_COL], s[Con.IS_CORRECT_COLUMN]).statistic
            for _, s in work.groupby(Con.PARTICIPANT_ID)]
    t = stats.ttest_1samp(rhos, 0)
    rows.append({"test": "per-participant Spearman(sitting, correct), vs 0",
                 "n_units": len(rhos), "statistic": float(t.statistic),
                 "p_value": float(t.pvalue), "effect": float(np.mean(rhos)),
                 "effect_is": "mean within-participant rho"})

    # The design matrix is built by hand rather than from a formula: `regime` has pandas'
    # string dtype, which patsy cannot read.
    y = work[Con.IS_CORRECT_COLUMN].astype(int).to_numpy()
    groups = pd.factorize(np.asarray(work[Con.PARTICIPANT_ID], dtype=object))[0]
    reg = np.asarray(work[REGIME_COL], dtype=object)
    base = [np.ones(len(y)), work[SESS_ORDER_COL].to_numpy(float),
            (reg == "partial knowledge").astype(float), (reg == "full knowledge").astype(float)]

    for label, X in [
        ("clustered logit: sitting + regime", np.column_stack(base)),
        ("clustered logit: sitting + regime + item difficulty",
         np.column_stack(base + [work[ITEM_DIFFICULTY_COL].to_numpy(float)])),
    ]:
        res = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        rows.append({"test": label, "n_units": len(set(groups)),
                     "statistic": float(np.exp(res.params[1])),
                     "p_value": float(res.pvalues[1]),
                     "effect": float(res.params[1]),
                     "effect_is": "log-odds per sitting (statistic = odds ratio)"})

    return pd.DataFrame(rows)


def plot_session_trend(df: pd.DataFrame, *, figsize=(12, 4.6), save: Optional[bool] = None, to_paper=None):
    """Two panels across sittings, from `session_summary`: what the participants did,
    and what the model and the confidence rating did.

    Left, participant accuracy against how the same items were answered in L1, by people
    with no sittings at all. The slope the two lines share is the items getting harder;
    the gap between them is what a session effect would have to explain.

    Right, the model's balanced accuracy and AUC, the rating's AUC and the mean
    predicted probability. If the model's discrimination holds while accuracy falls, the
    behaviour stayed readable and only the items changed.
    """
    s = session_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = s["session"].to_numpy()

    axes[0].plot(x, s["participant_accuracy"], marker="o", lw=2, color="#c44e52",
                 label="participant accuracy (KnowQA)")
    axes[0].plot(x, s[ITEM_DIFFICULTY_COL], marker="s", lw=2, ls="--", color="#4c72b0",
                 label="same items, answered in L1")
    for xi, a, b in zip(x, s["participant_accuracy"], s[ITEM_DIFFICULTY_COL]):
        axes[0].annotate(f"{a:.3f}", (xi, a), textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=8)
        axes[0].annotate(f"{b:.3f}", (xi, b), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8)
    axes[0].set_title("Accuracy across sittings, against item difficulty", fontsize=10)
    axes[0].set_ylabel("accuracy")

    for col, colour, name in [("model_balanced_accuracy", "#4c72b0", "model balanced acc"),
                              ("model_auc", "#55a868", "model AUC"),
                              ("confidence_auc", "#dd8452", "self-report AUC"),
                              ("mean_pred_prob", "#8c8c8c", "mean P(correct)")]:
        if col in s.columns:
            axes[1].plot(x, s[col], marker="o", lw=2, color=colour, label=name)
    axes[1].set_title("Model and self-report across sittings", fontsize=10)
    axes[1].set_ylabel("score")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xlabel("sitting (= article batch, they are confounded)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot="session_trend",
        tables={"data": df},
        save=save,
        to_paper=to_paper,
    )
    return fig


# ---------------------------------------------------------------------------
# LexTALE — English proficiency
# ---------------------------------------------------------------------------

# Each participant's LexTALE score, entered by hand from a list in participant-id order.
# The demographics questionnaire runs alongside the experiment, not through this
# pipeline, so there is no export to read: these six values are the only copy.
#
# LexTALE is scored 0-100; Lemhofer & Broersma (2012) read 80 and above as
# upper-intermediate to advanced. Study 2's participants have mixed language
# backgrounds, a stated limitation of that study next to L1's native-speaker sample.
LEXTALE_SCORES: Dict[str, float] = {
    "4000": 93.75,
    "4001": 88.75,
    "4002": 91.25,
    "4003": 81.25,
    "4004": 75.00,
    "4005": 95.00,
}

LEXTALE_COL = "lextale"


def attach_lextale(
    df: pd.DataFrame,
    scores: Mapping[str, float] = LEXTALE_SCORES,
) -> pd.DataFrame:
    """Add each participant's LexTALE score (`lextale`).

    Stops if a participant has no score: they would get NaN and then drop out of any
    correlation with proficiency, shrinking n without a word. Safe to call twice: the
    column is assigned, not merged.
    """
    out = df.copy()
    out[LEXTALE_COL] = out[Con.PARTICIPANT_ID].map(dict(scores))

    missing = sorted(set(out.loc[out[LEXTALE_COL].isna(), Con.PARTICIPANT_ID]))
    assert not missing, (
        f"no LexTALE score for participant(s) {missing}. Add them to LEXTALE_SCORES "
        "rather than letting those trials go unscored."
    )
    return out


def lextale_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: LexTALE score beside their accuracy, overall and per
    regime, then a group-mean row.

    Accuracy is split by regime because proficiency has no reason to act equally across
    them: under full knowledge the participant has just been shown the answer's content,
    leaving little for reading skill to do, while under no knowledge the answers are all
    they have. The mean row is marked by `n_participants`, so it cannot be mistaken for a
    seventh person.
    """
    work = df if LEXTALE_COL in df.columns else attach_lextale(df)

    acc = (
        work.pivot_table(index=Con.PARTICIPANT_ID, columns=REGIME_COL,
                         values=Con.IS_CORRECT_COLUMN, aggfunc="mean")
        .reindex(columns=_ordered_regimes(work))
    )
    out = pd.DataFrame({
        LEXTALE_COL: work.groupby(Con.PARTICIPANT_ID)[LEXTALE_COL].first(),
        "n": work.groupby(Con.PARTICIPANT_ID).size(),
        "participant_accuracy": work.groupby(Con.PARTICIPANT_ID)[Con.IS_CORRECT_COLUMN].mean(),
    }).join(acc).reset_index()

    mean_row = {Con.PARTICIPANT_ID: "mean", "n_participants": len(out)}
    for c in out.columns:
        if c != Con.PARTICIPANT_ID:
            mean_row[c] = out[c].mean()
    return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)


def lextale_correlations(df: pd.DataFrame, *, method: str = "spearman") -> pd.DataFrame:
    """LexTALE against accuracy across participants, overall and per regime: six points
    per row.

    Six participants is far too few to test this. The rows exist so the question is
    asked in the same place as the data grows; read the coefficients as description and
    the p-values as meaningless at this n. `method="spearman"` gives Spearman; any other
    value gives Pearson.
    """
    from scipy import stats

    work = df if LEXTALE_COL in df.columns else attach_lextale(df)
    rows = []
    for label, sub in [("all", work)] + [(r, work[work[REGIME_COL] == r])
                                         for r in _ordered_regimes(work)]:
        per = sub.groupby(Con.PARTICIPANT_ID).agg(
            lex=(LEXTALE_COL, "first"), acc=(Con.IS_CORRECT_COLUMN, "mean"))
        fn = stats.spearmanr if method == "spearman" else stats.pearsonr
        res = fn(per["lex"], per["acc"])
        rows.append({REGIME_COL: label, "n_participants": len(per),
                     f"{method}_r": float(res.statistic if hasattr(res, "statistic") else res[0]),
                     "p_value": float(res.pvalue if hasattr(res, "pvalue") else res[1])})
    return pd.DataFrame(rows)


def plot_lextale(df: pd.DataFrame, *, figsize=(12, 4.4)):
    """LexTALE beside the participants' accuracy. Returns the figure; does not save it.

    Left: each participant's score against the group mean, so who is high or low reads at
    a glance. Right: score against accuracy, one series per regime plus all trials:
    whether the people who read English better are the ones who answer correctly, and
    whether that depends on how much they were told.
    """
    s = lextale_summary(df)
    people = s[s[Con.PARTICIPANT_ID] != "mean"]
    mean_score = float(people[LEXTALE_COL].mean())
    regimes = _ordered_regimes(df)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    order = people.sort_values(LEXTALE_COL)
    axes[0].bar(order[Con.PARTICIPANT_ID], order[LEXTALE_COL], color="#4c72b0")
    axes[0].axhline(mean_score, ls="--", c="k", lw=1.2)
    axes[0].text(len(order) - 0.4, mean_score + 0.9, f"mean {mean_score:.2f}",
                 ha="right", fontsize=9)
    for x, v in zip(range(len(order)), order[LEXTALE_COL]):
        axes[0].text(x, v + 0.9, f"{v:g}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("LexTALE score")
    axes[0].set_xlabel("participant (sorted)")
    axes[0].set_title("English proficiency", fontsize=10)

    for r in regimes:
        axes[1].scatter(people[LEXTALE_COL], people[r], s=70, label=r,
                        color=_REGIME_COLOURS.get(r, "#8c8c8c"))
    axes[1].scatter(people[LEXTALE_COL], people["participant_accuracy"], s=90, marker="D",
                    facecolor="none", edgecolor="k", label="all trials")
    for _, row in people.iterrows():
        axes[1].annotate(row[Con.PARTICIPANT_ID], (row[LEXTALE_COL], row["participant_accuracy"]),
                         textcoords="offset points", xytext=(6, -3), fontsize=7)
    axes[1].set_xlabel("LexTALE score")
    axes[1].set_ylabel("participant accuracy")
    axes[1].set_title("Proficiency against correctness (6 points per series)", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return fig
