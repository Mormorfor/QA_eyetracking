"""KnowQA data showcase — what the collected Study 2 data looks like, and how the
L1-trained correctness model transfers to it, split by knowledge regime.

Descriptive counterpart to :mod:`comparison_runs`. That module runs the standard
train-L1 / test-new bundle once *per regime*; this one fits the model **once** and
attaches per-trial predictions, because the questions here — mean predicted
probability per regime, error composition per regime, what the model says about
lucky guesses — are all group-bys over per-trial probabilities rather than separate
evaluations.

``regime``, ``session_id`` and ``confidence_rating`` live in the raw IA report and
are dropped during feature building, so they are re-attached on
``(participant_id, TRIAL_INDEX)``. Practice trials are already absent from the
model-ready table; the merge is left-from-features, so they drop out on their own.

Two transfer caveats that the numbers here do not show on their own:

* **Pupil z-scores are baselined differently on the two sides.** L1 z-scores per
  participant; KnowQA per ``session_id`` (``know_qa_dataprep``), because one person
  sits for several sessions. ``mean_max_fix_pupil_size_z__correct`` is in the
  headline feature set, so that feature is not strictly the same quantity across
  the two datasets.
* **The scaler is L1's.** ``TrialLevelLogRegModel`` fits its ``StandardScaler`` on
  the training frame and only ``transform``s the test frame, so KnowQA features are
  standardized against L1's means and SDs. That is what transfer means here, but it
  does mean a KnowQA-specific shift shows up as a shifted feature, not a rescaled one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src import constants as Con
from src.data_paths import (
    KNOW_QA_FEATURES_PATH,
    KNOW_QA_IA_ANSWERS_PATH,
    READY_ALL_FEATURES_PATH,
)
from src.derived.correctness_measures import wilson_ci
from src.predictive_modeling.answer_correctness.model_data import load_all_features
from src.predictive_modeling.answer_correctness.models.logreg_model import (
    TrialLevelLogRegModel,
)
import src.predictive_modeling.answer_correctness.feature_groups as fg

# Explicit regime order, no -> partial -> full (increasing knowledge). Stated rather
# than discovered, so figures and tables keep this order regardless of the data.
REGIME_ORDER: tuple[str, ...] = ("no knowledge", "partial knowledge", "full knowledge")

REGIME_COL = "regime"
SESSION_COL = Con.SESSION_ID
CONFIDENCE_COL = "confidence_rating"

# Trial-level metadata carried by the IA report but dropped during feature building.
_META_COLS = (REGIME_COL, SESSION_COL, CONFIDENCE_COL)

# The paper's best cross-validated feature set: the ten hand-picked attention
# features plus the two last-fixation-before-confirm indicators.
SHOWCASE_FEATURE_COLS: List[str] = list(fg.SELECT_1_COLS) + list(fg.LAST_CONFIRM_COMPACT)

PROB_COL = "pred_prob"
PRED_COL = "pred_is_correct"

DataFrameOrPath = Union[pd.DataFrame, str, Path]


def _resolve(frame_or_path: DataFrameOrPath) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path
    return load_all_features(frame_or_path)


def load_showcase_frame(
    features: DataFrameOrPath = KNOW_QA_FEATURES_PATH,
    ia_source: DataFrameOrPath = KNOW_QA_IA_ANSWERS_PATH,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Model-ready KnowQA features with regime, session and confidence re-attached.

    Asserts that every feature-table trial picks up a regime: an unmatched trial
    would otherwise sit in the frame with a NaN regime, silently vanish from every
    per-regime group-by, and still be counted in the pooled totals.
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

    # Same normalization comparison_runs.attach_regime applies, so regime labels are
    # interchangeable between the two modules.
    ia[Con.PARTICIPANT_ID] = ia[Con.PARTICIPANT_ID].astype(str).str.strip().str.lower()
    ia[REGIME_COL] = ia[REGIME_COL].astype(str).str.strip().str.lower()

    meta = ia.drop_duplicates(subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID])

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
    """Regimes present in `df`, in REGIME_ORDER; anything unexpected is appended."""
    present = list(df[REGIME_COL].dropna().unique())
    known = [r for r in REGIME_ORDER if r in present]
    return known + sorted(r for r in present if r not in REGIME_ORDER)


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: sessions, trials, and trials in each regime."""
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
    """n / k / proportion with a Wilson interval, for a binary `value_col`."""
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
    """Accuracy with Wilson intervals, per regime (optionally split further by `by`).

    A pooled ``all`` row is prepended so the overall rate sits beside the regimes.
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


# Answer-screen pace measures. KnowQA has no paragraph screen, so every reading-speed
# quantity here is about the question + four answer options.
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
    """Mean / SD / median of each pace measure, per regime, plus a pooled row."""
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
    """Mean confidence per regime, split by whether the answer was actually correct."""
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
    """Share of trials at each 1-5 confidence rating, per regime (rows sum to 1)."""
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
    """Fit the correctness model on the whole of L1 — no split, this is the trained
    model the paper transfers to Study 2."""
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
    """Add `pred_prob` and `pred_is_correct` to a KnowQA frame.

    Reports how many feature cells the model's ``fill_value`` imputes on this side.
    That fill is a stated modelling choice (``todo.md`` T3.14), not a measurement, so
    the count belongs with the numbers it produced rather than in the model's
    internals.
    """
    cols = list(feature_cols)

    if verbose:
        na = df[cols].isna().sum()
        na = na[na > 0]
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
    y, p = sub[Con.IS_CORRECT_COLUMN].astype(int), sub[PRED_COL].astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    # Balanced accuracy is the mean of sensitivity and specificity, so it needs BOTH
    # outcome classes present. With one class the missing half is not zero and not the
    # other half -- it is undefined, and averaging over the one that exists would
    # report sensitivity under a name that means something else. Stay NaN.
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
    """Transfer performance and error composition per regime, plus a pooled row."""
    rows = []
    for label, sub in [("all", df)] + [(r, df[df[REGIME_COL] == r]) for r in _ordered_regimes(df)]:
        rows.append({REGIME_COL: label, **_confusion_row(sub)})
    return pd.DataFrame(rows)


def probability_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Predicted P(correct) per regime, split by the actual outcome.

    ``mean_prob_correct`` on the **no knowledge** regime is the key dissociation:
    those trials were answered correctly, but without any context to answer from, so
    a model reading behaviour rather than outcome should not be confident about them.
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
    """Split each regime's mean predicted probability into composition vs. level.

    The regimes differ in mean P(correct) for two quite different reasons: the model
    may read behaviour in a regime differently (a *level* effect), or the regime may
    simply contain more wrong trials, which the model scores low in every regime (a
    *composition* effect). ``counterfactual_mean`` re-mixes this regime's own
    correct/wrong probability levels at the **pooled** accuracy, so the remaining gap
    from the pooled mean is the part composition does not explain.
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
# Figures. Each returns the Figure so the caller decides whether to persist it.
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402  (kept with the plotting section)

_REGIME_COLOURS = {
    "no knowledge": "#c44e52",
    "partial knowledge": "#dd8452",
    "full knowledge": "#4c72b0",
    "all": "#8c8c8c",
}


def _colours(labels: Sequence[str]) -> List[str]:
    return [_REGIME_COLOURS.get(l, "#8c8c8c") for l in labels]


def plot_accuracy_by_regime(summary: pd.DataFrame, *, figsize=(7, 4.2)):
    """Bar chart of accuracy with Wilson intervals, from `correctness_by_regime`."""
    s = summary[summary[REGIME_COL] != "all"] if "all" in set(summary[REGIME_COL]) else summary
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(s))
    err = [s["accuracy"] - s["ci_low"], s["ci_high"] - s["accuracy"]]
    ax.bar(x, s["accuracy"], color=_colours(s[REGIME_COL]), yerr=err, capsize=5)
    ax.axhline(0.25, ls=":", c="grey", lw=1)
    ax.text(len(s) - 0.4, 0.26, "chance (1 of 4)", fontsize=8, color="grey", ha="right")
    for i, (_, r) in enumerate(s.iterrows()):
        ax.text(i, r["accuracy"] + 0.035, f"{r['accuracy']:.3f}\nn={r['n']}",
                ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(s[REGIME_COL])
    ax.set_ylim(0, 1.12); ax.set_ylabel("accuracy")
    ax.set_title("KnowQA accuracy by knowledge regime (Wilson 95% CI)")
    fig.tight_layout()
    return fig


def plot_pace_by_regime(df: pd.DataFrame, measure: str = "total_answering_RT",
                        *, figsize=(7, 4.2)):
    """Box plot of one pace measure across regimes."""
    regimes = _ordered_regimes(df)
    data = [df.loc[df[REGIME_COL] == r, measure].dropna() for r in regimes]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, labels=regimes, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], _colours(regimes)):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel(SPEED_COLS.get(measure, measure))
    ax.set_title(f"{SPEED_COLS.get(measure, measure)} by regime")
    fig.tight_layout()
    return fig


def plot_confidence_distribution(df: pd.DataFrame, *, figsize=(7, 4.2)):
    """Grouped bars: share of trials at each 1-5 confidence rating, per regime."""
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
    return fig


def plot_probability_by_regime(df: pd.DataFrame, *, split_by_outcome: bool = True,
                               figsize=(8, 4.5)):
    """Predicted P(correct) distribution per regime, optionally split by outcome.

    With ``split_by_outcome`` the point of the figure is the comparison *within* a
    column: if the model read knowledge rather than outcome, no-knowledge correct
    trials (lucky guesses) should sit lower than full-knowledge correct trials.
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
    return fig


# ---------------------------------------------------------------------------
# Item structure — why two participants can post identical accuracies
# ---------------------------------------------------------------------------


def item_correctness(df: pd.DataFrame, *, regime: Optional[str] = None) -> pd.DataFrame:
    """Per-item accuracy, confidence and model probability, hardest first.

    ``text_id_with_q`` is the item key (paragraph + question). With six participants
    an item is seen by at most a handful of people, so ``n`` belongs beside every rate.
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
    """How much item overlap each pair of participants has in one regime, and how
    often they answered the shared items the same way.

    The counterbalancing assigns whole item sets to regimes, so two participants can
    receive *identical* item sets. When they do, their accuracies are no longer
    independent draws — shared item difficulty removes a large part of the between-
    participant variance, which is what makes matching totals much less surprising
    than they look. ``same_outcome`` separates that from outright duplicated data:
    genuinely distinct participants agree on most items but not all.
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
    """Reshape the showcase frame into the tidy per-(run, trial) frame that
    :mod:`confidence_correlation` consumes.

    That module was written against ``comparison_runs.run_regime_split_comparison``,
    which fits once per regime; this frame comes from a single fit. The predictions are
    identical either way — the model never sees the regime — so the whole of
    ``correlations_by_run`` / ``correlations_by_participant`` /
    ``summarize_by_confidence_level`` and its figures apply unchanged.

    The pooled ``all`` block duplicates the per-regime rows by design: that module
    treats the pooled run as its own group.
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
    """ROC AUC, or NaN when the outcome has a single class (undefined, not 0.5)."""
    from sklearn.metrics import roc_auc_score

    ok = score.notna() & target.notna()
    y = target[ok]
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, score[ok]))


def discrimination_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Does the model separate correct from wrong better than the self-report does?

    The paper's claim is that P(correct) can stand in for an unreliable confidence
    rating. That is a statement about *discrimination*, so both are scored the same
    way: AUC for predicting ``is_correct``. ``auc_model`` and ``auc_confidence`` are
    then directly comparable, and ``auc_combined`` (their mean rank) says whether they
    carry the same information or add to each other.
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
    """One row per participant x `by` (regime by default, or SESSION_COL).

    ``model_accuracy`` is how often the model's predicted label matched the outcome;
    ``mean_pred_prob`` is what it believed on average. They answer different questions
    and can move in opposite directions.

    Prefer ``model_balanced_accuracy`` over ``model_accuracy`` when comparing cells:
    the base rate here runs from ~0.3 to 1.0, so plain accuracy mostly reports how
    often that participant was right rather than how well the model read them. Balanced
    accuracy needs both outcome classes, so it is NaN where a participant made no
    errors in a regime — ``n_wrong`` is carried alongside because in the full-knowledge
    cells it is 0-3 trials, and a metric resting on three trials should be read as such.
    """
    keys = [Con.PARTICIPANT_ID, *by]
    rows = []
    for vals, sub in df.groupby(keys, sort=True):
        vals = vals if isinstance(vals, tuple) else (vals,)
        row = dict(zip(keys, vals))
        row.update({
            "n": len(sub),
            "n_wrong": int((sub[Con.IS_CORRECT_COLUMN] == 0).sum()),
            "accuracy": sub[Con.IS_CORRECT_COLUMN].mean(),
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
                             figsize=(13, 4.2)):
    """One confusion matrix per regime, counts annotated.

    ``normalize="true"`` shades by row (share of each *actual* class), which is what
    makes the panels comparable: the regimes have base rates from 0.42 to 0.97, so raw
    counts alone would say more about the manipulation than about the model. Raw counts
    stay in the annotation.

    Under row normalisation the two diagonal fractions **are** specificity and
    sensitivity, so the panel's balanced accuracy is simply their mean — it is printed
    in the title rather than left for the reader to average.
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

        # Recomputed from this panel's own cells rather than passed in, so the title
        # can never drift from the matrix it sits above.
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
    return fig


def plot_probability_histograms(df: pd.DataFrame, *, bins: int = 20,
                                split_by_outcome: bool = True, figsize=(13, 3.8)):
    """Predicted P(correct) histogram per regime.

    Split by outcome, the question each panel answers is whether the two humps are in
    different places — i.e. whether the model separates correct from wrong *within*
    that regime, rather than just inheriting the regime's base rate.
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
    return fig


def plot_participant_profiles(df: pd.DataFrame, *,
                              measures: Sequence[str] = ("accuracy", "mean_confidence",
                                                         "mean_pred_prob", "median_RT_ms"),
                              figsize=(12, 7)):
    """Small multiples: one panel per measure, participants on x, a line per regime.

    Six participants is too few to test anything, so this is for spotting *shape* —
    whether a person is uniformly low, or low only in one regime, and whether their
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
    return fig


# ---------------------------------------------------------------------------
# Figures for the correlation / comparison tables
# ---------------------------------------------------------------------------

from src.viz.viz_helpers import p_to_stars  # noqa: E402


def _run_order_local(runs: Sequence[str]) -> List[str]:
    """Pooled 'all' first, then REGIME_ORDER, then anything unexpected."""
    out = [r for r in runs if r == "all"]
    out += [r for r in REGIME_ORDER if r in runs]
    return out + sorted(r for r in runs if r != "all" and r not in REGIME_ORDER)


def plot_correlations_by_run(corr_df: pd.DataFrame, *, method: str = "spearman",
                             figsize=(9, 4.6)):
    """Grouped bars of `correlations_by_run` - one cluster per run, one bar per comparison.

    The two comparisons have to be read together. ``is_correct ~ confidence`` is the
    reference - how calibrated the self-report is in that condition - and
    ``pred_prob ~ confidence`` is only interpretable against it. A regime where both are
    near zero says the self-report carried no signal there, not that the model failed.
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
    return fig


def plot_participant_correlation_forest(
    by_model: pd.DataFrame,
    by_correct: Optional[pd.DataFrame] = None,
    *,
    method: str = "spearman",
    figsize=(8, 4.6),
):
    """Per-participant correlation dot plot - the within-subject version of the bars.

    Confidence scales are personal, so the pooled r mixes a within-subject effect with
    between-subject scale use. One row per participant isolates the former; plotting
    both series together shows whether a participant whose self-report tracks their own
    correctness is also one the model agrees with.
    """
    r_col, p_col = f"{method}_r", f"{method}_p"
    pids = list(by_model[Con.PARTICIPANT_ID])
    y = np.arange(len(pids))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(by_model[r_col], y, s=70, color="#4c72b0", zorder=3,
               label="pred_prob ~ confidence")
    for yi, r, p in zip(y, by_model[r_col], by_model[p_col]):
        ax.text(r, yi + 0.18, p_to_stars(p), ha="center", fontsize=7, color="#4c72b0")

    if by_correct is not None:
        m = by_correct.set_index(Con.PARTICIPANT_ID).reindex(pids)
        ax.scatter(m[r_col], y, s=70, marker="D", color="#dd8452", zorder=3,
                   label="is_correct ~ confidence")
        for yi, r in zip(y, m[r_col]):
            if np.isfinite(r):
                ax.plot([by_model[r_col].iloc[yi], r], [yi, yi], c="grey", lw=1, zorder=1)

    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(pids)
    ax.set_ylim(-0.6, len(pids) - 0.1)   # room for the stars above the top row
    ax.set_xlabel(f"{method} r")
    ax.set_ylabel("participant")
    ax.set_title("Within-participant confidence correlations")
    ax.grid(axis="x", alpha=0.25)
    # Anchored outside the axes: loc="best" lands on the points when the dots cluster.
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    return fig


def plot_discrimination_roc(df: pd.DataFrame, *, figsize=(13, 4.2)):
    """ROC curves per regime: the model's P(correct) against the self-report.

    The AUC table says which discriminates better; the curves say *where*. A
    self-report that hugs the diagonal near the origin is one that never assigns low
    confidence to the trials it gets wrong.
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
    return fig


def plot_transfer_metrics(metrics: pd.DataFrame, *,
                          measures: Sequence[str] = ("accuracy", "balanced_accuracy",
                                                     "sensitivity", "specificity"),
                          figsize=(9.5, 4.4)):
    """Grouped bars of the transfer metrics, with each regime's base rate marked.

    The dashed line is the regime's own accuracy (``n_correct / n``) — i.e. the score a
    trivial "always predict correct" classifier would get there. It is drawn because
    raw accuracy is not interpretable without it: base rates run from 0.42 to 0.97, and
    under ``class_weight="balanced"`` the model deliberately gives up majority-class
    accuracy to catch the minority class, so in the high-base-rate regimes its accuracy
    sits *below* the line while its specificity stays ~0.91. Balanced accuracy is the
    bar to read; the marker is there to stop the accuracy bar being read as performance.
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
    return fig


def plot_probability_decomposition(dec: pd.DataFrame, *, figsize=(8.5, 4.2)):
    """Dumbbell: observed mean P(correct) vs the same regime re-mixed at pooled accuracy.

    The distance between the two dots is the part composition explains. A regime whose
    dots coincide is one where the model genuinely reads that condition differently.
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
    return fig


def _annotated_heatmap(ax, mat: pd.DataFrame, *, fmt: str = "{:.2f}",
                       cmap: str = "RdYlBu", vmin=None, vmax=None, cbar_label=""):
    """Shared heatmap body: shade `mat`, print every cell, label both axes."""
    im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    lo = vmin if vmin is not None else np.nanmin(mat.values)
    hi = vmax if vmax is not None else np.nanmax(mat.values)
    mid = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="grey")
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if abs(v - mid) > 0.42 * (hi - lo) else "black")
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
    measures: Sequence[str] = ("accuracy", "mean_confidence", "mean_pred_prob",
                               "model_balanced_accuracy"),
    figsize=(14, 3.8),
):
    """Participant x regime heatmap, one panel per measure.

    The table version of this is easy to read row-wise and hard to read column-wise;
    the shading makes a participant who is out of line in one regime only - rather than
    uniformly - visible at a glance.

    The model panel is **balanced** accuracy, because the first panel shows that the
    base rate ranges from 0.32 to 1.00 across these cells: plain accuracy there would
    largely restate panel one. A "-" marks a cell where the participant made no errors,
    so specificity — and therefore balanced accuracy — is undefined.
    """
    prof = participant_profile(df)
    use = [m for m in measures if m in prof.columns]
    regimes = _ordered_regimes(df)

    fig, axes = plt.subplots(1, len(use), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, use):
        mat = (prof.pivot(index=Con.PARTICIPANT_ID, columns=REGIME_COL, values=m)
                   .reindex(columns=regimes))
        # Confidence is a 1-5 scale; everything else here is a 0-1 rate.
        vmin, vmax = (1, 5) if m == "mean_confidence" else (0, 1)
        _annotated_heatmap(ax, mat, vmin=vmin, vmax=vmax)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("")
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.suptitle("Per participant, per regime", fontsize=11)
    fig.tight_layout()
    return fig


def plot_shared_item_structure(pairs: pd.DataFrame, *, figsize=(11, 4.4)):
    """Two heatmaps over participant pairs: how many items they shared, and how often
    they answered the shared ones the same way.

    The left panel is the design (counterbalancing pairs participants onto identical
    item sets); the right is the check that they are still different people - a pair
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
    return fig


def plot_item_difficulty(items: pd.DataFrame, *, figsize=(11, 4.4)):
    """Item accuracy: its distribution, and whether the model knew which were hard.

    The scatter is the interesting panel. Items on the diagonal are ones the model
    read the way the participants performed; items low-accuracy but high-probability
    are where it was confidently wrong, and those are worth reading individually.
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
    return fig


def plot_session_trajectories(
    df: pd.DataFrame,
    *,
    measures: Sequence[str] = ("accuracy", "mean_confidence", "median_RT_ms",
                               "mean_pred_prob"),
    figsize=(12, 7),
):
    """One line per participant across their three sittings, one panel per measure.

    Practice and fatigue are confounds for every pace measure, so a participant whose
    time falls monotonically across sessions is learning the interface rather than
    responding to the manipulation. Sessions are ordered by their id, which encodes
    the sitting.
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
    return fig


def plot_pace_panel(df: pd.DataFrame, *, figsize=(12, 6.5)):
    """Every pace measure at once, one panel each, regimes on the x axis.

    Bars are means with a standard-error whisker. These measures live on wildly
    different scales (milliseconds against a count of transitions), which is why this
    is small multiples rather than one grouped chart.
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
    return fig


def plot_confidence_by_regime(conf: pd.DataFrame, *, figsize=(8.5, 4.2)):
    """Mean confidence per regime, split by whether the answer was actually right.

    The vertical gap inside a regime is that condition's calibration: a participant who
    can tell when they are wrong shows two clearly separated bars.
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
    return fig


# ---------------------------------------------------------------------------
# Session order: is there a drift across the three sittings?
# ---------------------------------------------------------------------------
#
# READ THIS BEFORE INTERPRETING ANYTHING BELOW.
#
# Session order is **perfectly confounded with article batch** in the data as
# collected: every participant took batch 1 in sitting 1, batch 2 in sitting 2 and
# batch 3 in sitting 3, and the three batches share no items at all. So a decline
# across sittings has two indistinguishable explanations -- the participant tiring, or
# the later batches simply being harder -- and no number of additional participants
# separates them while the batch order stays fixed. Counterbalancing batch order is the
# only thing that would.
#
# What *can* be done without that is to ask how hard these items are for somebody with
# no sittings at all. Every KnowQA item also appears in L1, where it was seen under a
# completely different design, so L1's accuracy on the same items is an item-difficulty
# baseline that carries no session effect by construction. That is what
# `attach_item_difficulty` supplies and what the tests below control for.

SESS_ORDER_COL = "sess_order"
ITEM_DIFFICULTY_COL = "l1_item_acc"


def add_session_order(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sess_order` — 1, 2, 3 — ranking each participant's sittings.

    Ranked on ``session_id``, whose suffix encodes the (batch, list) of the sitting, so
    the ordering is the order they were run in.
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
    """Attach each item's **L1** accuracy as a session-free difficulty baseline.

    Asserts full coverage: a KnowQA item with no L1 counterpart would arrive as NaN and
    then be dropped from any model that conditions on difficulty, quietly changing which
    trials the session estimate is computed over.
    """
    l1 = _resolve(l1_source)
    diff = l1.groupby(Con.TEXT_ID_WITH_Q_COLUMN)[Con.IS_CORRECT_COLUMN].mean()
    out = df.merge(diff.rename(ITEM_DIFFICULTY_COL), left_on=Con.TEXT_ID_WITH_Q_COLUMN,
                   right_index=True, how="left")

    missing = int(out[ITEM_DIFFICULTY_COL].isna().sum())
    assert missing == 0, (
        f"{missing} KnowQA trial(s) have no L1 counterpart item, so no difficulty "
        "baseline. Do not drop them silently — decide what they should be."
    )
    return out


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per sitting: what the participants did, and what the model made of it.

    `l1_item_acc` is the session-free item-difficulty baseline — how the *same items*
    were answered in L1. Read the participant column against it: a drop that the L1
    column matches is the items getting harder, not the participants.
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
    """Is the decline across sittings real? Three tests, none of them conclusive at n=6.

    All three respect the clustering — trials are nested in six participants, so a test
    treating 870 trials as independent would be badly anti-conservative.

    * **Wilcoxon, sitting 1 vs 3** on the six per-participant accuracies. With six pairs
      the smallest attainable p is 0.031, so this cannot reach 0.01 however large the
      effect.
    * **Per-participant trend** — one Spearman rho per person between sitting and
      outcome, then a one-sample test on those six values.
    * **Clustered logistic regression**, with and without the L1 item-difficulty
      covariate. The comparison between the two rows is the point: if the session
      coefficient shrinks once difficulty is controlled, the decline was mostly the
      items.
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

    # Explicit design matrix rather than a formula: `regime` is a pandas StringDtype
    # column and patsy cannot interpret that dtype.
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


def plot_session_trend(df: pd.DataFrame, *, figsize=(12, 4.6)):
    """Two panels: what the participants did, and what the model did, across sittings.

    Left, the participant's accuracy is drawn against the L1 item-difficulty baseline —
    how the same items were answered by a sample with no sittings at all. The gap
    between the two lines is the part a session effect would have to explain; the slope
    they share is the items getting harder.

    Right, the model's own numbers. If its discrimination holds while accuracy falls,
    the behaviour stayed legible and only the items changed.
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
    return fig
