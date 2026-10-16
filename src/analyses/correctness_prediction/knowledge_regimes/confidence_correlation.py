"""Does the model's predicted probability track the participant's own confidence?

Study 2 asks after every trial how confident the participant is in the answer they just
gave (`confidence_rating`, 1 to 5). This module sets that rating against the model's
predicted P(correct) for the same trial, and asks three questions, which can disagree:

* **model vs confidence** -- does P(correct) rise with confidence? The headline: whether
  the eye-movement model picks up what the participant reports.
* **confidence vs correctness** -- is the rating itself calibrated? Without this
  reference, a null result above cannot be read: the rating may simply be noise.
* **within participant** -- confidence scales are personal (one person's "4" is another's
  "2"), so a pooled correlation mixes the effect within each person with differences in
  how people use the scale. Correlations computed per participant keep only the first.

Each is reported per run: each knowledge regime, plus all trials pooled (`all`).

The per-trial predictions come from `comparison_runs.run_regime_split_comparison` (via
`collect_prediction_frame`) or from `descriptives.to_prediction_frame`; both give one row
per run and trial. `run_confidence_correlation` does the whole analysis in one call. Its
default confidence source is the first pilot's IA report, like `comparison_runs`'s default
data; pass KnowQA's for the real data. The plot functions return `(figure, paths
written)`; with `save` on, they write the figure and its table to
`reports/correctness_prediction/knowledge_regimes/`.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import columns as Con
from src.config.datasets import NEW_EXP_IA_ANSWERS_PATH
from src.lib.plotting.output import save_output

from src.analyses.correctness_prediction.knowledge_regimes.comparison_runs import (
    POOLED_LABEL,
    DataFrameOrPath,
    run_regime_split_comparison,
)

# The confidence rating in Study 2's IA report: one per trial, on a 1-5 scale.
CONFIDENCE_COL = "confidence_rating"

# Column the per-trial prediction frame carries the model's P(correct) in.
PRED_PROB_COL = "pred_prob"

# Correlations reported for every run: Pearson (linear), Spearman (rank) and Kendall
# (rank, corrected for ties).
CORR_METHODS = ("pearson", "spearman", "kendall")

# Fewest complete pairs a correlation is computed from: below this the estimate means
# nothing, and scipy warns or returns NaN.
MIN_PAIRS = 3

# Colour per run label, so a regime keeps its colour across this module's figures.
_RUN_COLORS = {
    POOLED_LABEL: "#4d4d4d",
    "no knowledge": "#de2d26",
    "partial knowledge": "#e6a700",
    "full knowledge": "#2c7fb8",
}
_FALLBACK_COLOR = "#7570b3"


def _run_color(label: str) -> str:
    return _RUN_COLORS.get(label, _FALLBACK_COLOR)


# ---------------------------------------------------------------------------
# Loading the self-report
# ---------------------------------------------------------------------------


def load_confidence_map(
    ia_source: DataFrameOrPath,
    *,
    confidence_col: str = CONFIDENCE_COL,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
) -> pd.DataFrame:
    """One `confidence_rating` per trial, from the IA report (a DataFrame or a CSV path).

    The IA report repeats the rating on every word row of a trial; the first is kept.
    `participant_id` is stripped and lower-cased, as Study 2's cleaning does, so it
    matches the feature tables. A rating that is missing or not a number is dropped, so
    that trial is absent from the result.
    """
    cols = [participant_col, trial_col, confidence_col]

    if isinstance(ia_source, pd.DataFrame):
        ia = ia_source[cols].copy()
    else:
        ia = pd.read_csv(ia_source, low_memory=False, usecols=cols)

    ia[participant_col] = ia[participant_col].astype(str).str.strip().str.lower()
    ia[confidence_col] = pd.to_numeric(ia[confidence_col], errors="coerce")

    return (
        ia.dropna(subset=[confidence_col])
        .drop_duplicates(subset=[participant_col, trial_col])
        .reset_index(drop=True)
    )


def attach_confidence(
    df: pd.DataFrame,
    confidence_source: Optional[DataFrameOrPath],
    *,
    confidence_col: str = CONFIDENCE_COL,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
) -> pd.DataFrame:
    """`df` with a numeric `confidence_rating` column.

    If `df` already has the column, it is only turned into numbers (anything that is not
    a number becomes NaN). Otherwise it is merged in from `confidence_source`, on
    `(participant_id, TRIAL_INDEX)`; every trial of `df` is kept, and one with no rating
    gets NaN.
    """
    out = df.copy()

    if confidence_col in out.columns:
        out[confidence_col] = pd.to_numeric(out[confidence_col], errors="coerce")
        return out

    if confidence_source is None:
        raise ValueError(
            f"'{confidence_col}' is not in the frame and no confidence_source was "
            "given to merge it from."
        )

    conf_map = load_confidence_map(
        confidence_source,
        confidence_col=confidence_col,
        participant_col=participant_col,
        trial_col=trial_col,
    )
    return out.merge(conf_map, on=[participant_col, trial_col], how="left")


# ---------------------------------------------------------------------------
# Pulling predictions out of the regime-split runs
# ---------------------------------------------------------------------------


def collect_prediction_frame(
    regime_result: Dict[str, Any],
    *,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    """Every run of a `run_regime_split_comparison` result as one frame, one row per run
    and trial.

    Columns: `run` (the regime, or `all` for the pooled run), the trial keys, `y_true`,
    `y_pred`, `pred_prob` (the model's P(correct)) and `is_correct` (a copy of
    `y_true`). The pooled run repeats every regime's trials on purpose: it is analysed as
    a group of its own.

    Predictions are matched to trials by position. That is safe because the bundle
    predicts every row of its `test_df`, in order
    (`evaluate_single_model_on_prepared_split`).
    """
    frames: List[pd.DataFrame] = []

    for label, out in regime_result["outputs"].items():
        res = out["results"][next(iter(out["results"]))]
        test_df = out["test_df"]

        frame = pd.DataFrame(
            {
                "run": label,
                participant_col: test_df[participant_col].to_numpy(),
                trial_col: test_df[trial_col].to_numpy(),
                "y_true": np.asarray(res.y_true).reshape(-1),
                "y_pred": np.asarray(res.y_pred).reshape(-1),
                PRED_PROB_COL: np.asarray(res.y_prob, dtype=float).reshape(-1),
            }
        )
        if target_col not in frame.columns:
            frame[target_col] = frame["y_true"]
        frames.append(frame)

    if not frames:
        raise ValueError("regime_result['outputs'] is empty — nothing to analyse.")

    return pd.concat(frames, ignore_index=True)


def _run_order(runs: Sequence[str]) -> List[str]:
    """The pooled run first, then the others in the order they appear."""
    ordered = [r for r in runs if r == POOLED_LABEL]
    ordered += [r for r in runs if r != POOLED_LABEL]
    return ordered


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


def _correlate(x: np.ndarray, y: np.ndarray, method: str):
    """`(r, p)` by `method`, or `(nan, nan)` with too few pairs or a side that never varies."""
    if len(x) < MIN_PAIRS or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan

    fn = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[method]
    out = fn(x, y)
    return float(out[0]), float(out[1])


def _correlation_row(
    frame: pd.DataFrame,
    *,
    label_cols: Dict[str, Any],
    x_col: str,
    y_col: str,
    methods: Sequence[str],
) -> Dict[str, Any]:
    """One table row for `x_col` against `y_col`: n, both means, and every method's r and
    p, over the trials that have both values.

    The means are called `mean_x` / `mean_y` so that rows with different `y_col`s
    (predicted probability, actual correctness) stack into one table.
    """
    pair = frame[[x_col, y_col]].dropna()
    x = pair[x_col].to_numpy(dtype=float)
    y = pair[y_col].to_numpy(dtype=float)

    row: Dict[str, Any] = dict(label_cols)
    row["n"] = int(len(pair))
    row["mean_x"] = float(x.mean()) if len(x) else np.nan
    row["mean_y"] = float(y.mean()) if len(y) else np.nan

    for method in methods:
        stat, p = _correlate(x, y, method)
        row[f"{method}_r"] = stat
        row[f"{method}_p"] = p

    return row


def correlations_by_run(
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    target_col: str = Con.IS_CORRECT_COLUMN,
    methods: Sequence[str] = CORR_METHODS,
) -> pd.DataFrame:
    """Two rows of correlations per run: `pred_prob ~ confidence`, the headline, and
    `is_correct ~ confidence`, the reference that says whether the rating tracks
    correctness at all in that run.
    """
    rows: List[Dict[str, Any]] = []

    for run in _run_order(pred_df["run"].unique().tolist()):
        sub = pred_df.loc[pred_df["run"] == run]

        rows.append(
            _correlation_row(
                sub,
                label_cols={"run": run, "comparison": "pred_prob ~ confidence"},
                x_col=confidence_col,
                y_col=prob_col,
                methods=methods,
            )
        )
        rows.append(
            _correlation_row(
                sub,
                label_cols={"run": run, "comparison": "is_correct ~ confidence"},
                x_col=confidence_col,
                y_col=target_col,
                methods=methods,
            )
        )

    return pd.DataFrame(rows)


def correlations_by_participant(
    pred_df: pd.DataFrame,
    *,
    run: str = POOLED_LABEL,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    participant_col: str = Con.PARTICIPANT_ID,
    methods: Sequence[str] = CORR_METHODS,
) -> pd.DataFrame:
    """`pred_prob ~ confidence` within each participant, for one run: one row per person.

    Confidence scales are personal, so a pooled correlation mixes the effect within each
    person with differences in how people use the scale; this keeps only the first. Pass
    `prob_col="is_correct"` for the confidence-vs-correctness version. A participant who
    gave the same rating on every trial gets NaN: there is no variation to correlate.
    """
    sub = pred_df.loc[pred_df["run"] == run]
    rows = [
        _correlation_row(
            part,
            label_cols={"run": run, participant_col: pid},
            x_col=confidence_col,
            y_col=prob_col,
            methods=methods,
        )
        for pid, part in sub.groupby(participant_col, sort=True)
    ]
    return pd.DataFrame(rows)


def summarize_by_confidence_level(
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    """Per run and confidence rating: trials, mean / median / SD of predicted P(correct),
    and actual accuracy.

    The direct read of the headline question: if the model tracks confidence, mean
    `pred_prob` rises with the rating. `accuracy` beside it shows whether the
    participants were in fact right more often when they said they were more confident.
    """
    rows: List[Dict[str, Any]] = []

    for run in _run_order(pred_df["run"].unique().tolist()):
        sub = pred_df.loc[pred_df["run"] == run].dropna(subset=[confidence_col])
        for level, grp in sub.groupby(confidence_col, sort=True):
            rows.append(
                {
                    "run": run,
                    confidence_col: level,
                    "n": int(len(grp)),
                    "mean_pred_prob": float(grp[prob_col].mean()),
                    "median_pred_prob": float(grp[prob_col].median()),
                    "std_pred_prob": float(grp[prob_col].std(ddof=1))
                    if len(grp) > 1
                    else np.nan,
                    "accuracy": float(grp[target_col].mean()),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _grid(n: int, ncols: int, figsize_per: tuple):
    """A figure with `n` panels, `ncols` to a row, the unused ones hidden."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def plot_confidence_vs_prob_scatter(
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    target_col: str = Con.IS_CORRECT_COLUMN,
    title_prefix: str = "",
    ncols: int = 2,
    jitter: float = 0.12,
    random_state: int = 42,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "confidence_vs_predicted_probability_scatter",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Predicted P(correct) against confidence, one panel per run, one point per trial.

    Confidence is jittered sideways so overlapping trials stay visible. Filled markers
    are correct answers, hollow ones wrong. The dashed line is the least-squares fit
    through the un-jittered points, and each title gives the panel's Spearman rho.
    """
    runs = _run_order(pred_df["run"].unique().tolist())
    fig, axes = _grid(len(runs), ncols, (6.0, 4.5))
    rng = np.random.default_rng(random_state)

    for ax, run in zip(axes, runs):
        sub = pred_df.loc[pred_df["run"] == run].dropna(subset=[confidence_col, prob_col])
        conf = sub[confidence_col].to_numpy(dtype=float)
        prob = sub[prob_col].to_numpy(dtype=float)
        correct = sub[target_col].to_numpy() == 1

        x = conf + rng.uniform(-jitter, jitter, size=len(conf))
        color = _run_color(run)

        ax.scatter(
            x[correct], prob[correct], s=38, color=color, alpha=0.75,
            label=f"correct (n={int(correct.sum())})",
        )
        ax.scatter(
            x[~correct], prob[~correct], s=38, facecolors="none", edgecolors=color,
            alpha=0.9, label=f"wrong (n={int((~correct).sum())})",
        )

        if len(conf) >= MIN_PAIRS and not np.all(conf == conf[0]):
            slope, intercept = np.polyfit(conf, prob, 1)
            xs = np.linspace(conf.min(), conf.max(), 50)
            ax.plot(xs, slope * xs + intercept, "--", color=color, linewidth=1.5)

        rho, p = _correlate(conf, prob, "spearman")
        ax.set_title(
            f"{title_prefix}[{run}] — n={len(sub)}, "
            f"Spearman rho={rho:.2f}, p={p:.3f}"
            if np.isfinite(rho)
            else f"{title_prefix}[{run}] — n={len(sub)} (rho n/a)"
        )
        ax.set_xlabel("Self-reported confidence")
        ax.set_ylabel("Predicted P(correct)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    saved = save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot=plot,
        tables={"data": pred_df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
    ).paths
    return fig, saved


def plot_prob_by_confidence_level(
    level_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    title_prefix: str = "",
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "predicted_probability_by_confidence_level",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """Mean predicted P(correct) at each confidence rating, one line per run, from
    `summarize_by_confidence_level`.

    The grey dashed line is the pooled run's actual accuracy at each rating: the curve
    the model would follow if it tracked the same thing the participants report.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in _run_order(level_df["run"].unique().tolist()):
        sub = level_df.loc[level_df["run"] == run].sort_values(confidence_col)
        ax.plot(
            sub[confidence_col], sub["mean_pred_prob"],
            marker="o", color=_run_color(run), label=f"{run} (model)",
        )

    pooled = level_df.loc[level_df["run"] == POOLED_LABEL].sort_values(confidence_col)
    if not pooled.empty:
        ax.plot(
            pooled[confidence_col], pooled["accuracy"],
            marker="s", linestyle="--", color="#999999", label="all (actual accuracy)",
        )

    ax.set_title(f"{title_prefix}Predicted P(correct) by self-reported confidence")
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Mean predicted P(correct)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    plt.tight_layout()

    saved = save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot=plot,
        tables={"data": level_df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
    ).paths
    return fig, saved


def plot_prob_boxplot_by_confidence(
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    title_prefix: str = "",
    ncols: int = 2,
    save: Optional[bool] = None,
    subdir: Optional[str] = None,
    plot: str = "predicted_probability_boxplot_by_confidence",
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
):
    """The spread of predicted P(correct) at each confidence rating, one panel per run.

    A mean in `plot_prob_by_confidence_level` can hide a rating whose trials split into
    two groups rather than shift together; the boxes show the spread behind each point.
    """
    runs = _run_order(pred_df["run"].unique().tolist())
    fig, axes = _grid(len(runs), ncols, (6.0, 4.0))

    for ax, run in zip(axes, runs):
        sub = pred_df.loc[pred_df["run"] == run].dropna(subset=[confidence_col, prob_col])
        levels = sorted(sub[confidence_col].unique())
        data = [sub.loc[sub[confidence_col] == lv, prob_col].to_numpy() for lv in levels]

        if data:
            bp = ax.boxplot(data, labels=[f"{lv:g}\n(n={len(d)})" for lv, d in zip(levels, data)],
                            patch_artist=True, widths=0.6)
            for patch in bp["boxes"]:
                patch.set_facecolor(_run_color(run))
                patch.set_alpha(0.45)
            for median in bp["medians"]:
                median.set_color("black")

        ax.set_title(f"{title_prefix}[{run}]")
        ax.set_xlabel("Self-reported confidence")
        ax.set_ylabel("Predicted P(correct)")
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    saved = save_output(
        fig,
        analysis="correctness_prediction/knowledge_regimes",
        plot=plot,
        tables={"data": pred_df},
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        subdir=subdir,
    ).paths
    return fig, saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_confidence_correlation(
    regime_result: Optional[Dict[str, Any]] = None,
    *,
    confidence_source: Optional[DataFrameOrPath] = NEW_EXP_IA_ANSWERS_PATH,
    confidence_col: str = CONFIDENCE_COL,
    methods: Sequence[str] = CORR_METHODS,
    participant_run: str = POOLED_LABEL,
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: str = "confidence",
    split_tag: str = "trainL1_testnew",
    verbose: bool = True,
    **regime_kwargs: Any,
) -> Dict[str, Any]:
    """The whole analysis in one call: the model's predicted probabilities against the
    confidence ratings, per run and within participant, with three figures.

    `regime_result` is the output of `comparison_runs.run_regime_split_comparison`; pass
    one already computed to reuse its predictions, or `None` to run it first with
    `regime_kwargs`. Each trial's rating is taken from `confidence_source` (an IA report
    or a per-trial map, as a DataFrame or a CSV path). Its default is the first pilot's
    IA report, matching `comparison_runs`'s default data; for KnowQA, pass a
    `regime_result` run on KnowQA and `confidence_source=KNOW_QA_IA_ANSWERS_PATH`.

    Trials with no rating are left out of the correlations, and their number is printed;
    if no trial has one, it stops. The per-participant correlations use `participant_run`
    (default: the pooled run). If there is no run of that name, the first run is used
    instead, without a warning; with `verbose`, the run used is printed.

    With `save`, the figures and tables go to
    `reports/correctness_prediction/knowledge_regimes/`, in the subfolder
    `<split_tag>/logreg/<subdir>`.

    Returns a dict: `correlations` (per run, `pred_prob ~ confidence` and the
    `is_correct ~ confidence` reference), `by_confidence` (per run and rating: trials,
    predicted probability, actual accuracy), `by_participant` (the within-participant
    correlations), `predictions` (the per-(run, trial) frame all of it is computed from),
    `regime_result` (the one passed in, or the one run here), `figures` and `paths`.
    """
    if regime_result is None:
        regime_result = run_regime_split_comparison(
            save=save, close=close, verbose=verbose, **regime_kwargs
        )

    pred_df = collect_prediction_frame(regime_result)
    pred_df = attach_confidence(
        pred_df, confidence_source, confidence_col=confidence_col
    )

    n_missing = int(pred_df[confidence_col].isna().sum())
    if n_missing:
        print(
            f"[confidence] {n_missing}/{len(pred_df)} run-trial rows have no "
            f"'{confidence_col}' and are dropped from the correlations."
        )
    if pred_df[confidence_col].notna().sum() == 0:
        raise ValueError(
            f"No trial matched a '{confidence_col}' value — check that "
            "confidence_source covers the same participants/trials as the features."
        )

    corr_df = correlations_by_run(
        pred_df, confidence_col=confidence_col, methods=methods
    )
    level_df = summarize_by_confidence_level(pred_df, confidence_col=confidence_col)

    runs = pred_df["run"].unique().tolist()
    part_run = participant_run if participant_run in runs else runs[0]
    part_df = correlations_by_participant(
        pred_df, run=part_run, confidence_col=confidence_col, methods=methods
    )

    if verbose:
        headline = corr_df.loc[corr_df["comparison"] == "pred_prob ~ confidence"]
        print("\nPredicted probability vs. self-reported confidence:")
        print(headline.to_string(index=False))
        print(f"\nWithin-participant correlations (run='{part_run}'):")
        print(part_df.to_string(index=False))
        for method in methods:
            col = f"{method}_r"
            if col in part_df.columns:
                print(f"  mean within-participant {method}: {part_df[col].mean():.3f}")

    base_dir = "/".join(x for x in (split_tag or "full_fit", "logreg", subdir) if x)

    fig_scatter, scatter_paths = plot_confidence_vs_prob_scatter(
        pred_df, confidence_col=confidence_col, save=save,
        subdir=base_dir, to_paper=to_paper, dpi=dpi, close=close,
    )
    fig_levels, level_paths = plot_prob_by_confidence_level(
        level_df, confidence_col=confidence_col, save=save,
        subdir=base_dir, to_paper=to_paper, dpi=dpi, close=close,
    )
    fig_box, box_paths = plot_prob_boxplot_by_confidence(
        pred_df, confidence_col=confidence_col, save=save,
        subdir=base_dir, to_paper=to_paper, dpi=dpi, close=close,
    )

    csv_paths = save_output(
        None,
        analysis="correctness_prediction/knowledge_regimes",
        plot="confidence_tables",
        tables={
            "correlations": corr_df,
            "by_level": level_df,
            "by_participant": part_df,
            "trial_predictions": pred_df,
        },
        save=save,
        to_paper=to_paper,
        subdir=base_dir,
    ).tables

    return {
        "correlations": corr_df,
        "by_confidence": level_df,
        "by_participant": part_df,
        "predictions": pred_df,
        "regime_result": regime_result,
        "figures": {
            "scatter": fig_scatter,
            "levels": fig_levels,
            "boxplot": fig_box,
        },
        "paths": {
            "scatter": scatter_paths,
            "levels": level_paths,
            "boxplot": box_paths,
            **csv_paths,
        },
    }
