"""Do the model's predicted probabilities track self-reported confidence?

The new (question-answering) experiment asks the participant, after every trial, how
confident they are in the answer they just gave (``confidence_rating``, 1-5). That
rating is a *trial-level* value which the IA report duplicates across every
fixation/IA row of the trial, so it is collapsed back to one value per
``(participant_id, TRIAL_INDEX)`` before being joined onto the model-ready features.

This module takes the per-regime train-on-L1 / test-on-new-data runs produced by
:func:`..comparison_runs.run_regime_split_comparison`, pulls out the model's predicted
P(correct) for every test trial, attaches the self-reported confidence, and asks
whether the two agree — overall, per knowledge regime, and within participant.

Three separate questions are reported, because they can disagree:

* **model vs. confidence** — does P(correct) rise with the participant's confidence?
  This is the headline: it says whether the eye-movement model is picking up the same
  signal the participant introspects on.
* **confidence vs. correctness** — is the self-report itself calibrated? Without this
  reference, a null model-vs-confidence result is uninterpretable (the confidence
  rating may simply be noise).
* **within-participant** — confidence scales are personal (one participant's "4" is
  another's "2"), so the pooled correlation mixes a within-subject effect with
  between-subject scale-use differences. The per-participant correlations, and their
  mean, isolate the within-subject part.

Everything is parameterized, so the same entry point works for the current
three-participant test run and, unchanged, for the full experiment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import constants as Con
from src.data_paths import NEW_EXP_IA_ANSWERS_PATH
from src.predictive_modeling.common.viz_utils import maybe_save_plot
from src.viz.plot_output import save_df_csv, _answer_correctness_rel_dir

from src.predictive_modeling.answer_correctness.knowledge_regimes_analysis.comparison_runs import (
    POOLED_LABEL,
    DataFrameOrPath,
    run_regime_split_comparison,
)

# Trial-level self-report column in the new-experiment IA report (1-5 Likert).
CONFIDENCE_COL = "confidence_rating"

# Column the per-trial prediction frame carries the model's P(correct) in.
PRED_PROB_COL = "pred_prob"

# Correlation flavours reported for every run: linear, rank, and rank (tie-robust).
CORR_METHODS = ("pearson", "spearman", "kendall")

# Minimum non-null pairs before a correlation is computed at all (below this the
# estimate is meaningless and scipy warns / returns nan).
MIN_PAIRS = 3

# Colour per regime label, so the same condition keeps its colour across figures.
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
    """Collapse the IA report to one ``confidence_rating`` per trial.

    ``ia_source`` is the raw/cleaned IA report (a DataFrame or a CSV path). The rating
    is a trial-level answer duplicated across the trial's fixation/IA rows, so a single
    value per ``(participant_id, TRIAL_INDEX)`` is kept. ``participant_id`` is
    lower-cased to match the feature tables (which lower-case it during cleaning).
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
    """Return ``df`` with a numeric per-trial ``confidence_rating`` column.

    If the column is already present it is only coerced to numeric; otherwise it is
    merged in from ``confidence_source`` on the trial keys.
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
    """Flatten every run in a regime-split result into one tidy per-trial frame.

    Returns one row per (run, trial) with the trial keys, the true label, the model's
    predicted label, and its predicted P(correct). ``run`` is the run's label — each
    knowledge regime plus the pooled ``all`` baseline, so the pooled rows duplicate the
    per-regime ones by design (the pooled run is analysed as its own group).

    ``y_prob`` is positionally aligned with ``test_df`` (see
    ``evaluate_single_model_on_prepared_split``), so the columns are zipped by position.
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
    """Pooled baseline first, then the regimes in the order the runs were produced."""
    ordered = [r for r in runs if r == POOLED_LABEL]
    ordered += [r for r in runs if r != POOLED_LABEL]
    return ordered


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


def _correlate(x: np.ndarray, y: np.ndarray, method: str):
    """Return ``(statistic, p_value)``, or ``(nan, nan)`` when it is not computable."""
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
    """One correlation row (all methods) for ``x_col`` vs ``y_col`` in ``frame``.

    ``mean_x`` / ``mean_y`` are named generically so rows comparing different y columns
    (predicted probability vs. actual correctness) stack into one table.
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
    """Model-vs-confidence and confidence-vs-correctness correlations, per run.

    Two rows per run: the headline ``pred_prob ~ confidence`` correlation, and the
    ``is_correct ~ confidence`` reference that says whether the self-report is
    calibrated at all in that condition.
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
    """Within-participant ``pred_prob ~ confidence`` correlations for one run.

    Confidence scales are personal, so the pooled correlation confounds the
    within-subject effect with between-subject scale use. One row per participant
    isolates the former; participants who used a single rating for every trial come out
    as ``nan`` (no variance to correlate).
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
    """Per (run, confidence level): trial count, predicted P(correct), actual accuracy.

    The direct read of the headline question — if the model tracks confidence, mean
    ``pred_prob`` climbs down the rating scale; the ``accuracy`` column next to it shows
    whether the participants' own confidence climbed with their actual performance.
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
    save: bool = False,
    rel_dir: str = "answer_correctness/confidence",
    filename: str = "confidence_vs_predicted_probability_scatter",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Scatter of predicted P(correct) against the self-reported confidence, per run.

    Confidence is jittered horizontally so overlapping trials stay visible; marker fill
    encodes the true outcome, and the dashed line is the least-squares fit through the
    un-jittered points. Each panel is annotated with its Spearman rho.
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
    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved


def plot_prob_by_confidence_level(
    level_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    title_prefix: str = "",
    save: bool = False,
    rel_dir: str = "answer_correctness/confidence",
    filename: str = "predicted_probability_by_confidence_level",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Mean predicted P(correct) per confidence level, one line per run.

    The grey dashed line is the pooled *actual* accuracy at each confidence level — the
    curve the model would have to follow to be tracking the same thing the participants
    are reporting.
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

    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    return fig, saved


def plot_prob_boxplot_by_confidence(
    pred_df: pd.DataFrame,
    *,
    confidence_col: str = CONFIDENCE_COL,
    prob_col: str = PRED_PROB_COL,
    title_prefix: str = "",
    ncols: int = 2,
    save: bool = False,
    rel_dir: str = "answer_correctness/confidence",
    filename: str = "predicted_probability_boxplot_by_confidence",
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
):
    """Distribution (not just the mean) of predicted P(correct) at each confidence level.

    The means in :func:`plot_prob_by_confidence_level` can hide a level whose trials are
    split rather than shifted; the boxes show the spread behind each point.
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
    saved = maybe_save_plot(
        fig=fig, save=save, rel_dir=rel_dir, filename=filename,
        paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
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
    save: bool = False,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: str = "confidence",
    split_tag: str = "trainL1_testnew",
    verbose: bool = True,
    **regime_kwargs: Any,
) -> Dict[str, Any]:
    """Correlate the correctness model's predicted probabilities with self-reported confidence.

    Parameters
    ----------
    regime_result :
        The dict returned by :func:`..comparison_runs.run_regime_split_comparison`.
        Pass the already-computed result to reuse its fitted model and predictions;
        ``None`` (default) runs the regime split first, forwarding ``regime_kwargs``.
    confidence_source :
        Where to read the per-trial ``confidence_rating`` from: a raw IA report
        DataFrame, a per-trial mapping, or a CSV path (default: the new-exp cleaned IA
        report). Ignored if the predictions already carry the column.
    methods :
        Correlation flavours to report (default: Pearson, Spearman, Kendall).
    participant_run :
        Which run the per-participant correlations are computed on (default: the pooled
        ``all`` baseline, i.e. every new-data trial).
    save, paper_dirs, dpi, close :
        Plot/CSV output control, matching the rest of the answer-correctness pipeline.

    Returns
    -------
    dict with:
        ``correlations`` : per-run ``pred_prob ~ confidence`` and the
            ``is_correct ~ confidence`` calibration reference.
        ``by_confidence`` : per (run, confidence level) counts, predicted probability
            and actual accuracy.
        ``by_participant`` : within-participant correlations on ``participant_run``.
        ``predictions`` : the tidy per-(run, trial) frame everything is computed from.
        ``regime_result`` : the underlying regime-split result (freshly run or the one
            that was passed in).
        ``figures`` / ``paths`` : the created figures and any saved file paths.
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

    rel_dir = _answer_correctness_rel_dir(
        model_family="logreg", subdir=subdir, split_tag=split_tag
    )

    fig_scatter, scatter_paths = plot_confidence_vs_prob_scatter(
        pred_df, confidence_col=confidence_col, save=save,
        rel_dir=rel_dir, paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    fig_levels, level_paths = plot_prob_by_confidence_level(
        level_df, confidence_col=confidence_col, save=save,
        rel_dir=rel_dir, paper_dirs=paper_dirs, dpi=dpi, close=close,
    )
    fig_box, box_paths = plot_prob_boxplot_by_confidence(
        pred_df, confidence_col=confidence_col, save=save,
        rel_dir=rel_dir, paper_dirs=paper_dirs, dpi=dpi, close=close,
    )

    csv_paths: Dict[str, Any] = {}
    if save:
        for name, frame in (
            ("confidence_correlations", corr_df),
            ("confidence_by_level", level_df),
            ("confidence_by_participant", part_df),
            ("confidence_trial_predictions", pred_df),
        ):
            csv_paths[name] = save_df_csv(
                frame, rel_dir=rel_dir, filename=name, paper_dirs=paper_dirs
            )

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
