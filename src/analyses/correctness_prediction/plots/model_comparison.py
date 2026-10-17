"""The model-comparison figure: several models' balanced accuracy, one panel per test.

One figure per dataset, drawn from `comparison.run_model_comparison`. Study 1 has a panel
for each kind of held-out trial and one for their average; Study 2 has one for each
knowledge regime and one for all of them together.

Every panel lists the same models in the same order, so a model can be followed across a
row, and all panels share one x-axis. The bars start at chance (0.5), so a bar's length
is how far the model is above chance. Study 1's error bars are the interval across the
ten folds; Study 2 is one fit and one test set, so it has none. A model that was not
tested on the dataset is left out of the panels and named under the figure.

Every figure saves its numbers beside it; see `conventions.md`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lib.plotting.output import save_output
from src.modeling.evaluate import AVERAGE_NOVELTY
from src.modeling.feature_sets import load_feature_sets
from src.analyses.correctness_prediction.comparison import (
    PAPER_MODELS,
    POOLED,
    TESTS,
    run_model_comparison,
)

ANALYSIS = "correctness_prediction"

# The balanced accuracy of a model that guesses. The bars start here.
CHANCE = 0.5

BAR_COLOR = "#4C72B0"
ERROR_COLOR = "#333333"
CHANCE_COLOR = "#888888"
GRID_COLOR = "#E5E5E5"

DATASET_TITLES = {
    "l1": "Study 1 (L1): ten-fold cross-validation",
    "knowqa": "Study 2 (KnowQA): trained on every Study 1 trial",
}

# Each panel's title: the name the tables use, then what it means.
TEST_TITLES = {
    "new_item": "new_item\nnew paragraph",
    "new_subject": "new_subject\nnew participant",
    "both": "both\nnew participant and paragraph",
    AVERAGE_NOVELTY: "average\nof the three",
    "no knowledge": "no knowledge",
    "partial knowledge": "partial knowledge",
    "full knowledge": "full knowledge",
    POOLED: "all regimes together",
}


def _panel_title(test: str, panel: pd.DataFrame, dataset: str) -> str:
    """The test's title; for Study 2, with how many trials it scored."""
    title = TEST_TITLES.get(test, test)
    if dataset != "knowqa":
        return title
    # Every tested model scores the same trials, so the panel has one count.
    counts = panel["n_scored"].dropna().unique()
    assert len(counts) == 1, f"{test}: the models scored different numbers of trials: {counts}"
    return f"{title}\n{int(counts[0])} trials"


def plot_model_comparison(
    comparison: Mapping[str, pd.DataFrame],
    *,
    dataset: str,
    models: Optional[Sequence[str]] = None,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    show_values: bool = True,
    save: Optional[bool] = None,
    to_paper=None,
    close: Optional[bool] = None,
    **facets,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Each model's balanced accuracy on every test of one dataset, as bars.

    `comparison` is what `run_model_comparison` returns, and `dataset` is "l1" or
    "knowqa". `models` picks the models to show, top to bottom; by default every model
    in the comparison, best first on the last panel (the average for Study 1, all
    regimes for Study 2). `labels` maps a model to its display name; by default it is
    shown by its name and number of features. `show_values` writes each score beside
    its bar.

    Saved with the scores it shows and, for Study 1, the cross-validation rows behind
    them. `**facets` go into the file name, e.g. `models="paper"`, so figures of
    different model sets do not overwrite each other.

    Returns the figure and the scores it shows.
    """
    if dataset not in TESTS:
        raise KeyError(f"dataset must be one of {sorted(TESTS)}, got {dataset!r}")
    tests = TESTS[dataset]
    scores = comparison["scores"]
    rows = scores[scores["dataset"] == dataset]

    available = list(dict.fromkeys(rows["model"]))
    if models is None:
        chosen = available
    else:
        unknown = [m for m in models if m not in available]
        if unknown:
            raise KeyError(f"not in this comparison: {unknown}; it has {available}")
        chosen = list(models)
    rows = rows[rows["model"].isin(chosen)]

    # A model with no score on any test was not tested on this dataset.
    has_score = rows.groupby("model")["balanced_accuracy"].apply(lambda s: s.notna().any())
    shown = [m for m in chosen if has_score[m]]
    untested = [m for m in chosen if not has_score[m]]
    if not shown:
        raise ValueError(f"none of {chosen} was tested on {dataset}")
    if models is None:
        last = rows[rows["test"] == tests[-1]].set_index("model")["balanced_accuracy"]
        shown = sorted(shown, key=lambda m: -last[m])

    n_features = rows.drop_duplicates("model").set_index("model")["n_features"]
    names = [
        labels[m] if labels and m in labels
        else f"{m}\n{int(n_features[m])} feature{'' if n_features[m] == 1 else 's'}"
        for m in shown
    ]

    # One x-axis for every panel: from below chance (or the lowest score) to past the
    # highest interval, with room for the value labels.
    ends = rows[rows["model"].isin(shown)][["balanced_accuracy", "ci_low", "ci_high"]].to_numpy(float)
    x_lo = min(CHANCE, np.nanmin(ends)) - 0.03
    x_hi = np.nanmax(ends) + (0.09 if show_values else 0.03)

    y = np.arange(len(shown))
    fig, axes = plt.subplots(
        1, len(tests), sharey=True, sharex=True, layout="constrained",
        figsize=(2.6 * len(tests) + 2.4, 0.5 * len(shown) + 1.8),
    )
    axes = np.atleast_1d(axes)

    for ax, test in zip(axes, tests):
        panel = rows[rows["test"] == test].set_index("model").reindex(shown)
        values = panel["balanced_accuracy"].to_numpy(float)
        lows = panel["ci_low"].to_numpy(float)
        highs = panel["ci_high"].to_numpy(float)
        has_value = np.isfinite(values)
        has_ci = has_value & np.isfinite(lows) & np.isfinite(highs)

        ax.barh(
            y[has_value], values[has_value] - CHANCE, left=CHANCE,
            height=0.62, color=BAR_COLOR, zorder=2,
        )
        if has_ci.any():
            ax.errorbar(
                values[has_ci], y[has_ci],
                xerr=np.vstack([values[has_ci] - lows[has_ci], highs[has_ci] - values[has_ci]]),
                fmt="none", ecolor=ERROR_COLOR, elinewidth=1.1, capsize=3, zorder=3,
            )
        # Dashed because it is a threshold, not part of the grid.
        ax.axvline(CHANCE, color=CHANCE_COLOR, linestyle="--", linewidth=1, zorder=1)

        if show_values:
            for yi, value, end, ok in zip(y, values, np.where(has_ci, highs, values), has_value):
                if ok:
                    ax.text(end + 0.008, yi, f"{value:.3f}", va="center", fontsize=8.5)
                else:
                    # A tested model with no score on this one test: e.g. a regime
                    # with only correct answers, where balanced accuracy is undefined.
                    ax.text(CHANCE + 0.008, yi, "no score", va="center", fontsize=8.5,
                            color=CHANCE_COLOR)

        ax.set_title(_panel_title(test, panel, dataset), fontsize=10)
        ax.set_xlim(x_lo, x_hi)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()  # first model at the top; shared by every panel

    fig.suptitle(title or DATASET_TITLES[dataset], fontsize=12)
    fig.supxlabel("balanced accuracy  (dashed line: chance)", fontsize=10)

    if untested:
        notes: Dict[str, List[str]] = {}
        for m in untested:
            notes.setdefault(rows.loc[rows["model"] == m, "note"].iloc[0], []).append(m)
        # Just below the figure's edge, under the axis label. The tight crop that
        # `save_output` and notebooks use takes it in.
        fig.text(
            0.01, 0.0,
            "\n".join(f"Not shown -- {note}: {', '.join(ms)}" for note, ms in notes.items()),
            fontsize=8.5, color="#555555", va="top", ha="left",
        )

    # The scores in panel order, then model order (untested models last).
    order = {m: i for i, m in enumerate(shown + untested)}
    plotted = (
        rows.assign(
            _test=rows["test"].map({t: i for i, t in enumerate(tests)}),
            _model=rows["model"].map(order),
        )
        .sort_values(["_test", "_model"])
        .drop(columns=["_test", "_model"])
        .reset_index(drop=True)
    )
    tables = {"scores": plotted}
    if dataset == "l1":
        folds = comparison["folds"]
        tables["folds"] = folds[folds["model"].isin(shown)]

    save_output(
        fig,
        analysis=ANALYSIS,
        plot="model_comparison",
        tables=tables,
        save=save,
        to_paper=to_paper,
        close=close,
        dataset=dataset,
        **facets,
    )
    return fig, plotted


def run_model_comparison_figures(
    word_df: pd.DataFrame,
    *,
    save: Optional[bool] = None,
    to_paper=None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """The paper's model comparison: the sets in `PAPER_MODELS`, from
    `configs/feature_sets/`, scored on Study 1 and Study 2, one figure each. Returns
    the comparison."""
    saved = load_feature_sets()
    models = {name: saved[name] for name in PAPER_MODELS}
    comparison = run_model_comparison(word_df, models, verbose=verbose)
    for dataset in TESTS:
        plot_model_comparison(comparison, dataset=dataset, save=save, to_paper=to_paper, close=True)
    return comparison
