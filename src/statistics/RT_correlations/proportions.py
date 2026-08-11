"""
Dwell proportions: how a trial's dwell time is split, on each of the two screens.

The answer screen already has them -- `area_dwell_proportion__answer_A`, ...,
`__question`, built by `data_prep.data_csv_generation.create_dwell_proportions`
as each area's share of the trial's total IA dwell time. The paragraph screen
has the same quantity per span (critical / distractor / outside), built the same
way by `answer_RTs.features._dwell_proportion` and cached in
`PARAGRAPH_SPAN_FEATURES_PATH`; it is simply not in the model-ready feature file
that the reading-time maps run on. `add_text_dwell_proportions` merges it in, and
the rest of this module maps the three text proportions against the five answer
ones with the same participant-level machinery as the RT / TFD maps.

Two things follow from proportions being proportions:

- Each side is compositional -- the 3 text shares sum to 1 and so do the 5
  answer shares. Within a side the parts are negatively coupled by
  construction, which is why the *rows* of a map cannot all move the same way:
  reading more of the critical span means reading proportionally less of the
  rest, whatever the answers do. The cross-screen cells themselves are still
  ordinary correlations, but a whole row (or column) is not free.
- Trials where nothing at all was dwelt on the answer screen leave all five
  answer shares at 0 (9 trials of 19436). They are dropped by default -- that is
  a missing screen, not a reading pattern.

Usage mirrors `report`::

    df = load_all_proportion_features()
    group_dfs = load_group_proportion_features()
    stats = proportion_map_stats(df)
    plot_proportion_maps({"All participants": stats})
    summarise_map(stats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src import constants as Con
from src.constants import TRIAL_ID_COLS
from src.data_paths import PARAGRAPH_SPAN_FEATURES_PATH
from src.statistics.RT_correlations.columns import ANSWERS, REGIONS
from src.statistics.RT_correlations.comparisons import (
    compare_groups,
    compare_rows_within_column,
)
from src.statistics.RT_correlations.correlations import DEFAULT_METHOD, corr_map_stats
from src.statistics.RT_correlations.data import GROUP_NAMES, load_features
from src.statistics.RT_correlations.plots import plot_corr_map_pair

__all__ = [
    "PROPORTION_PREFIX",
    "proportion_col",
    "region_proportion_cols",
    "answer_proportion_cols",
    "add_text_dwell_proportions",
    "load_proportion_features",
    "load_all_proportion_features",
    "load_group_proportion_features",
    "proportion_map_stats",
    "group_proportion_map_stats",
    "plot_proportion_maps",
    "proportion_region_contrasts",
    "proportion_group_contrast",
    "proportion_summary",
]

PROPORTION_PREFIX = Con.AREA_DWELL_PROPORTION  # "area_dwell_proportion"


def proportion_col(part: str) -> str:
    """`proportion_col("critical")` -> `area_dwell_proportion__critical`."""
    return f"{PROPORTION_PREFIX}__{part}"


def region_proportion_cols() -> list[str]:
    """Rows of a proportion map: the three paragraph spans."""
    return [proportion_col(region) for region in REGIONS]


def answer_proportion_cols() -> list[str]:
    """Columns of a proportion map: the four answers and the question."""
    return [proportion_col(answer) for answer in ANSWERS]


# ---------------------------------------------------------------------------
# Attaching the text-side proportions
# ---------------------------------------------------------------------------

def add_text_dwell_proportions(
    df: pd.DataFrame,
    paragraph_features: Optional[pd.DataFrame] = None,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add `area_dwell_proportion__{critical,distractor,outside}` to a feature frame.

    Left-joined on (participant_id, TRIAL_INDEX) from the cached paragraph-span
    features, which cover every trial of every participant, so nothing is
    dropped and no row is duplicated. `rebuild=True` (or a missing cache)
    re-runs the extraction from the paragraph IA report -- minutes, not seconds.
    """
    if paragraph_features is None:
        # Imported lazily: rebuilding is rare and pulls in the modelling stack.
        from src.predictive_modeling.answer_RTs.features import (
            load_paragraph_features,
            save_paragraph_features,
        )

        path = Path(paragraph_features_path)
        if rebuild or not path.exists():
            if verbose:
                print(f"building paragraph span features -> {path}")
            paragraph_features = save_paragraph_features(
                output_path=path, verbose=verbose
            )
        else:
            paragraph_features = load_paragraph_features(path)

    keys = list(TRIAL_ID_COLS)
    cols = region_proportion_cols()
    missing = [c for c in cols if c not in paragraph_features.columns]
    if missing:
        raise KeyError(f"paragraph features have no {missing}; rebuild them")

    out = df.merge(paragraph_features[keys + cols], on=keys, how="left")
    if verbose:
        matched = int(out[cols[0]].notna().sum())
        print(f"text dwell proportions: {matched}/{len(out)} trials matched")
    return out


def _drop_empty_answer_screens(
    df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """Drop trials with no dwell anywhere on the answer screen (all shares 0)."""
    empty = df[answer_proportion_cols()].sum(axis=1) == 0
    if verbose and empty.any():
        print(f"dropping {int(empty.sum())} trials with no answer-screen dwell")
    return df.loc[~empty].reset_index(drop=True)


def load_proportion_features(
    name: str = "all",
    drop_empty_answer_screens: bool = True,
    rebuild: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """`data.load_features` plus the three text dwell proportions.

    `rebuild` is passed to `load_features` (the model-ready cache), not to the
    paragraph-span cache; pass `rebuild=True` to `add_text_dwell_proportions`
    directly to regenerate that one.
    """
    df = load_features(name, rebuild=rebuild, verbose=verbose)
    df = add_text_dwell_proportions(df, verbose=verbose, **kwargs)
    if drop_empty_answer_screens:
        df = _drop_empty_answer_screens(df, verbose=verbose)
    return df


def load_all_proportion_features(**kwargs) -> pd.DataFrame:
    """Proportion features for all participants."""
    return load_proportion_features("all", **kwargs)


def load_group_proportion_features(**kwargs) -> dict[str, pd.DataFrame]:
    """Proportion features per reading regime: `{"hunters": ..., "gatherers": ...}`."""
    return {name: load_proportion_features(name, **kwargs) for name in GROUP_NAMES}


# ---------------------------------------------------------------------------
# The map, and the tests on it
# ---------------------------------------------------------------------------

def proportion_map_stats(
    df: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict:
    """Participant-level stats for the text-vs-answer proportion map.

    One map (proportions have no RT/TFD or pure/normalized variants), otherwise
    identical to `corr_map_stats`: within-participant r, Fisher z, one-sample
    t-test across participants, BH-corrected over the 15 cells.
    """
    return corr_map_stats(
        df,
        region_proportion_cols(),
        answer_proportion_cols(),
        method=method,
        **kwargs,
    )


def group_proportion_map_stats(
    group_dfs: dict[str, pd.DataFrame],
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict[str, dict]:
    """`proportion_map_stats` for each group, e.g. hunters and gatherers."""
    return {
        name: proportion_map_stats(frame, method=method, **kwargs)
        for name, frame in group_dfs.items()
    }


def plot_proportion_maps(
    stats_by_label: dict[str, dict],
    suptitle: str | None = None,
    **kwargs,
):
    """Proportion maps side by side -- typically all / hunters / gatherers.

    A shared colour scale across the panels, so the same colour means the same
    r in all of them.
    """
    first = next(iter(stats_by_label.values()))
    method = first.get("method", DEFAULT_METHOD)
    # Per-panel width matching `plot_maps` (14 for two panels), so the CI text
    # of one panel does not run into the next panel's row labels.
    n = len(stats_by_label)
    kwargs.setdefault("figsize", (7.0 * n, 4.5))
    return plot_corr_map_pair(
        stats_by_label,
        suptitle=(
            suptitle
            if suptitle is not None
            else (
                "Dwell proportions: text spans vs. answers/question "
                f"({method}, participant-level)"
            )
        ),
        **kwargs,
    )


def proportion_region_contrasts(stats: dict, **kwargs) -> pd.DataFrame:
    """Span-vs-span comparisons within each answer column of a proportion map."""
    return compare_rows_within_column(
        stats["z"], region_proportion_cols(), answer_proportion_cols(), **kwargs
    )


def proportion_group_contrast(
    stats_by_group: dict[str, dict],
    label_a: str = "hunters",
    label_b: str = "gatherers",
    region: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Between-group comparison of every cell of the proportion map.

    `region` restricts to one span, which also makes that span the
    multiple-comparison family: 5 tests instead of 15.
    """
    z_a = stats_by_group[label_a]["z"]
    z_b = stats_by_group[label_b]["z"]

    if region is not None:
        row = proportion_col(region)
        keep = [c for c in z_a.columns if c[0] == row]
        if not keep:
            raise KeyError(f"no cells for span {region!r} in the proportion map")
        z_a, z_b = z_a[keep], z_b[keep]

    return compare_groups(z_a, z_b, label_a, label_b, **kwargs)


def proportion_summary(df: pd.DataFrame, cols: Sequence[str] | None = None):
    """Mean / sd of each proportion column, as a sanity check on the shares."""
    cols = list(cols or region_proportion_cols() + answer_proportion_cols())
    return df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
