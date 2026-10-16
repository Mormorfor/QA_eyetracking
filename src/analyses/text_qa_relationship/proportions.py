"""
The correlation maps for dwell proportions: how a trial's looking time is split
between the areas of each screen.

On the answer screen, `area_dwell_proportion__answer_A`, ..., `__question` are
each area's share of the trial's looking time on that screen
(`features.area_metrics.create_dwell_proportions`). The paragraph screen has the
same measure for its three parts, `area_dwell_proportion__critical`,
`__distractor` and `__outside`, built by `features.paragraph.spans` and saved at
`PARAGRAPH_SPAN_FEATURES_PATH`. The model-ready tables already carry both sets,
so `add_text_dwell_proportions` normally finds the paragraph shares in place; it
joins them from the paragraph file only for a table that lacks them. The map
puts the three paragraph shares against the five answer-screen shares, with the
same participant-level tests as the RT / TFD maps.

Two things follow from these being proportions:

- Each side adds up to 1 -- the three paragraph shares, and the five answer
  shares -- so within a side the parts pull against each other by
  construction: reading more of the critical part means reading
  proportionally less of the rest, whatever the answers do. That is why the
  rows of a map cannot all move the same way. Each cell is still an ordinary
  correlation, but a whole row (or column) is not free.
- A trial with no looking time on the answer screen has all five answer shares
  at 0 (9 trials of 19,436). These are dropped by default: that is a missing
  screen, not a reading pattern.

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

from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import PARAGRAPH_SPAN_FEATURES_PATH
from src.analyses.text_qa_relationship.columns import ANSWERS, REGIONS
from src.analyses.text_qa_relationship.comparisons import (
    compare_groups,
    compare_rows_within_column,
)
from src.analyses.text_qa_relationship.correlations import DEFAULT_METHOD, corr_map_stats
from src.analyses.text_qa_relationship.data import GROUP_NAMES, load_features
from src.analyses.text_qa_relationship.plots import plot_corr_map_pair

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
    """Rows of a proportion map: the three paragraph parts."""
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
    """Make sure a trial table has the three paragraph shares,
    `area_dwell_proportion__{critical,distractor,outside}`.

    A table that already has all three (every model-ready table does) is
    returned unchanged. Otherwise they are joined on (participant_id,
    TRIAL_INDEX) from the paragraph file, and the run stops if any trial is left
    without a value, since that trial would silently become NaN and drop out of
    its correlations. A trial repeated in the file would duplicate rows; nothing
    checks for that.

    Unless `paragraph_features` is passed in, the paragraph file is loaded
    first, even for a table that already has the columns. With `rebuild=True`,
    or if the file is missing, it is rebuilt from the paragraph reports, which
    takes minutes.
    """
    if paragraph_features is None:
        # Imported only when needed: rebuilding is rare and pulls in the
        # feature-building code.
        from src.features.paragraph.spans import (
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

    # The model-ready tables already carry these three columns, joined in from
    # this same file by features.build.build_trial_level_model_df. Merging them
    # again would make pandas rename both copies to _x / _y, so a table that has
    # all three is used as it is.
    already = [c for c in cols if c in df.columns]
    if len(already) == len(cols):
        if verbose:
            print(
                "text dwell proportions: already present on the frame "
                f"({len(df)} trials); not re-merging"
            )
        return df
    if already:
        raise KeyError(
            f"frame carries only some of {cols} ({already}). Half-populated span "
            "proportions mean the upstream merge changed; fix that rather than "
            "topping the columns up here."
        )

    missing = [c for c in cols if c not in paragraph_features.columns]
    if missing:
        raise KeyError(f"paragraph features have no {missing}; rebuild them")

    out = df.merge(paragraph_features[keys + cols], on=keys, how="left")
    matched = int(out[cols[0]].notna().sum())
    if matched != len(out):
        raise ValueError(
            f"text dwell proportions: only {matched}/{len(out)} trials matched the "
            "paragraph cache. Every trial should be covered -- an unmatched trial "
            "would silently become NaN and drop out of the correlation pairwise."
        )
    if verbose:
        print(f"text dwell proportions: {matched}/{len(out)} trials matched")
    return out


def _drop_empty_answer_screens(
    df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """Drop the trials with no looking time on the answer screen (all five shares 0)."""
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
    """`data.load_features` plus the three paragraph shares, without the trials
    that have no looking time on the answer screen (unless
    `drop_empty_answer_screens=False`).

    `rebuild` applies to the model-ready table only; to rebuild the paragraph
    file, call `add_text_dwell_proportions(..., rebuild=True)` directly. Other
    keyword arguments go to `add_text_dwell_proportions`.
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
    """Proportion features for each group: `{"hunters": ..., "gatherers": ...}`."""
    return {name: load_proportion_features(name, **kwargs) for name in GROUP_NAMES}


# ---------------------------------------------------------------------------
# The map, and the tests on it
# ---------------------------------------------------------------------------

def proportion_map_stats(
    df: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> dict:
    """Participant-level tests for the proportion map: `corr_map_stats` on the
    three paragraph shares against the five answer-screen shares.

    There is one map only: proportions have no RT / TFD or pure / normalized
    versions. Keyword arguments go to `corr_map_stats`; by default the p-values
    are corrected over the 15 cells (Benjamini-Hochberg).
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
    """Proportion maps side by side -- typically all / hunters / gatherers -- on
    one colour scale, so the same colour means the same r in every panel.

    Saved through `plot_corr_map_pair` under the same plot name as the
    reading-time maps, told apart from them by `measure="dwell_proportions"`
    in the file name.
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
        # `measure` says which quantity the map shows: same rows and columns as
        # the reading-time maps, but a share of the screen's looking time rather
        # than time per word. There is no `scaling` here -- a proportion has no
        # units to scale.
        plot=kwargs.pop("plot", "text_qa_corr_map"),
        measure="dwell_proportions",
        panels="+".join(stats_by_label),
        **kwargs,
    )


def proportion_region_contrasts(stats: dict, **kwargs) -> pd.DataFrame:
    """Part-vs-part comparisons within each column of a proportion map."""
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
    """Compare the two groups on every cell of the proportion map (`compare_groups`).

    `region` limits the comparison to one paragraph part, which also makes that
    row the multiple-comparison family: 5 tests instead of 15.
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
    """Count, mean, SD, min, median and max of each share, as a quick check on them."""
    cols = list(cols or region_proportion_cols() + answer_proportion_cols())
    return df[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
