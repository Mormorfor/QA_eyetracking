"""The one function that saves figures and their numbers.

Everything under ``reports/`` is written by :func:`save_output`, and nothing
else writes there. Three rules:

1. **Describe the plot; don't build a path.** A caller says which analysis the
   figure belongs to, what it is, and which settings ("facets") tell it apart
   from its siblings. The folder and the filename are worked out from that, so
   every file is named the same way.

2. **The numbers are saved with the figure.** ``tables`` is required. A figure
   that genuinely has no numbers behind it (an illustration, a scanpath example)
   passes ``tables={}``, so "this has no numbers" is always a stated choice.

3. **The folders follow the code.** Output goes to
   ``reports/<analysis>/figures/`` and ``reports/<analysis>/tables/``, so the
   path alone says which analysis made a file, and a figure without its tables
   is easy to spot.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd

# src/lib/plotting/output.py -> parents[3] is the repo root. The number is a
# count of folders from this file, so it must change if the file moves. A wrong
# count would not raise by itself -- every output would quietly land in the wrong
# folder -- so the assert below checks it.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
assert (PROJECT_ROOT / "src" / "lib" / "plotting").is_dir(), (
    f"PROJECT_ROOT resolved to {PROJECT_ROOT}, which does not look like the repo "
    "root -- the parents[] depth above is stale after a move."
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_ROOT = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Copying figures into the paper
#
# save_output can also copy each figure and table into the Overleaf-synced
# papers/ repo. This is switched OFF, so that regenerating a figure does not
# change what the paper compiles until you decide to sync.
#
# Asking for a copy while it is switched off RAISES an error rather than quietly
# doing nothing: code that thinks it updated the paper, and didn't, is the
# outcome to avoid.
# ---------------------------------------------------------------------------

PAPER_MIRROR_ENABLED = False
PAPER_ROOT = PROJECT_ROOT / "papers" / "correctness_prediction"

# Copies land in one folder inside the Overleaf repo, laid out exactly like the
# local reports/ folder:
#
#     papers/correctness_prediction/reports/<analysis>/figures/...
#                                          /<analysis>/tables/...
#
# So a copy's path is its local path with a different start, and finding the
# local file behind a paper figure means swapping that start.
PAPER_REPORTS_DIRNAME = "reports"


# ---------------------------------------------------------------------------
# The list of analyses
#
# Each analysis name is a folder under reports/, matching the analyses/<name>/
# code that writes it. Names are checked against this list: a typo would
# otherwise create a new folder and hide the figure where nobody looks.
# ---------------------------------------------------------------------------

ANALYSES: Dict[str, str] = {
    # --- paper code -------------------------------------------------------
    "attention_allocation": "Per-area attention metrics across the five screen areas",
    "scan_strategies": "Opening-scan strategies, dominance, first-visit matrices, dominant eye",
    "last_visitation": "Last area fixated before select / before confirm",
    "correctness_associations": "Correctness against scan effort, dwell, RT, preference matching",
    "correctness_prediction": "The headline answer-correctness model",
    "correctness_prediction/person_variance": "Models fitted per participant: coefficients and accuracy",
    "correctness_prediction/knowledge_regimes": "Study 2 regimes, confidence, transfer",
    "answer_rt_comparison": "Correct vs distractor reading-time asymmetry",
    "text_qa_relationship": "Time on each paragraph part against time on each answer and the question",
    # --- explorations: kept, but not used in the paper --------------------
    "explorations/answer_reading_times": "Predicting reading time on each answer",
    "explorations/answer_location": "Predicting which screen position was chosen",
    "explorations/participant_clustering": "Participant clustering and similarity",
    "explorations/feature_search": "Feature selection and column-set generation",
    "explorations/text_answer_effects": "Mixed-effects models of paragraph parts against answer dwell",
    "explorations/unlikely_analysis": "Trials whose prediction disagrees with the outcome",
    "explorations/text_alignment": "KnowQA stored-text / interest-area alignment investigation",
    "explorations/time_course": "Before / during / after the first look at the chosen answer (unfinished)",
}


def analysis_dir(analysis: str) -> Path:
    """Return ``reports/<analysis>``, validating the name against ANALYSES."""
    if analysis not in ANALYSES:
        known = "\n  ".join(sorted(ANALYSES))
        raise KeyError(
            f"Unknown analysis {analysis!r}. If it is genuinely new, add it to "
            f"ANALYSES in src/lib/plotting/output.py.\nKnown:\n  {known}"
        )
    return REPORTS_ROOT / analysis


# ---------------------------------------------------------------------------
# Defaults
#
# A notebook can set these once ("save everything as pdf") instead of passing
# the same arguments to every call. Any single call can still override them.
# ---------------------------------------------------------------------------


@dataclass
class OutputDefaults:
    """Session-wide defaults for `save_output`. Nothing is saved unless asked."""

    save: bool = False
    to_paper: bool = False
    ext: str = "png"
    dpi: int = 300
    bbox_inches: Optional[str] = "tight"
    close: bool = False
    manifest: bool = True


DEFAULTS = OutputDefaults()


def set_output_defaults(**kwargs: Any) -> OutputDefaults:
    """Set session-wide save defaults, e.g. ``set_output_defaults(save=True, ext="pdf")``."""
    global DEFAULTS
    unknown = set(kwargs) - set(DEFAULTS.__dataclass_fields__)
    if unknown:
        raise TypeError(f"Unknown output defaults: {sorted(unknown)}")
    DEFAULTS = replace(DEFAULTS, **kwargs)
    return DEFAULTS


def _pick(value: Any, default_field: str) -> Any:
    """The call's own value if it gave one, otherwise the session default."""
    return getattr(DEFAULTS, default_field) if value is None else value


# ---------------------------------------------------------------------------
# File names
#
# <plot>__<facet>-<value>__<facet>-<value>.<ext>
#
# Each setting is written as name-value, so a file name reads on its own and
# `ls *group-hunters*` finds every hunters figure. Settings appear in the order
# the caller passed them, so one plot function always produces the same shape
# of name.
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^0-9A-Za-z._+-]+")


# ---------------------------------------------------------------------------
# Abbreviations
#
# Column names are long (`first_encounter_avg_pupil_size_z` is 32 characters),
# and they appear in both the file name and the folder above it. Without
# shortening, some paths under reports/ would pass Windows' 260-character limit
# when the repo is cloned into a deep folder.
#
# This table is the only place abbreviations are defined. To shorten a name, add
# a row here rather than passing a short string at the call site -- otherwise
# one concept ends up with two spellings.
#
# It applies to every part of a path: setting values, plot names, table names
# and `subdir`. `manifest.json` keeps the full names, so nothing is lost.
# ---------------------------------------------------------------------------

ABBREVIATIONS: Dict[str, str] = {
    # participant groups
    "all_participants": "all_P",
    "all_participants+hunters+gatherers": "all_P+hunters+gatherers",
    # per-area metrics -- "fix" and "size" carry no information here, every one
    # of these is a fixation measure and a size
    "first_encounter_avg_pupil_size_z": "first_enc_pupil_z",
    "first_encounter_avg_pupil_size": "first_enc_pupil",
    "mean_first_fixation_duration": "mean_first_fix_dur",
    "mean_avg_fix_pupil_size_z": "mean_avg_pupil_z",
    "mean_max_fix_pupil_size_z": "mean_max_pupil_z",
    "mean_min_fix_pupil_size_z": "mean_min_pupil_z",
    "mean_avg_fix_pupil_size": "mean_avg_pupil",
    "mean_max_fix_pupil_size": "mean_max_pupil",
    "mean_min_fix_pupil_size": "mean_min_pupil",
    "area_dwell_proportion": "dwell_prop",
    "mean_fixations_count": "mean_fix_count",
    "num_label_visits": "n_lbl_visits",
    "mean_dwell_time": "mean_dwell",
    "segment_ia_row_count": "ia_rows",
    "fixation_count": "fix_count",
    # plot names -- the correctness_by_* family is 29-42 characters
    "correctness_by_trial_mean_dwell_continuous": "corr_by_dwell_cont",
    "correctness_by_trial_mean_dwell_threshold": "corr_by_dwell_thresh",
    "correctness_by_seq_len_continuous": "corr_by_seq_len_cont",
    "correctness_by_seq_len_threshold": "corr_by_seq_len_thresh",
    "correctness_by_total_answering_rt": "corr_by_total_RT",
    "correctness_by_back_and_forth": "corr_by_back_forth",
    "correctness_by_matching": "corr_by_matching",
    "area_pairwise_significance": "area_pairwise_sig",
    "area_label_by_loc_heatmap": "area_lbl_by_loc",
    "dominant_strategies_above_threshold": "dom_strat_above_thresh",
    "dominant_strategy_by_eye_crosstab": "dom_strat_by_eye_tab",
    "dominant_strategies_by_eye": "dom_strat_by_eye",
    "dominant_strategy_proportion": "dom_strat_prop",
    "dominance_raw_vs_completed": "dom_raw_vs_compl",
    "strategy_count_distribution": "strat_count_dist",
    "strategy_completed": "strat_completed",
    # misc long facet values
    "CONFIRM_FINAL_ANSWER_RT": "confirm_RT",
    "dwell_proportions": "dwell_props",
    "normalized": "norm",
}

# Safety checks on the table, run on import. If two names shortened to the same
# string, two different results would be saved into one file.
_collisions = {
    short: [long for long in ABBREVIATIONS if ABBREVIATIONS[long] == short]
    for short in set(ABBREVIATIONS.values())
}
_clashes = {s: L for s, L in _collisions.items() if len(L) > 1}
if _clashes:
    raise ValueError(f"ABBREVIATIONS is not injective: {_clashes}")
# An abbreviation that is itself a key would be shortened again.
_chained = set(ABBREVIATIONS.values()) & set(ABBREVIATIONS)
if _chained:
    raise ValueError(f"ABBREVIATIONS maps onto its own keys (would chain): {_chained}")

# The lookup ignores upper/lower case, because different plot code spells the
# same group differently ("all participants", "All participants",
# "all_participants") and all three should get one file name. Names themselves
# keep their case -- `answer_A` and `answer_a` are different columns -- so this
# is only safe while no two keys differ by case alone, which is checked here.
_ci = [k.lower() for k in ABBREVIATIONS]
_dupe_ci = {k for k in _ci if _ci.count(k) > 1}
if _dupe_ci:
    raise ValueError(f"ABBREVIATIONS keys collide case-insensitively: {_dupe_ci}")
_ABBREV_CI = {k.lower(): v for k, v in ABBREVIATIONS.items()}
del _collisions, _clashes, _chained, _ci, _dupe_ci


def slug(value: Any, *, abbreviate: bool = False) -> str:
    """Turn one part of a name into something safe for a file name.

    Spaces and odd characters become ``_``. Upper/lower case is kept, because
    ``answer_A`` and ``answer_a`` are different columns. ``True``/``False``
    become ``yes``/``no``.

    With ``abbreviate=True`` the result is also shortened through
    :data:`ABBREVIATIONS`.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = _SLUG_STRIP.sub("_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return _ABBREV_CI.get(text.lower(), text) if abbreviate else text


def build_stem(plot: str, facets: Mapping[str, Any]) -> str:
    """The file name, without extension, shared by a figure and its tables.

    Setting values are shortened; setting names are not, because they are
    already short and they are what makes the name readable (`group-hunters`,
    not `g-hunters`). Settings whose value is None are left out.
    """
    parts = [slug(plot, abbreviate=True)]
    for key, value in facets.items():
        if value is None:
            continue
        parts.append(f"{slug(key)}-{slug(value, abbreviate=True)}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Pooling many small tables into one
#
# Some analyses draw hundreds of near-identical figures, one for each
# combination of settings -- `attention_allocation` loops over group x metric x
# chosen answer x with/without the question. Saving each figure's numbers
# separately would give hundreds of tiny CSV files. Inside `collect_tables`,
# those numbers are instead gathered into one long table per plot type, with
# the settings as extra columns, so they can be filtered and compared in one
# place.
#
# The figures are still saved one per file. Only the tables are pooled.
# ---------------------------------------------------------------------------

_TABLE_SINK: Optional["TableCollector"] = None


class TableCollector:
    """Gathers the tables of one loop of figures; see :func:`collect_tables`."""

    def __init__(self, analysis: str, plots: Optional[Sequence[str]] = None):
        self.analysis = analysis
        self.plots = set(plots) if plots else None
        self.rows: Dict[str, list] = {}

    def claims(self, analysis: str, plot: str) -> bool:
        """Whether a save for this analysis and plot belongs in this collection."""
        return analysis == self.analysis and (self.plots is None or plot in self.plots)

    def add(self, plot: str, facets: Mapping[str, Any], tables: Mapping[str, Any]) -> None:
        """Add one figure's tables, tagging every row with that figure's settings.

        Only DataFrames are pooled. A table given as a dict, list or Series is
        skipped here, and while a collection is active `save_output` does not
        write it anywhere else either -- so it is not saved at all.
        """
        for name, obj in tables.items():
            if not isinstance(obj, pd.DataFrame):
                continue          # not saved: see the docstring
            self.rows.setdefault(f"{plot}__{name}", []).append(
                obj.assign(**{k: v for k, v in facets.items() if v is not None})
            )

    def flush(self, save: bool = True) -> Dict[str, Any]:
        """Write one long table per plot type and table name."""
        written = {}
        for key, frames in self.rows.items():
            plot, _, name = key.partition("__")
            long_df = pd.concat(frames, ignore_index=True)
            written[key] = save_output(
                None,
                analysis=self.analysis,
                plot=plot,
                tables={name: long_df},
                save=save,
                manifest=False,   # not a figure, so no manifest entry
            )                     # no settings: this is the whole plot type's table
        return written


@contextmanager
def collect_tables(analysis: str, plots: Optional[Sequence[str]] = None):
    """Inside this block, pool every saved table into one long table per plot type.

    Used as ``with collect_tables("attention_allocation", plots=[...]):`` around
    a loop that saves many figures. ``plots`` limits the pooling to the named plot
    types; leave it out to pool everything saved under ``analysis``. Blocks
    cannot be nested.
    """
    global _TABLE_SINK
    if _TABLE_SINK is not None:
        raise RuntimeError("collect_tables is already active; it does not nest")
    _TABLE_SINK = TableCollector(analysis, plots)
    try:
        yield _TABLE_SINK
    finally:
        sink, _TABLE_SINK = _TABLE_SINK, None
    sink.flush()


def abbreviate_subdir(subdir: Union[str, Path, None]) -> Path:
    """Shorten each folder name in a ``subdir`` through :data:`ABBREVIATIONS`.

    A subdir is named after a plot or a setting, so it is shortened the same way
    as the file name -- otherwise the long name would just move up a folder.
    """
    if not subdir:
        return Path()
    return Path(*(slug(part, abbreviate=True) for part in Path(subdir).parts))


# ---------------------------------------------------------------------------
# The saver
# ---------------------------------------------------------------------------


@dataclass
class SavedOutput:
    """What a :func:`save_output` call wrote.

    Counts as False when nothing was written, so ``if result:`` works.
    """

    stem: str = ""
    analysis: str = ""
    figure: Optional[Path] = None
    tables: Dict[str, Path] = field(default_factory=dict)
    paper: Dict[str, Path] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.figure is not None or bool(self.tables)

    @property
    def paths(self) -> list:
        """Every file written: figure, tables, and any copies in the paper."""
        out = [self.figure] if self.figure else []
        return out + list(self.tables.values()) + list(self.paper.values())


def _write_table(obj: Any, out_dir: Path, name: str) -> Path:
    """Write one table. DataFrame/Series -> csv, dict/list -> json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, pd.Series):
        obj = obj.to_frame()

    if isinstance(obj, pd.DataFrame):
        path = out_dir / f"{name}.csv"
        obj.to_csv(path, index=False)
        return path

    if isinstance(obj, (dict, list, tuple)):
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=_json_default)
        return path

    raise TypeError(
        f"Cannot save table {name!r} of type {type(obj).__name__}. "
        "Pass a DataFrame, Series, dict or list."
    )


def _json_default(obj: Any) -> Any:
    """Let numpy numbers and arrays be written to JSON."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _caller() -> str:
    """The name of the function that asked for the save.

    Recorded in the manifest, so a figure can be traced to the code that made it.
    """
    for frame in inspect.stack()[1:]:
        module = frame.frame.f_globals.get("__name__", "")
        if module != __name__:
            return f"{module}.{frame.function}"
    return "unknown"


def _update_manifest(analysis: str, stem: str, entry: Dict[str, Any]) -> None:
    """Record what this figure is in reports/<analysis>/manifest.json.

    The manifest is an index of every figure in the analysis: its settings in
    full, the function that made it, its tables, and when it was saved.
    """
    path = analysis_dir(analysis) / "manifest.json"
    manifest: Dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    manifest[stem] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(manifest.items())), handle, indent=2, default=_json_default)


def save_output(
    fig: Optional[plt.Figure] = None,
    *,
    analysis: str,
    plot: str,
    tables: Mapping[str, Any],
    save: Optional[bool] = None,
    to_paper: Optional[bool] = None,
    ext: Optional[str] = None,
    dpi: Optional[int] = None,
    bbox_inches: Optional[str] = None,
    close: Optional[bool] = None,
    manifest: Optional[bool] = None,
    path: Optional[Union[str, Path]] = None,
    source: Optional[str] = None,
    subdir: Optional[str] = None,
    **facets: Any,
) -> SavedOutput:
    """Save one figure and the numbers behind it.

    Writes ``reports/<analysis>/figures/<name>.<ext>`` and, for each entry in
    ``tables``, ``reports/<analysis>/tables/<name>__<table>.{csv,json}``. The
    ``<name>`` is built from ``plot`` and the settings: ``<plot>__<key>-<value>...``.

    Parameters
    ----------
    fig
        The figure. ``None`` saves tables only.
    analysis
        Which analysis this belongs to; must be a key of :data:`ANALYSES`.
    plot
        What the figure is, e.g. ``"correctness_by_seq_len"``.
    tables
        Required. ``{name: DataFrame | Series | dict | list}``. Pass ``{}`` to
        say that this figure genuinely has no numbers behind it.
    save
        Whether to write anything at all. Defaults to :data:`DEFAULTS`.
    to_paper
        Also copy into the Overleaf paper repo, at
        ``papers/correctness_prediction/reports/<analysis>/{figures,tables}/``.
        Only allowed while :data:`PAPER_MIRROR_ENABLED` is True.
    path
        Save the figure here instead of the usual place. Tables still go to the
        usual place.
    source
        Optional note on where the data came from (a dataset, a run name), kept
        in the manifest.
    subdir
        Optional subfolder under ``figures/`` and ``tables/``, to keep a large
        analysis browsable. The file name still carries every setting, so a file
        moved out of its subfolder loses nothing.
    **facets
        The settings that tell this figure apart from its siblings, e.g.
        ``group="hunters", threshold=4``. They become the file name, in the
        order given.

    Returns
    -------
    SavedOutput
        What was written. Counts as False when ``save`` is off, so
        ``if result:`` says whether anything was saved.
    """
    if tables is None:
        raise TypeError(
            f"{analysis}/{plot}: tables= is required. Pass the frames behind the "
            "figure, or tables={} to declare that it has none."
        )

    save = _pick(save, "save")
    stem = build_stem(plot, facets)

    if not save:
        return SavedOutput(stem=stem, analysis=analysis)

    to_paper = _pick(to_paper, "to_paper")
    ext = _pick(ext, "ext")
    dpi = _pick(dpi, "dpi")
    bbox_inches = _pick(bbox_inches, "bbox_inches")
    close = _pick(close, "close")
    manifest = _pick(manifest, "manifest")

    if to_paper and not PAPER_MIRROR_ENABLED:
        raise RuntimeError(
            f"{analysis}/{stem}: to_paper=True but paper mirroring is disabled. "
            "Set PAPER_MIRROR_ENABLED = True in src/lib/plotting/output.py to "
            "copy figures into the Overleaf repo."
        )

    base = analysis_dir(analysis)
    leaf = abbreviate_subdir(subdir)
    result = SavedOutput(stem=stem, analysis=analysis)

    if fig is not None:
        if path is not None:
            fig_path = Path(path)
        else:
            fig_path = base / "figures" / leaf / f"{stem}.{ext}"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches)
        result.figure = fig_path

    if _TABLE_SINK is not None and _TABLE_SINK.claims(analysis, plot):
        # Inside collect_tables: the numbers go into the pooled table instead of
        # their own file. The figure is still saved as usual.
        _TABLE_SINK.add(plot, facets, tables)
    else:
        for name, obj in tables.items():
            result.tables[name] = _write_table(
                obj, base / "tables" / leaf, f"{stem}__{slug(name, abbreviate=True)}"
            )

    if to_paper:
        # Same relative layout as reports/, under one root in the paper repo.
        paper_base = PAPER_ROOT / PAPER_REPORTS_DIRNAME / analysis
        if fig is not None:
            paper_fig = paper_base / "figures" / leaf / f"{stem}.{ext}"
            paper_fig.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(paper_fig, dpi=dpi, bbox_inches=bbox_inches)
            result.paper["figure"] = paper_fig
        for name, obj in tables.items():
            result.paper[name] = _write_table(
                obj, paper_base / "tables" / leaf, f"{stem}__{slug(name, abbreviate=True)}"
            )

    if manifest:
        _update_manifest(
            analysis,
            stem,
            {
                "plot": plot,
                "facets": {k: v for k, v in facets.items() if v is not None},
                "produced_by": _caller(),
                "figure": result.figure.name if result.figure else None,
                # The shortened folder name, because this is used to find the
                # file on disk. The full name is still in `facets` above.
                "subdir": str(leaf) if str(leaf) != "." else None,
                "tables": sorted(tables),
                "source": source,
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
        )

    if close and fig is not None:
        plt.close(fig)

    return result
