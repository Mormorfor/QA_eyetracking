"""The single write path for every figure and every number this project produces.

Everything under ``reports/`` is written by :func:`save_output`, and nothing else
writes there. That is the whole point of the module: whether a result survives a
run must not depend on which analysis produced it.

Three ideas hold it together.

**1. Callers describe the plot; they do not build a path.** A caller says which
analysis it belongs to, what the figure is, and which facets distinguish this
instance from its siblings. The directory and the filename are derived from that.
Hand-written paths are what let the old tree drift into four separator
conventions, two spellings of ``all_participants``, and names like
``dendrogram.png`` that mean nothing without their folder.

**2. The numbers travel with the figure.** ``tables`` is required. A figure whose
numbers are not saved is the failure mode that made ``docs/findings.md``
necessary — results recoverable only by reading a PNG. A genuinely data-less
figure (an illustration, a scanpath example) passes ``tables={}``, which is
greppable: "this has no numbers" becomes a stated choice rather than an omission.

**3. The layout mirrors the code.** ``reports/<analysis>/figures/`` and
``reports/<analysis>/tables/`` — so "which code made this figure?" is answerable
from the path alone, and a figure with no ``tables/`` sibling is visible at a
glance rather than requiring a comparison of two parallel trees.

See ``docs/restructure-map.md`` §9 for the layout decision and ``docs/todo.md``
T1.3 / T4.0 for the requirement this implements.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_ROOT = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Overleaf mirroring
#
# save_output can mirror into the Overleaf-synced papers/ repo. It is OFF, by
# instruction (2026-09-20): nothing goes to Overleaf during the restructure,
# because regenerating figures would otherwise change what the paper compiles
# while the numbers behind them are still moving.
#
# Asking to mirror while mirroring is disabled RAISES rather than quietly doing
# nothing — a caller that believes it published to the paper and did not is
# exactly the silent-wrong-outcome this module exists to prevent.
# ---------------------------------------------------------------------------

PAPER_MIRROR_ENABLED = False
PAPER_ROOT = PROJECT_ROOT / "papers" / "correctness_prediction"

# Everything mirrored lands under ONE folder inside the Overleaf repo, laid out
# exactly like the local reports/ tree:
#
#     papers/correctness_prediction/reports/<analysis>/figures/...
#                                          /<analysis>/tables/...
#
# The old arrangement put `figures/` and `report_data/` at the paper's top level,
# which flooded the repo root and split one analysis across two trees. One root
# also means the mirror path is the local path with a different prefix, so
# "which local file is this?" is answerable by swapping the root.
#
# The existing papers/correctness_prediction/{figures,report_data}/ folders are
# left exactly as they are -- papers/ is Diana's and Overleaf-synced.
PAPER_REPORTS_DIRNAME = "reports"


# ---------------------------------------------------------------------------
# The analysis registry
#
# An analysis name is a folder under reports/. It is validated rather than taken
# on trust: a typo would otherwise mint a new top-level folder and hide the
# figure somewhere nobody looks, which reads as "saved" but is not findable.
#
# Names track the target package layout in docs/restructure-map.md §3, so that
# reports/<name>/ and the analyses/<name>/ package that writes it stay in step.
# ---------------------------------------------------------------------------

ANALYSES: Dict[str, str] = {
    # --- paper code -------------------------------------------------------
    "attention_allocation": "Per-area attention metrics across the five screen areas",
    "scan_strategies": "Opening-scan strategies, dominance, first-visit matrices, dominant eye",
    "last_visitation": "Last area fixated before select / before confirm",
    "time_course": "Before / during / after the decision",
    "correctness_associations": "Correctness against scan effort, dwell, RT, preference matching",
    "correctness_prediction": "The headline answer-correctness model",
    "correctness_prediction/person_variance": "Per-participant LOO coefficients and accuracy",
    "correctness_prediction/knowledge_regimes": "Study 2 regimes, confidence, transfer",
    "answer_rt_comparison": "Correct vs distractor reading-time asymmetry",
    "text_qa_relationship": "Paragraph spans against answer/question dwell (RT_correlations)",
    # --- parked / future directions --------------------------------------
    "explorations/answer_reading_times": "Answer reading-time regression (answer_RTs)",
    "explorations/answer_location": "Answer-position prediction (answer_loc)",
    "explorations/participant_clustering": "Participant clustering and similarity",
    "explorations/feature_search": "Feature selection and column-set generation",
    "explorations/text_answer_effects": "Superseded mixed text->answer models",
    "explorations/unlikely_analysis": "Trials whose prediction disagrees with the outcome",
    "explorations/text_alignment": "KnowQA stored-text / interest-area alignment investigation",
}


def analysis_dir(analysis: str) -> Path:
    """Return ``reports/<analysis>``, validating the name against ANALYSES."""
    if analysis not in ANALYSES:
        known = "\n  ".join(sorted(ANALYSES))
        raise KeyError(
            f"Unknown analysis {analysis!r}. Add it to plot_output.ANALYSES if it is "
            f"genuinely new — do not invent a folder at the call site.\nKnown:\n  {known}"
        )
    return REPORTS_ROOT / analysis


# ---------------------------------------------------------------------------
# Defaults
#
# One object so a notebook can set its policy once ("save everything as pdf this
# session") instead of threading the same arguments through every call, while any
# single call can still override any field.
# ---------------------------------------------------------------------------


@dataclass
class OutputDefaults:
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
    return getattr(DEFAULTS, default_field) if value is None else value


# ---------------------------------------------------------------------------
# Naming
#
# <plot>__<facet>-<value>__<facet>-<value>.<ext>
#
# Key-value rather than positional so the name is readable on its own and
# `ls *group-hunters*` works. Facet order is the caller's keyword order, so a
# given plot function always produces the same shape of name.
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^0-9A-Za-z._+-]+")


# ---------------------------------------------------------------------------
# Abbreviations
#
# Names are read by a person, so they stay words rather than codes -- but the
# project's column vocabulary is long (`first_encounter_avg_pupil_size_z` is 32
# characters) and it lands in the filename *and* in the folder above it. Left
# alone, the longest path under reports/ reaches 259 of Windows' 260-character
# limit, so the repo only checks out on Windows from a root of <= 60 characters.
#
# This is the one place abbreviations are defined. Add a row rather than
# shortening a name at a call site, or the same concept ends up with two
# spellings -- which is exactly what `ANALYSES` exists to prevent for folders.
#
# Applied to every path component: facet values, plot names, table names and the
# `subdir`. That is what makes it sufficient -- a subdir is always either a plot
# name or a facet value, so one table covers all three.
#
# `manifest.json` keeps the FULL, unabbreviated facets, so nothing is lost: the
# manifest is the index, the filename is the human-readable label.
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
    "sequence_length": "seq_len",
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

# Two concepts abbreviating to one string would silently merge two real results.
_collisions = {
    short: [long for long in ABBREVIATIONS if ABBREVIATIONS[long] == short]
    for short in set(ABBREVIATIONS.values())
}
_clashes = {s: L for s, L in _collisions.items() if len(L) > 1}
if _clashes:
    raise ValueError(f"ABBREVIATIONS is not injective: {_clashes}")
# An abbreviation that is itself a key would apply twice and be unstable.
_chained = set(ABBREVIATIONS.values()) & set(ABBREVIATIONS)
if _chained:
    raise ValueError(f"ABBREVIATIONS maps onto its own keys (would chain): {_chained}")

# Lookup is case-insensitive, because the same concept is spelled several ways at
# the call sites -- the all-participants group arrives as "all participants",
# "All participants" and "all_participants" from three different plot families.
# (That inconsistency is worth fixing upstream; normalising here stops it
# splitting one group across two filenames in the meantime.) `slug` itself stays
# case-preserving, so `answer_A` and `answer_a` are still distinct -- only the
# table lookup folds case, which is safe exactly while no two keys differ only by
# case.
_ci = [k.lower() for k in ABBREVIATIONS]
_dupe_ci = {k for k in _ci if _ci.count(k) > 1}
if _dupe_ci:
    raise ValueError(f"ABBREVIATIONS keys collide case-insensitively: {_dupe_ci}")
_ABBREV_CI = {k.lower(): v for k, v in ABBREVIATIONS.items()}
del _collisions, _clashes, _chained, _ci, _dupe_ci


def slug(value: Any, *, abbreviate: bool = False) -> str:
    """Normalise one name component. Spaces and separators collapse to ``_``.

    Case is preserved: ``answer_A`` and ``answer_a`` are different columns in this
    project, so lowercasing names would merge two real things.

    ``abbreviate=True`` additionally maps the whole component through
    :data:`ABBREVIATIONS`. It is applied *after* normalisation, so the three
    spellings of the all-participants group (``all participants``,
    ``All participants``, ``all_participants``) all reach the table as one key.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = _SLUG_STRIP.sub("_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return _ABBREV_CI.get(text.lower(), text) if abbreviate else text


def build_stem(plot: str, facets: Mapping[str, Any]) -> str:
    """Build the shared filename stem for a figure and its tables.

    Facet *keys* are never abbreviated -- they are the short half already, and
    they are what makes the name readable (`group-hunters`, not `g-hunters`).
    """
    parts = [slug(plot, abbreviate=True)]
    for key, value in facets.items():
        if value is None:
            continue
        parts.append(f"{slug(key)}-{slug(value, abbreviate=True)}")
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Collecting a sweep into one table
#
# Some analyses are parameter sweeps: `attention_allocation` is 3 plot families
# over group x metric x selected x questions, which used to write 600 separate
# CSVs averaging 506 bytes -- each a 4x4 matrix. That satisfies "persist the
# numbers" only in the letter. One long table per family, with the facets as
# columns, is the same numbers in a form you can actually group and filter, and
# it is what lets `findings.md` be regenerated by a groupby rather than by
# globbing several hundred files.
#
# The figures are still written one per file -- only the tables are pooled.
# ---------------------------------------------------------------------------

_TABLE_SINK: Optional["TableCollector"] = None


class TableCollector:
    """Accumulates the tables of one sweep; see :func:`collect_tables`."""

    def __init__(self, analysis: str, plots: Optional[Sequence[str]] = None):
        self.analysis = analysis
        self.plots = set(plots) if plots else None
        self.rows: Dict[str, list] = {}

    def claims(self, analysis: str, plot: str) -> bool:
        return analysis == self.analysis and (self.plots is None or plot in self.plots)

    def add(self, plot: str, facets: Mapping[str, Any], tables: Mapping[str, Any]) -> None:
        for name, obj in tables.items():
            if not isinstance(obj, pd.DataFrame):
                continue          # json payloads keep their own file
            self.rows.setdefault(f"{plot}__{name}", []).append(
                obj.assign(**{k: v for k, v in facets.items() if v is not None})
            )

    def flush(self, save: bool = True) -> Dict[str, Any]:
        """Write one long table per (plot, table-name) pair."""
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
                manifest=False,   # not a figure; the long table is its own record
            )                     # no facets: this IS the family's table
        return written


@contextmanager
def collect_tables(analysis: str, plots: Optional[Sequence[str]] = None):
    """Pool every table written inside the block into one long table per family.

    ``plots`` limits the collection to named plot families; omit it to take
    everything written under ``analysis``. Nesting is not supported -- one sweep
    at a time is the only thing this is for.
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
    """Apply :data:`ABBREVIATIONS` to each component of a ``subdir``.

    A subdir is always either a plot name or a facet value, so it draws on the
    same vocabulary as the stem and must shorten with it -- otherwise the long
    name is removed from the filename and left in the folder above it.
    """
    if not subdir:
        return Path()
    return Path(*(slug(part, abbreviate=True) for part in Path(subdir).parts))


# ---------------------------------------------------------------------------
# The saver
# ---------------------------------------------------------------------------


@dataclass
class SavedOutput:
    """What a :func:`save_output` call actually wrote."""

    stem: str = ""
    analysis: str = ""
    figure: Optional[Path] = None
    tables: Dict[str, Path] = field(default_factory=dict)
    paper: Dict[str, Path] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.figure is not None or bool(self.tables)

    @property
    def paths(self) -> list:
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
    """Make numpy scalars and arrays JSON-writable rather than failing late."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _caller() -> str:
    """Identify the plotting function that asked for the save.

    Recorded in the manifest so a figure can be traced back to the code that made
    it without grepping for its filename.
    """
    for frame in inspect.stack()[1:]:
        module = frame.frame.f_globals.get("__name__", "")
        if module != __name__:
            return f"{module}.{frame.function}"
    return "unknown"


def _update_manifest(analysis: str, stem: str, entry: Dict[str, Any]) -> None:
    """Record what this figure is, keyed by stem, in reports/<analysis>/manifest.json."""
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
    """Save one figure and its numbers.

    Writes ``reports/<analysis>/figures/<stem>.<ext>`` and, for each entry in
    ``tables``, ``reports/<analysis>/tables/<stem>__<key>.{csv,json}`` — where
    ``stem`` is ``<plot>__<facet>-<value>...`` built from ``plot`` and ``facets``.

    Parameters
    ----------
    fig
        The figure. ``None`` saves tables only.
    analysis
        Which analysis owns this output; must be a key of :data:`ANALYSES`.
    plot
        What the figure is, e.g. ``"correctness_by_seq_len"``.
    tables
        Required. ``{name: DataFrame | Series | dict | list}``. Pass ``{}`` to
        declare that this figure genuinely has no numbers behind it.
    save
        Whether to write anything at all. Defaults to :data:`DEFAULTS`.
    to_paper
        Mirror into the Overleaf-synced paper repo, at
        ``papers/correctness_prediction/reports/<analysis>/{figures,tables}/`` —
        the same relative layout as ``reports/``, under one root so the paper
        repo does not gain a tree per output kind. Off by default and currently
        gated by :data:`PAPER_MIRROR_ENABLED`.
    path
        Escape hatch: an explicit output path for the figure, bypassing the
        derived location. Tables still follow ``analysis``.
    source
        Optional note on where the data came from (a dataset path, a run name),
        recorded in the manifest.
    subdir
        Optional folder under ``figures/`` and ``tables/``, purely so a large
        analysis stays browsable. The filename still carries every facet, so a
        file moved out of its subfolder loses nothing — the redundancy is
        deliberate.
    **facets
        Name/value pairs distinguishing this figure from its siblings, e.g.
        ``group="hunters", threshold=4``. They become the filename tags in the
        order given.

    Returns
    -------
    SavedOutput
        Falsy when ``save`` is off, so ``if result:`` tells you whether anything
        was written.
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
            "Set plot_output.PAPER_MIRROR_ENABLED = True to sync figures into the "
            "Overleaf repo — deliberately off during the restructure."
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
        # A sweep is collecting: the numbers go into one long table at the end
        # instead of one small file per figure. The figure is still written
        # normally, and `tables=` was still required and passed -- the contract
        # is unchanged, only the destination is.
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
                # The abbreviated form, because this has to locate the file:
                # `figures/<subdir>/<figure>` is how a reader resolves an entry,
                # and the directory on disk is abbreviated like everything else.
                # The unabbreviated value is not lost -- it is the facet it came
                # from, recorded in full in `facets` above.
                "subdir": str(leaf) if str(leaf) != "." else None,
                "tables": sorted(tables),
                "source": source,
                "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
        )

    if close and fig is not None:
        plt.close(fig)

    return result
