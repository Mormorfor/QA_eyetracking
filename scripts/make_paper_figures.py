"""Regenerate every figure the paper uses, in the order the paper uses them.

    python scripts/make_paper_figures.py --list
    python scripts/make_paper_figures.py

**This is a skeleton** (Diana, 2026-10-07: *"You can make it a skeleton for now. We will work
on paper figures, but this comes when the codebase is already nice and organized."*). The
structure and the inventory are here; the wiring is partial on purpose.

**What it will not do is pretend.** Every entry below is marked `WIRED` or `TODO`, and a run
prints the TODO list at the end. A script that silently regenerates eight of fourteen figures
and exits 0 is worse than no script, because it looks like coverage.

**Why some are TODO.** They are not oversights, they are the open items:

* `correctness_prediction` figures come from cross-validation runs that take real time and are
  driven from notebooks; the CV results were also deleted with the old `reports/` tree and
  need regenerating first (`restructure-map.md` section 9).
* the error-analysis figures live in `notebooks/drivers/general_model_confusion.ipynb`, which
  persists nothing -- that is `todo.md` T4.0, pushed to Diana's manual pass.
* the paper's figure list is itself not final: `draft2.tex` references figures in prose only,
  with **no uncommented `\\includegraphics` anywhere**, so there is no machine-readable list to
  check this against yet.

Until those close, the honest output of this script is "here is what I can rebuild, and here
is what still needs a human".
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Figure:
    """One paper figure: where it belongs, and how (or whether) it can be rebuilt."""

    paper_section: str
    analysis: str
    how: str          # "analysis:<name>" -> run_analysis, or a note for TODO entries
    wired: bool


# Ordered as draft2 presents them (`research-context.md` section 3).
FIGURES: list[Figure] = [
    Figure("Results - First-scan behavior", "scan_strategies",
           "analysis:scan_strategies", True),
    Figure("Results - End of trial behavior", "last_visitation",
           "analysis:last_visitation", True),
    Figure("Results - Attention allocation", "attention_allocation",
           "analysis:attention_allocation", True),
    Figure("Results - Attention allocation (time course)", "time_course",
           "analysis:time_course", True),
    Figure("Results - Correctness associations", "correctness_associations",
           "analysis:correctness_associations", True),
    Figure("Results - Text Associations", "text_qa_relationship",
           "sequence of map/contrast calls; notebooks/drivers/text_associations.ipynb", False),
    Figure("Results - Error analysis", "correctness_prediction",
           "notebooks/drivers/general_model_confusion.ipynb persists nothing (T4.0)", False),
    Figure("Results - Participant-level coefficients", "correctness_prediction/person_variance",
           "needs the cached LOO pickle, deleted with the old reports/ tree", False),
    Figure("Methods - Models comparison", "correctness_prediction",
           "needs a cross-validation run; CV results not yet regenerated", False),
    Figure("Methods - Study 2 regimes and confidence", "correctness_prediction/knowledge_regimes",
           "notebooks/drivers/preliminary_analysis.ipynb", False),
    Figure("(unplaced) correct-vs-distractor RT asymmetry", "answer_rt_comparison",
           "one call; see analyses/answer_rt_comparison/plots.py", False),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    wired = [f for f in FIGURES if f.wired]
    todo = [f for f in FIGURES if not f.wired]

    if args.list:
        for f in FIGURES:
            print(f"  [{'WIRED' if f.wired else 'TODO '}] {f.paper_section:46s} {f.analysis}")
        return 0

    import run_analysis

    for f in wired:
        name = f.how.split(":", 1)[1]
        print(f"\n=== {f.paper_section} -> {name} ===")
        run_analysis.main(["--name", name] + (["--no-save"] if args.no_save else []))

    print(f"\nregenerated {len(wired)} of {len(FIGURES)} figure groups.")
    print(f"{len(todo)} still need a human:")
    for f in todo:
        print(f"  - {f.paper_section}: {f.how}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
