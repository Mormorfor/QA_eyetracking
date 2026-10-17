"""Regenerate every figure the paper uses, in the order the paper uses them.

    python scripts/make_paper_figures.py --list
    python scripts/make_paper_figures.py

**This is a skeleton.** The inventory is complete, the wiring is partial. Every entry below
is marked `WIRED` (rebuilt by `run_analysis.py`) or `TODO` (needs a person or a notebook), and
a run ends by listing the TODO ones, so a partial run cannot pass for a complete one.

Why some are TODO:

* the per-participant figures (`correctness_prediction/person_variance`) come from runs
  that take a long time and are run from notebooks;
* the paper's list of figures is not final: `draft2.tex` names its figures in the text only,
  with no `\\includegraphics` that is not commented out, so there is no list to check
  against yet.
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
    Figure("Results - Correctness associations", "correctness_associations",
           "analysis:correctness_associations", True),
    Figure("Results - Text Associations", "text_qa_relationship",
           "sequence of map/contrast calls; notebooks/drivers/text_associations.ipynb", False),
    Figure("Results - Error analysis", "error_analysis",
           "analysis:error_analysis", True),
    Figure("Results - Participant-level coefficients", "correctness_prediction/person_variance",
           "needs the per-participant runs; notebooks/drivers/per_person_runs.ipynb", False),
    Figure("Methods - Models comparison", "correctness_prediction",
           "analysis:correctness_prediction", True),
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
