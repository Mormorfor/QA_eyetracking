# 2026-09-05 — Restructure map agreed; docs moved into the repo

Two decisions, recorded together because they happened in the same sitting.

---

## 1. `docs/restructure-map.md` is the plan of record

Diana: *"It might still be subject to change but I think we call it all tested and agreed upon
for now."*

**Agreed** means: a new Claude reading this repo should follow the map rather than re-litigate
it, and should propose a change to the map when the map looks wrong — not quietly do something
else. **Revisable** means the map is expected to meet friction during implementation; §13 names
the three places I expect it.

**Agreed is not started.** No stage begins without being proposed first. The hard rules in
`CLAUDE.md` still hold in full — no git operations, nothing under `papers/`, no hand-edited
data files, no data file moved without agreeing the how and why.

### The four choices the shape rests on

Asked as priorities questions on 2026-09-05; each one changes the design if reversed.

| Question | Answer | What it ruled out |
|---|---|---|
| Top-level axis | one shared data-prep area, then **a folder per functionality**, subfolders where related | a `study1_l1/` ⁄ `study2_knowqa/` split. Study is configuration, never a folder |
| Optimize for | clear and easy to follow, **open to adding new things by default**; extra folders are not a cost | flattening for convenience |
| Notebooks | **thin drivers plus scripted entry points** | notebooks as the source of any reported number |
| Scope | everything except `archive/`; data moves proposed individually | leaving `data/` naming as it is |

### Decisions settled inside the map

| | |
|---|---|
| Package name | `qa_eyetrack` |
| `matching_correctness` | → `correctness_associations/` |
| `total_answering_RT_normalized` | → `correctness_associations/`. Settled by reading the code: its figure is `plot_correctness_by_total_answering_rt_continuous`, so it is another `correctness_by_*` plot. **No `descriptives/` folder is needed** — every "basic" figure turned out to be a figure *about* something |
| per-area reading times | **a per-area quantity is a feature; a contrast between per-area quantities is an analysis.** RTs → `features/reading_times.py`; the correct-vs-distractor asymmetry → `analyses/answer_rt_comparison/` |
| `new_exp_try_runs` | confirmed to be `testrun_QA` |
| the two pilots | grouped under `data/datasets/pilots/`, fully runnable, with a `kind` field on the dataset record so code can tell — not only a human reading a listing |
| `archive/` | out of scope this round |

---

## 2. The docs moved out of `Claude outputs\` and into the repo

They were drafted in `Claude outputs\` while the placement was unsettled. That folder is now
**retired** — it is not a location any doc should return to, and anything still in it is a
leftover.

| now at | was |
|---|---|
| `CLAUDE.md` (repo root) | `Claude outputs\CLAUDE.md` |
| `docs/conventions.md` · `pitfalls.md` | same names |
| `docs/todo.md` · `findings.md` · `research-context.md` | " |
| `docs/data-pipeline.md` · `glossary.md` · `restructure-map.md` | " |
| `docs/decisions/` | new — this file is its first entry |

**Why the root for `CLAUDE.md` specifically:** it is read automatically at session start, and
its `@docs/conventions.md` / `@docs/pitfalls.md` imports resolve relative to it. Those two
imports only began working at the moment of this move.

### Two index entries that were promised and never written

`CLAUDE.md` listed `docs/outputs.md` and `docs/status.md`. Both were absorbed by the
restructure map before they were written, so the index now points at where the content
actually lives rather than at missing files:

- outputs conventions → `restructure-map.md` §9, plus `todo.md` T4.0
- live / parked / future-directions → `restructure-map.md` §3, where the split is structural
  (`analyses/` vs `explorations/`) rather than a list to maintain, plus `research-context.md` §7

### What did **not** change status

`docs/findings.md` keeps its ⚠️ **NOT VERIFIED BY DIANA** header. Diana's instruction stands:
it is useful for knowing what a figure folder holds, and must not be quoted. It gets rebuilt
from saved CSVs once T4.0 lands, at which point the warning comes off.
