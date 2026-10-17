"""Does paragraph reading predict how the answers are then read? -- parked.

Two files that only make sense together:

* `answers_paragraphs_csv.py` builds `{hunters,gatherers}_paragraph_answer_merge.csv`
  -- paragraph-span reading times joined onto answer-screen measures, one row per
  (trial, answer).
* `mixed_text_answer_effects.py` fits the mixed models on it.

**Parked, and broken.** What the merge measures is now built properly elsewhere: the
paragraph parts in `features/paragraph/spans.py`, the answers in
`features/area_metrics.py`, joined by trial in `features/model_ready.csv`. The merged
CSVs on disk were made from older data, so do not quote a number from them; and the
merge's default inputs, `hunters.csv` / `gatherers.csv`, are not written by the build.

`mixed_text_answer_effects` does not import: the installed `pymer4` (0.9) has no `Lmer`.
The notebook of the same name, beside this file, drives it.
"""
