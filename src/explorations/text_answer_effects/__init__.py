"""Does paragraph reading predict how the answers are then read? -- parked.

Two files that only make sense together:

* `answers_paragraphs_csv.py` builds `{hunters,gatherers}_paragraph_answer_merge.csv`
  -- paragraph-span reading times joined onto answer-screen measures, one row per
  (trial, answer).
* `mixed_text_answer_effects.py` fits the mixed models on it.

**Parked, and probably headed for `archive/` (Diana, 2026-10-06.)** The merge
predates the split into separate answer and paragraph pipelines, and everything it
measures now exists in better form: paragraph spans come from
`features/paragraph/spans.py`, the answer side from `features/area_metrics.py`,
and they meet at trial level in the dataset's `features/model_ready.csv` rather than in a
bespoke CSV. Checked 2026-10-06: the merged CSVs are dated 2026-04-22, so they are
pre-rebuild (before T3.6, T3.18, T3.20, T3.21) -- **do not quote a number out of
them**; rebuild first if the question comes back.

`mixed_text_answer_effects` does not import under the current environment
(`pymer4` 0.9 dropped `Lmer`). That is a known parked failure, not a regression --
see `docs/todo_after_restructure.md`.
"""
