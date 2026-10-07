# notebooks/exploration

Free-form. **Never the source of a reported number.**

Parked strands, one-off investigations and presentation material. Three of these have
zero executed cells (`old_clustering_attempt`, `unlikely_analysis`, and `RT_pred`'s GBM
section) -- they are not 'run and cleared', they have not run in their current state.

`presentation_prep` writes to a folder **outside the repository**, and its `collect_triples`
docstring contradicts its filter (it says `is_correct == 1`, the code filters `== 0`).
Diana ruled 2026-10-06 that the contradiction is harmless because the talk has been given;
if anything is ever lifted out of it, read the filter rather than the comment.
