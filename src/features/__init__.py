"""IA-level tables -> trial-level features.

Knows what a measurement is: dwell proportion, skip rate, reading time,
scanning strategy, pupil z-score. Does NOT know what an EyeLink export looks
like -- that is `ingest/`. The boundary between them is the tidy word-level
table, and **`features/` must not import from `ingest/`.**

That rule holds with no exception as of 2026-10-06. The last one -- `build.py`
reaching for `ingest.geometry` -- closed when the row-level builders and
`FUNCTION_REGISTRY` moved to `ingest/` (`docs/restructure-map.md` §5.4).

Where things came from, 2026-10-06 (stage C). Comments dated before then name
the old paths, so this is the key that resolves them:

    derived/area_metrics.py        -> features/area_metrics.py
    derived/reading_times.py       -> features/reading_times.py
    derived/pupil_norm.py          -> features/pupil.py
    derived/pattern_breaking.py    -> features/strategies.py
    derived/select_confirm_last.py -> features/last_visited.py
    derived/preference_matching.py -> features/preference.py
    derived/paragraph_prep.py      -> features/paragraph/spans.py
    data_prep/data_csv_generation.py (part) -> area_metrics · pupil ·
                                               sequences · last_visited · scope
"""
