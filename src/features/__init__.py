"""Tidy word-level tables -> one row of features per trial.

This folder decides what to measure: dwell time, skip rate, reading time,
opening scan, pupil size. It does not know what the eye tracker's raw exports
look like -- that is `ingest/` -- and it never imports from `ingest/`.

    fixations.py      reads a fixation report and places each fixation on a word
    area_metrics.py   the per-area measurements, for both screens
    reading_times.py  how long each area was looked at, three ways
    sequences.py      the order areas were looked at, and statistics of it
    strategies.py     the opening scan and each person's dominant one
    pupil.py          pupil size in mm, z-scored per person
    last_visited.py   which area was looked at last
    preference.py     did the person choose the answer their eyes favoured
    scope.py          splitting Study 1 into hunters and gatherers
    build.py          puts it all together into the final model table
    paragraph/        the same measurements for the paragraph screen
"""
