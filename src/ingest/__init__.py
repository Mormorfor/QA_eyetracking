"""Raw eye-tracker reports -> tidy tables, one row per word.

This folder knows what the eye tracker's exports look like: their formats, the
"." it writes for a missing value, how trials and sessions are identified, how
to rebuild the button presses, and which screen area each word sits in. It does
not decide what to measure -- that is `features/`.

    build.py          the pipeline: runs everything below in order
    registry.py       the list of features the pipeline builds
    readers.py        loads the raw reports
    geometry.py       which screen area each word is in
    base_features.py  the per-word features (item ids, correctness, areas, ...)
    clicks.py         when each answer was selected and confirmed
    knowqa.py         converts and cleans Study 2's reports, then runs build.py

There is one pipeline for every dataset. What differs between datasets is
written on each dataset's record in `config/datasets.py`, not branched on here.
"""
