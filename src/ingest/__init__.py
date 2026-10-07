"""Raw vendor exports -> tidy IA-level tables.

Knows what an EyeLink export looks like: report formats, the "." sentinel,
trial and session identity, button-click reconstruction, and the screen
geometry each word sits in. Does NOT know what a dwell proportion is -- that
is `features/`.

**One pipeline, two front ends**, which the old folder names hid: `knowqa.py`
is a front end that calls the shared `build.main()`, not a second pipeline.
What differs per dataset is declared on the `Dataset` record in
`config/datasets.py` (`skip_base_features`, `has_paragraph`), not branched on
here.

Where things came from, 2026-10-06 (stage C). Comments dated before then name
the old paths, so this is the key that resolves them:

    data_prep/data_csv_generation.py (part) -> readers · geometry · build
    data_prep/button_clicks_processing.py   -> clicks.py
    data_prep/know_qa_dataprep.py           -> knowqa.py
    features/build.py (2026-10-06, stage D step 1)
                                            -> base_features.py + registry.py
"""
