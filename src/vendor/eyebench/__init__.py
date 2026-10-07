"""Trial-level paragraph feature extraction from the lab's EyeBench / OneStop pipeline.

Everything EyeBench lives here -- the code as received, the configuration it
needs, our adapter, and the driver -- by decision (Diana, 2026-10-06): one place,
and it must not be able to break the rest of the project. Before the Stage C
restructure these four were split across `src/external/EyeBench/` and
`src/derived/external/EyeBench/`, which made the wrapper look like vendored code
and the vendored code look like ours.

| file | whose | what |
|---|---|---|
| `utils - paragraph feature extraction.py` | theirs | as received. Not importable by name (spaces, dash) -- loaded by path, see below |
| `configs/` | theirs, trimmed | only the names the file above imports, copied from that project's `src/configs/` |
| `paragraph_trial_features.py` | ours | bridges our raw paragraph reports to what the extraction expects |
| `runner.py` | ours | cache-aware driver: pick participants, reuse the cached CSV or rebuild |

**Two local changes to the vendored file, both recorded so a fresh copy can
replace it:** an f-string reflowed onto one line (multi-line f-string expressions
need Python 3.12; this env is 3.11), and the header comment at the top, which is
ours.

Usage -- import lazily, never at module scope:

    from src.vendor.eyebench.runner import get_paragraph_trial_features
    trial_features = get_paragraph_trial_features()

A full build reads two multi-GB reports and takes ~1.5 h, so the driver loads the
cached `PARAGRAPH_TRIAL_FEATURES_PATH` unless `rebuild=True`.
"""
