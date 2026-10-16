"""Trial-level paragraph feature extraction from the lab's EyeBench / OneStop pipeline.

Everything EyeBench is in this one folder -- the code as received, the configuration
it needs, our adapter and the driver -- and nothing outside it imports it at module
level, so it cannot break the rest of the project. Which file is whose:

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
