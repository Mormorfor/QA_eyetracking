"""Does the stored text match the words on screen? A problem diagnosis, not an analysis.

`text_alignment_investigation.ipynb`, beside this file, is the whole of it. It found that the
stored question and answer texts sometimes do not match what the presentation software
displayed -- in both studies -- so counting words in the stored text put some words in the
wrong screen area. Every such trial is drawn in the notebook, and its conclusions are at the
top.

Its findings no longer reach the data: the pipeline now takes each word's area from where it
sits on screen (`ingest/geometry.py`). `ingest/knowqa.py::check_text_alignment` still lists
the mismatched trials whenever KnowQA's reports are cleaned. Figures go to
`reports/explorations/text_alignment/`.
"""
