"""Predicting answer reading time from the paragraph text. **Parked.**

Out of scope for this paper (`research-context.md` section 6): it is not working well, and
draft2's "Paragraph associations" heading is empty because of it. Its paragraph
extraction *was* load-bearing; T6.1 moved that to `features/paragraph/spans.py` on
2026-09-23, and the compatibility shim was deleted in stage E. Nothing live imports
this package.
"""
