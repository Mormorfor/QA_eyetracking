"""The project's Python code. Run from the repo root; see docs/data-pipeline.md for build order."""

import sys

# Windows consoles default to cp1252, which cannot encode several characters this
# project prints -- the "done" tick, "<=" in threshold group labels, arrows in
# sequence output. On 2026-10-07 that crashed an L1 rebuild on its very last line,
# AFTER an hour of work and after every output file had been written, with a
# UnicodeEncodeError from a decorative checkmark.
#
# This sets the console encoding, not the data. Nothing analytical changes: files
# are written by pandas with its own encoding, and `errors="replace"` only ever
# affects what reaches the terminal. The alternative -- stripping the characters --
# WOULD change data, because some of them are category labels that end up in saved
# tables and figure legends (`<= 4` vs `> 4` in the correctness-threshold family).
#
# It is here rather than left to the caller because the repo ships publicly and has
# to run from scratch on a clean machine; "remember to set PYTHONIOENCODING" is not
# something a reader can be expected to know.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Not a reconfigurable text stream (captured, redirected to a buffer, or a
        # notebook's own writer). Those handle unicode themselves.
        pass
