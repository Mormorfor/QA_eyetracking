"""Third-party code kept as received, plus the thin wrappers that call it.

One rule governs this folder: **nothing on the live path may import it at module
scope.** Every consumer imports lazily, inside the function that needs it, so a
breakage in vendored code cannot stop the rest of the project from importing.
This can be checked: after importing every module the paper's code uses, no
`src.vendor.*` entry is in `sys.modules`.

That is what makes it safe to drop in a newer upstream copy as it comes: if the
copy breaks, only the one call that imports it breaks.
"""
