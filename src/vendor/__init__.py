"""Third-party code kept as received, plus the thin wrappers that call it.

One rule governs this folder: **nothing on the live path may import it at module
scope.** Every consumer imports lazily, inside the function that needs it, so a
breakage in vendored code cannot stop the rest of the project from importing.
The acceptance check is mechanical -- import every live module and assert no
`src.vendor.*` entry appears in `sys.modules`.

That rule is what makes "replace wholesale" affordable: a newer upstream copy can
be dropped in without regard for whether it still parses under this project's
Python, because the blast radius is one lazily-imported call site.
"""
