# `depiction_io` declares `cyclopts` for an entry point that no longer exists

Severity: **low** | Status: **fixed** (#59) | Found: 2026-08-10
File: `pkgs/depiction_io/pyproject.toml:20`

## Symptom

```toml
# only for the `parse_spectra` debugging entry point
"cyclopts",
```

`parse_spectra.py` was deleted in the imzy migration — it survives only in the untracked
`build/lib` tree (see `009-stale-build-dirs-resurrect-deleted-modules.md`). No source file in
the package imports `cyclopts`:

```
$ grep -rn cyclopts pkgs/depiction_io/src/
pkgs/depiction_io/src/depiction_io.egg-info/PKG-INFO:14:Requires-Dist: cyclopts
pkgs/depiction_io/src/depiction_io.egg-info/requires.txt:6:cyclopts
```

Both hits are generated metadata, not code.

## Why it matters

`depiction_io` advertises itself as the deliberately minimal package that downstream
applications should depend on. Carrying a CLI framework for a deleted debugging script pulls
extra transitive packages into every consumer's environment and undercuts that claim.

## Fix

Delete the two lines. Then re-run whatever import scan backs the "deliberately minimal" claim
so the assertion and the manifest agree again.
