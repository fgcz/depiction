# `alphapept`- and `awkward`-gated code has no coverage, and the ROADMAP points at it

Severity: **low** | Status: **accepted — documented** | Found: 2026-08-10
Files: `src/depiction/spectrum/matching/isotope_pattern_matcher.py:205`,
`src/depiction/spectrum/peak_filtering/filter_by_isotope_pattern.py`,
`src/depiction/tools/experimental/msi_hdf5.py`, `docs/refactoring/ROADMAP.md:490-492`

## Symptom

Two clusters of code that no environment can exercise:

1. **`alphapept`-gated isotope code.** `isotope_pattern_matcher.py:205` imports
   `alphapept.chem` / `alphapept.constants` *inside* a function, and `alphapept` is declared in
   no extra anywhere. The two test files guard with `pytest.importorskip("alphapept")`, so they
   skip in every environment and the code has zero coverage.

   Note the modules themselves **do** import fine — the dependency is lazy:

   ```
   $ .venv/bin/python -c "import depiction.spectrum.matching.isotope_pattern_matcher"; echo $?
   0
   ```

   An earlier draft of this finding claimed both modules were unimportable. That is wrong; only
   the `alphapept`-dependent function and its tests are dead.

2. **`msi_hdf5.py` cannot be imported at all** — `ModuleNotFoundError: No module named
   'awkward'`, also declared in no extra.

## Why it is filed as accepted

Both are experimental surfaces with no production callers. Deleting them is defensible but is a
judgement call about what belongs in the archive, not a defect to fix.

## The one thing worth correcting

`docs/refactoring/ROADMAP.md:490-492` cites `msi_hdf5.py` as "the one caller" of
`get_spectrum_n_points` (referenced from `imzy_reader.py:158`). Since that module cannot be
imported, the accurate statement is that **no live caller remains**. As written it sends a
successor hunting for a module that does not load, to understand a method that may no longer
need to exist.

Fix that sentence if you touch the ROADMAP for `015-stale-documentation-claims.md` anyway.

## Notes

If `alphapept` is worth keeping reachable, declaring it in an `isotopes` extra would at least
let the two test files run somewhere. It is a heavy dependency for two modules with no callers,
so leaving it undeclared is also reasonable — just not silently.
