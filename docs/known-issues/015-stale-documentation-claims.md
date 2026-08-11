# Four documents describe code that no longer exists

Severity: **low** | Status: **fixed** (#59) | Found: 2026-08-10
Files: `pkgs/depiction_io/README.md:9`,
`pkgs/depiction_io/src/depiction_io/imzml/parser/README.md`,
`src/depiction_targeted_preproc/README-old.md`,
`docs/refactoring/ROADMAP.md:671`, `docs/refactoring/public-test-data.md:217,257`

## Symptom

Individually trivial, collectively the thing that makes a dormant repo untrustworthy: a reader
who follows any of these ends up hunting for code that was deleted.

1. **`pkgs/depiction_io/README.md:9`** tells downstream users that `depiction` carries a
   B-Fabric integration. It was removed in `68938d9`. The paragraph was authored in `26d44bb`,
   a *descendant* of that commit — so the sentence was false the day it was written.

2. **`imzml/parser/README.md`** still advertises the hand-rolled imzML parser deleted in
   Phase E, and a Rust rewrite that exists nowhere. Only `parse_metadata.py` survives in that
   directory.

3. **`src/depiction_targeted_preproc/README-old.md`** is superseded design debris describing
   artifacts (`panel_csv`, `calibration_q5.hdf5`, …) that were never built. It sits next to a
   `README.md` whose entire content is `TODO: Add a description`, so it reads as the
   authoritative document.

4. **`ROADMAP.md:671`** prescribes a verification step that can never pass:
   `uv pip show depiction | grep -i 'pyimzml\|bioio'   # must be empty`. `bioio` is a
   deliberate runtime dependency of `depiction` (`pyproject.toml:36-39`), so the check always
   reports a failure. The two following lines behave as documented.

5. **`public-test-data.md:217,257`** still calls the pre-refactor baseline diff "still open"
   and "the largest untested seam". It was run and passed — see `baseline-diff.md`.

## Fix

- (1) drop the clause naming the B-Fabric integration.
- (2) replace the file with one sentence: only `parse_metadata.py` remains, and why.
- (3) delete `README-old.md`; give `README.md` two sentences of actual description.
- (4) narrow the grep to `pyimzml`.
- (5) point both sentences at `baseline-diff.md`.

## Notes

Item 4 is the one worth doing even if you skip the rest — a successor following the ROADMAP's
own verification block would conclude the migration is incomplete.
