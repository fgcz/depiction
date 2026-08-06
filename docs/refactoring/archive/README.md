# Archive: rescued artefacts from the aborted refactorings

Everything here was **untracked** and existed in exactly one place on disk before being
copied in on 2026-08-06. It is kept for reference only — nothing here is on the import
path, and the `.rescued` suffix exists to make that obvious.

See [../ROADMAP.md](../ROADMAP.md) for what is planned to be done with it.

## Source code (from the `separate-depiction-io` worktree, branch at `c0f7951`)

That branch's two commits *deleted* these files from `src/depiction/persistence/image/`
but the replacement copies under `src/depiction/image/` were never `git add`ed — so the
branch does not import on a fresh clone, and these were the only copies in existence.
They resolve the `MultiChannelImage` circular-dependency knot and are reused verbatim in
roadmap Phase B.

| File | Was destined for |
|---|---|
| `ome_tiff.py.rescued` | `src/depiction/image/ome_tiff.py` |
| `hdf5_image_format.py.rescued` | `src/depiction/image/hdf5_image_format.py` |
| `test_ome_tiff.py.rescued` | `tests/unit/image/test_ome_tiff.py` — differs from the tracked `tests/unit/persistence/image/test_ome_tiff.py` by one import line only |

## Plan documents

| File | Origin | Status |
|---|---|---|
| `2026-03-separate-depiction-io-plan.md` | `feats/separate-depiction-io/shimmying-painting-pond.md` | ~80% executed, superseded by `../ROADMAP.md`. Its Phase 1 and its circular-dependency resolution are still correct and get reused. |
| `2025-03-multi-channel-image-plan.md` | `.local/refactor_depiction_image.md` | Never started. Splits `MultiChannelImage` into masked/unmasked and renames `AlphaChannel`→`MaskChannel`. **Deliberately out of scope** for the current roadmap. Only other trace is branch `better-bg-handling`. |

The `feats/separate-depiction-io` worktree was removed after this rescue. It held nothing
else of value: its one modified tracked file (`tests/unit/tools/test_generate_ion_image.py`)
carried TIC tests that already landed on `dev` in commit `8900272`.
