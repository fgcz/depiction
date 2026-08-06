# depiction: finish the aborted split, migrate I/O to `imzy`, leave it archivable

Status: **partially executed**, 2026-08-06. Author: Leonardo Schwarz.

Phases A, B and F are done. Phases C, D and E are **not started** and are described
below as a plan for whoever picks this up; see [What a successor needs](#what-a-successor-needs).

## Context

`depiction` had accumulated **three mutually inconsistent, never-completed refactorings**
over ~14 months. None reached a green test run, and the debris was actively confusing.

| Attempt | Where | State |
|---|---|---|
| `split-packages` (Apr 2025) | tag `archive/split-packages` | 4-way split (`depiction_io` / `depiction_image_io` / `depiction_image_ops` / `depiction_tools`), 221 files, abandoned mid-move at "start moving image code" |
| `refactor_depiction_image.md` (Mar 2025) | [archive/2025-03-multi-channel-image-plan.md](archive/2025-03-multi-channel-image-plan.md) | `MultiChannelImage` → `MultiChannelImage` + `MultiChannelMaskedImage`, `AlphaChannel`→`MaskChannel`. Never started |
| `separate-depiction-io` (Mar 2026) | tag `archive/separate-depiction-io`, plan at [archive/2026-03-separate-depiction-io-plan.md](archive/2026-03-separate-depiction-io-plan.md) | 2-way split to a uv workspace, 108 files, ~80% done, **did not import** — the commits deleted `ome_tiff.py` and `hdf5_image_format.py` from `persistence/` but the replacement copies were never `git add`ed |

Separately, [`vandeplaslab/imzy`](https://github.com/vandeplaslab/imzy) (BSD-3, v0.3.0,
actively maintained) covers most of what `depiction.persistence` did and adds what it
lacked entirely: **Bruker `.d` (TSF/TDF/NeoFlex) and Waters `.raw`** readers, plus a
materially better imzML writer than the `pyimzml` one still in use.

**Goal:** land the two-package split, replace the hand-rolled imzML I/O with `imzy`, and
leave the repo coherent, reproducible and green — suitable for going dormant.

### Decisions taken

- **End state: archive / dormant.** Nobody actively maintains it after handover; it must
  be reproducible and coherent, not feature-complete.
- **Split scope: two packages** — `depiction_io` + `depiction`, uv workspace. Not the
  4-way split. The `MultiChannelImage` refactoring stays out of scope (archived, not
  executed).
- **imzy: adapter first, full replacement as the target.** Build the adapter behind the
  *existing* `GenericReader`/`GenericWriter` protocols, then close the upstream gaps, then
  delete the custom parser.
- **Archive hygiene (Phase F) was pulled forward, ahead of the imzy work.** Phase C only
  pays off if Phase E also lands, and Phase E is blocked on inputs that may never arrive.
  Phase F is unblocked and leaves the repo coherent regardless.

---

## What was done

### Day 0 — Rescue ✅

Two files existed **only** as untracked files inside a gitignored worktree. All four
rescued artefacts are in [`archive/`](archive/): `ome_tiff.py.rescued`,
`hdf5_image_format.py.rescued`, and the two archived plans.

### Phase A — Consolidate and de-risk ✅ (PR #36)

1. **Debris cleared.** The untracked `depiction_io/`, `depiction_image_io/`,
   `depiction_image_ops/`, `depiction_spectrum_ops/`, `depiction_tools/` egg-info dirs are
   gone, as are the merged and aborted branches. The two aborted attempts were **tagged
   rather than deleted** (`archive/split-packages`, `archive/separate-depiction-io`).
2. **CI made honest.** `ruff` is enforced through pre-commit and CI, and
   `[tool.ruff] target-version` now matches `requires-python` (it said `py39`, which made
   ruff report every `match` statement in the codebase as a syntax error).
3. **Differential-test harness built** — `tests/differential/`. `corpus.py` generates a
   parametrised corpus covering continuous × processed, float32 × float64 m/z and
   intensity, 2D × 3D coordinates, a single-spectrum edge case and a single-point-spectra
   edge case; `test_reader_parity.py` asserts byte-identical spectra, coordinates,
   `n_spectra` and `imzml_mode` over every registered backend.

   **This harness is the load-bearing deliverable of the whole plan.** Adding a second
   entry to `READ_FILE_BACKENDS` turns every test in it into an A/B parity check.
4. **Zlib handled without a specimen.** Nothing in either toolchain can *produce* a
   compressed `.ibd` — depiction's writer passes no compression argument to `pyimzml`, and
   imzy's writer emits `MS:1000576` unconditionally. So `corpus.compress_case` rewrites an
   uncompressed case into a zlib twin, whose expected contents are by construction exactly
   its uncompressed twin's. Any discrepancy is a decompression bug and nothing else. A real
   FGCZ specimen would only add provenance, not coverage.

**Six real bugs surfaced**, most by the lint gate or the harness rather than by reading:

1. **3D imzML files silently lost the z coordinate** — `if not position_z` on an
   `Element` with no children, which is falsy. Also matters for Phase C: imzy reads
   `xyz_coordinates`, so post-swap 3D files would have gained a column and looked like a
   regression.
2. `depiction_targeted_preproc/app_interface/dispatch_app.py` did not import at all —
   it referenced a `params_io` module that exists nowhere.
3. `GenericReadFile.is_checksum_valid` declared a method where every implementation and
   caller uses the property form.
4. `calibration/__init__.py` wrote `__ALL__`.
5. `cluster_hdbscan.py` computed `n_clusters` and then passed a hardcoded `10`.
6. `OmeTiff.read_image`/`write_image` were missing from the built docs (autodoc pointed
   at a `format_ome_tiff` module deleted months earlier).

### Phase B — The two-package split ✅ (PR #36)

Layout, as a uv workspace with `[tool.uv.workspace] members = [".", "pkgs/*"]`:

```
pkgs/depiction_io/src/depiction_io/
├── types.py            # GenericRead/Write{File,er} protocols
├── file_checksums.py
├── imzml_zip.py
├── pixel_size.py
├── imzml/              # reader, writer, mode enum, compression, parser/
└── ram/                # RamReadFile / RamReader / RamWriteFile
```

The two circular-dependency knots were resolved by moving *sideways*, not down:
`Hdf5ImageFormat` and `OmeTiff` take and return `MultiChannelImage`, so both live in
`src/depiction/image/`. Consequently `depiction_io` declares no `bioio*` and no
`tifffile` — the prior attempt left those stale and shipped a README advertising OME-TIFF
support the package no longer had.

`noxfile.py` gained `tests_depiction` / `tests_depiction_io`, the latter installing *only*
`depiction_io` so that an accidental dependency on `depiction` fails there rather than
being masked by the parent environment.

**Deliberately not done:** ~~a `depiction.persistence` deprecation shim~~. It would have
lived about two weeks before Phase F deleted it again, leaving a successor with two import
paths and no reason for either. All 134 import statements across 86 files were rewritten
in the same commit as the move. The concern behind the original correction — that the
prior attempt's one-commit rewrite is *why* nothing was independently reviewable — was
addressed differently: five commits, each green under `nox`, with the mechanical bulk
quarantined in one of them and verified by a grep gate plus unchanged test counts on both
sides of the split.

### Phase F — Archive hygiene ✅

- **README** corrected (it claimed *"Python 3.12 is required, 3.13 is not compatible
  yet"*, contradicted by `requires-python`, `.python-version` and CI) and now describes the
  workspace layout.
- **`uv.lock` committed**, so the environment reproduces from a cold clone. The single
  highest-value archive action.
- **Sphinx docs retargeted** at the new layout, and — more importantly — **built in CI**.
  `nox -s docs` runs `sphinx-build -W`. Nothing built the docs before, which is exactly how
  the stale `format_ome_tiff` autodoc target survived unnoticed.
- **`system_tests` resolved honestly.** The only fixture is a 1.26 GB acquisition that
  cannot be committed, and the assertions are pinned to it, so the CI job the old TODO
  promised is not possible as written. The tests now skip with a message naming what is
  missing instead of failing with `FileNotFoundError` for anyone outside FGCZ, the
  provenance is recorded in `system_tests/README.md`, and the dead commented-out CI job is
  replaced by the reason it is absent.
- **Branches culled.** See below.

#### Branch inventory

Everything with unmerged content was tagged before deletion; nothing was lost.

| Branch | Disposition |
|---|---|
| `optional-package`, `specify-dtype` | deleted — squash-merged, `git cherry` clean against `dev` |
| `spatial-dist-plot`, `tic-image` | deleted — merged as PRs #31 and #33; changed files verified byte-identical to `dev` |
| `opencode/nimble-wizard` | deleted with its worktree — zero commits ahead |
| `extract-physical` | tag `archive/extract-physical` — adds `physical_coordinates` parsing |
| `tiff-orientation` | tag `archive/tiff-orientation` — **byte-identical diff to `extract-physical`**, despite the name |
| `imzml-zip-pipeline` | tag `archive/imzml-zip-pipeline` — zip inputs for `depiction_targeted_preproc` |
| `better-bg-handling` | tag `archive/better-bg-handling` — 76 commits, Oct 2024 |
| `dev-calibration` | tag `archive/dev-calibration` — 16 commits, May 2024 |

Nothing was rebased and merged. Every branch with real work touches
`src/depiction/persistence/`, which Phase B deleted, so each would need a manual rebase
onto the new layout — worth doing only for code someone will maintain. Note that
`extract-physical`'s feature exists natively in imzy
(`BaseReader.get_physical_coordinates`, `x_pixel_size`, `y_pixel_size`), so re-deriving it
after Phase C would be cheaper than rebasing it now.

---

## What was deliberately not done

- **The deprecation shim** (above).
- **`ANN` and `D103` are not enforced by ruff.** ~210 pre-existing violations (missing
  annotations, missing docstrings on public functions) on a codebase heading for an
  archived state. What is enforced can actually be kept green, which is the point.
- **`TC` is not enforced.** Moving an import into a `TYPE_CHECKING` block breaks anything
  that resolves annotations at runtime, and this project does that in two places at once —
  pydantic models and cyclopts CLIs. 56 violations, all in that blast radius.
- **`UP042` (`class Foo(str, Enum)` → `StrEnum`) is ignored.** Not cosmetic: the two
  differ in `str()` and f-string interpolation (`"Foo.X"` vs `"x"`). All six occurrences
  are config enums flowing through pydantic models and snakemake YAML, so converting them
  is a serialization change needing its own commit and tests.
- **`system_tests` in CI** — impossible with a 1.26 GB non-redistributable fixture; see
  `system_tests/README.md`.
- **Phases C, D and E** — see below.

### Known, still unfixed

`src/depiction/tools/create_imzml_pool.py:67` has an orphaned
`str(imzml_file.imzml_file.absolute())` whose result is discarded, leaving `@abs_path`
undefined in the pandas query two lines below. Pre-existing; out of scope for the work
above.

---

## What a successor needs

### Phase C — imzy adapter behind the existing protocols (not started)

New module `pkgs/depiction_io/src/depiction_io/imzy_backend/`:

- `ImzyReadFile(GenericReadFile)` — holds a path, cheap and picklable. `n_spectra` ←
  `n_pixels`, `coordinates` ← `xyz_coordinates`, `pixel_size` ←
  `(x_pixel_size, y_pixel_size)`. Reuse the existing `FileChecksums` and
  `ParseMetadata.ibd_checksums` for `is_checksum_valid`; imzy parses no checksums and both
  classes are already backend-independent.
- `ImzyReader(GenericReader)` — wraps `imzy.IMZMLReader`. `imzml_mode` has to be derived
  the same way the legacy reader derives it (all m/z byte offsets identical → continuous),
  because imzy has no mode concept. Point `get_spectra` at `_read_spectra(indices)`, which
  opens the `.ibd` once per chunk; imzy seek/reads where depiction mmaps, and that is where
  the difference shows up under `ReadSpectraParallel`.
- Backend selection defaulting to legacy, with `.d` paths routing to imzy unconditionally
  (nothing else can read them) and a clear error on macOS, where `imzy/plugins.py` disables
  the Bruker readers.
- **The zlib guard lands in the same commit.** `ImzyReadFile` must detect `MS:1000574` on
  open and raise `NotImplementedError`. Silent noise is not an acceptable interim state,
  and this guard is what makes the migration safe to abandon mid-flight. Assert it against
  the corpus's existing `*_zlib` cases.

Then flip the writer, with the guard described in gap (4) below.

### Phase D — Upstream contributions to imzy (not started)

**Four** gaps, not the three originally identified. Each is independently useful; open
them as issues first, with the failing case.

1. **No zlib support — the one hard blocker.** `_read_spectrum` is a bare
   `np.frombuffer(mz_bytes, dtype=self.mz_precision)`; nothing anywhere in imzy inspects
   `MS:1000574`, and `byte_offsets` stores array lengths, not encoded lengths. A
   zlib-compressed `.ibd` therefore **silently produces noise, not an error**. Reference
   implementation: `imzml_reader.py`, `parse_spectra.py`, `compression.py`. Reader half
   only — neither toolchain writes compressed output. Worth reporting the silent-corruption
   behaviour as a bug in its own right, independent of the fix.
2. **Readers are not picklable.** No `__getstate__`/`__setstate__`, and
   `WriteSpectraParallel` needs them. Note that `BaseReader._get_reader_kwargs()` returns
   `{}` and `IMZMLReader` does not override it, so for imzML this is simply "re-open by
   path" — simpler than it first appears.
3. **`.icache` sidecar is written next to the input `.imzML`.** Read-only input trees
   degrade to a full re-parse on every open, and `_write_icache_safely` swallows the
   failure. A `cache_dir` argument and/or an `IMZY_CACHE_DIR` env var would fix it.
4. **The writer silently drops empty spectra.** `IMZMLWriter.add_spectrum` catches
   `_EmptySpectrumError`, warns and returns `False` — *before* consulting its own
   `on_error="error"` setting. `filter_peaks` can emit empty spectra today and `pyimzml`
   writes them, so a naive writer flip would drop pixels and desynchronise `n_spectra`
   from `coordinates`. Until this is fixed upstream, the adapter must raise on an empty m/z
   array before delegating, and must treat a `False` return as an error.

Assume upstream review is slow. **Do not block on merge:** carry each fix in the adapter
(or a pinned fork) and leave a `# upstream: <PR url>` marker at every carried patch.

### Phase E — Flip the default, delete the custom parser (not started, blocked)

**Blocked on inputs only the maintainer can provide:** a real dataset plus pre-refactor
baseline outputs. Without them there is no way to tell a writer regression from a
legitimate difference.

1. Make imzy the default backend; run the full suite, `system_tests`, and at least one
   real end-to-end `depiction_targeted_preproc` run diffed against a baseline.
2. **Delete** `depiction_io/imzml/parser/` (`parse_spectra.py`, `parse_metadata.py`,
   `cv_params.py`), `imzml_reader.py`, `compression.py`, `imzml_alignment_tracker.py`, and
   their tests. Retarget the integration tests under `tests/integration/imzml_parser/` at
   the imzy backend rather than deleting them — they encode real cvParam edge cases
   (cf. commit `46f3f56 "handle weird cvParam entries"`) worth keeping.
3. **Keep** `ram/` (no imzy equivalent), `file_checksums.py`, `imzml_zip.py`,
   `pixel_size.py`, `types.py`.
4. *Optional, skippable:* `GenericReader.imzml_mode` / `ImzmlModeEnum` is imzML-specific
   naming that becomes nonsense once Bruker `.d` readers exist. Renaming to `spectra_mode`
   / `SpectraMode` is mechanical (33 sites) but pure churn.

Net effect: roughly **−1,100 LOC of custom parsing**, `pyimzml` gone, Bruker `.d` support
gained.

### Cost to be aware of

`imzy` pulls roughly 40 transitive packages — matplotlib, h5py, hdf5plugin, mpire,
requests, numba, and the same author's `koyo` / `ims-utils` / `yoki5`. All licences are
permissive (BSD-3 / MIT / Apache-2.0), so `nox -s licensecheck` survives, but
`depiction_io`'s "minimal I/O package" property does not. Worth paying once, deliberately,
and only if Phase E follows.

---

## Verification

```bash
uv sync --extra testing --extra dev
nox                                           # lint + both test suites + licensecheck + docs
uv run pytest tests/differential -v           # reader parity over every registered backend
uv run pytest tests/unit/parallel_ops -v      # pickling across process boundaries
nox -s system_tests                           # real data, slow, skips without the local fixture
depiction-tools --help
```

Dependency isolation — the check that proves the split is real, not cosmetic:

```bash
uv pip show depiction    | grep -i 'pyimzml\|bioio'   # must be empty
uv pip show depiction_io | grep -i 'bioio\|tifffile'  # must be empty
```

From Phase C onward, additionally:

```bash
DEPICTION_IO_BACKEND=legacy uv run pytest tests -q
DEPICTION_IO_BACKEND=imzy   uv run pytest tests -q    # identical results
```

---

## Risks

| Risk | Mitigation |
|---|---|
| Zlib-compressed input → imzy silently reads noise | Hard guard raising `NotImplementedError` before any default flip; zlib twins permanently in the differential corpus; upstream PR (1) |
| imzy's writer silently drops empty spectra | Adapter-side guard that raises before delegating and on a `False` return; upstream PR (4) |
| imzy PRs not merged quickly | Carry patches in-tree behind `# upstream: <url>` markers; never block a phase on review |
| `WriteSpectraParallel` chunk/merge breaks under the new writer | Flip the writer while the reader is still known-good, so failures have one cause |
| Bruker readers unavailable on macOS | Expected — `imzy/plugins.py` disables them there. Local dev is imzML-only; gate Bruker tests on platform |
| The migration is abandoned mid-flight | Phases A, B and F are done and green, and left the repo better off on their own. The legacy backend stays the default until Phase E, so an abandoned Phase C is inert code plus this document, not a fourth half-finished refactoring |
