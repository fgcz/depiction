# depiction: finish the aborted split, migrate I/O to `imzy`, leave it archivable

Status: **partially executed**, 2026-08-07. Author: Leonardo Schwarz.

Phases A, B, C, E, F and G are done, and so is the baseline diff that used to be the largest
open risk: the pipeline produces **identical output** before and after the migration, on two
real acquisitions, with the `.ibd` files byte-identical past their UUID header — see
[`baseline-diff.md`](baseline-diff.md). **Phase D — the upstream contributions to `imzy` — is
now the only work not started**, and all five of its gaps are worked around in-tree today. It
is described below as a plan for whoever picks it up; see
[What a successor needs](#what-a-successor-needs).

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
materially better imzML writer than the `pyimzml` one then in use.

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
on top of the imzy backend is cheaper than rebasing it.

### Phase C — imzy behind the existing protocols ✅

`pkgs/depiction_io/src/depiction_io/imzy_backend/` implements `GenericReadFile` /
`GenericReader` on imzy, and `ImzmlWriter` now writes through imzy instead of `pyimzml`.
The reader is **opt-in and not the default**; the writer flip is unconditional.

- **`ImzyReadFile` / `ImzyReader`.** Construction does no I/O, so the read file is cheap to
  hand to a worker process; `n_spectra` ← `n_pixels`, `coordinates` ← `xyz_coordinates`,
  and `get_spectra` is routed onto `_read_spectra(indices)`, which opens the `.ibd` once per
  chunk rather than once per spectrum. Checksums and pixel size go through the existing
  `FileChecksums` / `ParseMetadata` rather than imzy, which parses no checksums and reports
  a pixel size of 1 where the file declares none.
- **`imzml_scan.py` — one streaming pass, two answers.** It refuses compressed input, and
  it reports whether `IMS:1000052` is declared. The second half was not foreseen: imzy's
  `xyz_coordinates` is always `(n, 3)`, inventing `z = 1`, while the legacy reader returns
  `(n, 2)` for such a file — and `reader.coordinates[i]` is passed straight to
  `writer.add_spectrum` in five tools, so an invented z column would have propagated.
  Deriving it in the same walk as the compression check costs nothing and makes the two
  backends agree by construction rather than by test.
- **`backend.py`.** `get_read_file(path)` returned the legacy reader unless
  `DEPICTION_IO_BACKEND=imzy` or an explicit argument said otherwise; a non-imzML path went
  to imzy regardless, with a clear error on macOS where `imzy/plugins.py` disables the
  Bruker readers. At this point the tools still constructed `ImzmlReadFile` directly, so the
  environment variable did not redirect them — routing those call sites was deliberately
  left to Phase E rather than made an unverifiable change to every tool for the benefit of a
  backend not yet trusted. *Superseded by Phase E: there is one backend now and the
  environment variable is gone.*
- **Writer.** `pyimzml` is gone. Two imzy behaviours had to be corrected first — see Phase D
  gaps (4) and (5) — and imzy rewrites the output suffix to `.imzML` rather than using the
  path it was given, so `ImzmlWriter.open` rejects any other spelling instead of quietly
  writing somewhere else.

  **Two behaviour changes a successor should know about**, both consequences of imzy
  refusing to write nothing:

  - *An empty spectrum is refused by the writer.* pyimzml wrote it; imzy warns and drops
    it, which would desynchronise the pixel count; this adapter raises. The one caller that
    can produce one, `filter_peaks`, now drops the pixel explicitly and logs it — which is
    what its sibling `pick_peaks` has always done for the same situation. The policy sits
    in the tool, where it is visible, rather than in the I/O layer, where it was silent.
  - *Closing a writer with no spectra raises.* Reachable from `SubsampleImzml` with a ratio
    that rounds to zero and from `CutoutRectangularRegion` with an empty selection, both of
    which used to produce a file — though a malformed one, per the TODO this replaced. The
    error does not mask a failure raised inside the `with` body.
- **Tests.** `tests/differential/` now runs every assertion against both readers, and gained
  a pickle round trip, a writer-output module (`test_writer.py`), and a
  `WriteSpectraParallel` → `MergeImzml` round trip. The Bruker `.d` path is plumbed but
  **exercised by nothing**: it cannot run on macOS and there is no fixture.

Two pre-existing bugs surfaced while doing this; see *Known, still unfixed* below.

### Phase E — imzy as the default, custom parser deleted ✅

The reader is the default and the hand-rolled parser is gone: **−1,238 lines**
(`imzml_reader.py`, `imzml_read_file.py`, `parser/parse_spectra.py`, `parser/cv_params.py`,
`compression.py` and the two test modules for them, one of which was `@unittest.skip`ped in
its entirety and had been asserting nothing for a long time).

Done in four steps, each green under `nox`, with the legacy backend still registered in the
differential harness through the third — so the A/B parity suite was proving the two readers
indistinguishable at the exact commit that flipped the default.

- **zlib is read rather than refused.** This was the one hard blocker (Phase D gap 1), and
  not a hypothetical: the only real-world imzML in the repository, the chunks under
  `tests/integration/imzml_parser/`, is a zlib-compressed FGCZ export. `imzml_scan.py` now
  collects `IMS:1000104` encoded lengths during the walk it was already doing, keyed by the
  block's offset — which means the reader never has to work out which `binaryDataArray` was
  the m/z one, and a continuous file's shared m/z block collapses to one entry.
  `zlib_reader.py` subclasses imzy's `IMZMLReader` and inflates at the three places it turns
  bytes into floats; `_ENCODED_READ_SITES` pins that list so an imzy upgrade adding a fourth
  fails loudly rather than quietly returning noise again. Numpress stays refused — undoing
  it needs a codec, not a length. Both guards were mutation-checked.
- **The tools go through the seam.** ~35 `ImzmlReadFile(...)` constructions became
  `get_read_file(...)`, and the annotations that named a concrete reader became the
  protocols they actually require, now exported from `depiction_io`. Behaviour-preserving
  while the default was still legacy, which is what made the flip a one-line change.
- **`ImzyReadFile` was not quite a drop-in.** It had no `file_sizes_bytes`/`file_sizes_mb`
  and its `summary()` omitted the file-size and m/z-range lines, so flipping first would
  have silently shortened what `limit_mz_range` prints. Ported before the flip, and the
  two backends' `summary()` output was compared as whole strings while both existed.
- **`get_read_file` collapsed to one implementation**, taking `ReadBackend`,
  `DEFAULT_BACKEND` and `DEPICTION_IO_BACKEND` with it. The function stays: the choice it
  makes for a vendor format on macOS is still real.

**Kept, deliberately** — a successor should not "finish the job" by deleting these:
`parser/parse_metadata.py` is how `ImzyReadFile` gets checksums and a pixel size that is
`None` rather than `1` when the file declares none; `imzml_alignment_tracker.py` is still
used by the writer.

**What the differential suite lost.** It is no longer an A/B comparison, and its docstring
says so. It keeps most of its value because `corpus.Case` carries the source arrays
independently of any reader, so the assertions are backend-against-ground-truth, and
`RamReadFile` still cross-checks. What is genuinely gone is a second *XML parser* to
disagree with imzy: a bug shared symmetrically by this writer and this reader would no
longer show up there. Only real acquisitions can catch that class of thing, which is what
[`public-test-data.md`](public-test-data.md) and Phase G are for.

**Not done at the time, by decision:** the end-to-end `depiction_targeted_preproc` run diffed
against a pre-refactor baseline. Done since — see
[The pre-refactor baseline diff](#the-pre-refactor-baseline-diff-).

### Real-acquisition reader checks ✅ (PR #41)

The gap the paragraph above names, partly closed. `tests/real_data/` fetches the two
redistributable acquisitions from [`public-test-data.md`](public-test-data.md) into a
gitignored `.test-data/` and asserts what they read as — 38 tests, 8 s once the files are
local, skipped when they are not. Before this, the only check of imzy against third-party
data was an ad-hoc script that no longer exists, run against a reader that no longer exists;
its results survived only as a table in a Markdown file.

The expectations came from *both* readers while both existed, so a future disagreement is
evidence about imzy rather than about how the numbers were derived. Beyond re-asserting
them, the tests cover the declared `IMS:1000091` checksum (the only real-file exercise
`parse_metadata.py` has), `scan_imzml` on genuine vendor headers, the z-column decision
checked against the XML rather than against the scan, batched reads, and a
`ReadSpectraParallel` round trip.

**One recorded number was wrong**: the mouse-kidney file declares no pixel size at all —
its 150 µm raster is the deposit's description, not the imzML's. Corrected in
`public-test-data.md`, and it constrains any spatial assertion Phase G builds on that
fixture. Everything else reproduced exactly.

This does not close Phase G, and it is not the baseline diff. It covers the reader on two
files: not the writer, not the pipeline.

### Phase G — a public fixture the system tests actually run on ✅ (PR #43)

The system tests ran on one 1.26 GB non-redistributable acquisition and asserted four
constants copied from one of its outputs, so they skipped everywhere except one laptop.
Both halves of that are fixed: the assertions now derive from the inputs, and the 59 MB
public pair from [`public-test-data.md`](public-test-data.md) runs the whole
`CALIB_IMAGES` chain on every PR — **31 s in a warm environment, 65 s in a fresh one** where
numba still has to compile, against 104 s for the tonsil.

- **`system_tests/fixtures.py`** describes both acquisitions and is the only file to touch
  to swap one. It imports the public manifest from `tests/real_data/datasets.py` instead of
  repeating its URLs, which is what `system_tests/README.md` had been telling a successor to
  do since Phase F.
- **The panel is generated, and says so.** The FGCZ panel is B-Fabric dataset `53798` and is
  not ours to publish, so the public fixture uses the twenty strongest well-separated peaks
  of its own mean spectrum, regenerable by `system_tests/panels/make_mouse_kidney_panel.py`.
  It identifies nothing; it exists because
  `CalibrationMethodConstantGlobalShift.preprocess_image_features` takes `np.nanmedian` over
  per-reference peak distances, so a panel that finds no peaks yields a NaN shift, NaN m/z
  arrays, and an output that is quietly all background.
- **Three assertions the old test could not make**, and they are the ones that earn the run:
  the foreground mask equals the acquisition's coordinate set exactly; `calibrated.imzML`
  keeps `raw.imzML`'s spectrum count and coordinates; and no `IMS:1000052` appears in the
  output that was not in the input. That last one is the Phase C z-axis risk — *"one round
  trip would have made the change permanent"* — checked on a real end-to-end pipeline run
  rather than on the synthetic corpus, for the first time.

**And it immediately found something, which is the argument for having done it.** The first
run in a clean `nox` environment failed at import: `snakemake_invoke` was declared as
`git+https://github.com/leoschwarz/snakemake_invoke` with **no revision**. `uv.lock` pinned
commit `e7e3c33`, but the nox sessions install with `uv pip install .`, which does not consult
the lockfile — so every fresh environment resolved the git HEAD instead, and upstream had
since moved `SnakemakeInvoke` out of `__init__.py`.

The effect was invisible in three ways at once. The committed lockfile kept `.venv` working,
so it never reproduced locally. Nothing under `tests/` imports `process_chunk`, so the default
`nox` stayed green. And because the failure is an *import* error it happens at collection,
before any fixture can skip — meaning Phase F's *"the tests now skip with a message naming
what is missing"* had quietly stopped being true on a cold clone. Fixed by pinning the
declaration to the revision the lockfile already recorded; the relock changed two lines and
nothing else. **A repository intended to go dormant cannot carry an unpinned VCS dependency**,
and this is the only one.

**What the run still is not.** One calibration method, one artifact (`CALIB_IMAGES`), one
acquisition in CI, and no comparison against any earlier version of this code. It shows the
pipeline runs and that its output is consistent with its input; it does not show the output
is the same as it was before the migration. That is the baseline diff, immediately below.

### The pre-refactor baseline diff ✅

The risk this table used to call *"open, and now the largest remaining one by some
distance"*. Full write-up in [`baseline-diff.md`](baseline-diff.md); harness in
[`system_tests/baseline/`](../../system_tests/baseline/README.md).

**The migration changed nothing in the data.** The `CALIB_IMAGES` chain was run on both
fixtures in this tree and in the tree at `ed222b3`, and every artifact matches exactly, with
no tolerance: all 10131 spectra's m/z and intensity arrays, the coordinates, the image
values, the channel names, the calibration coefficients, the OME-TIFF and the SpatialData
zarr. The `.ibd` files are **byte-identical after their 16-byte UUID header** — `cmp -l`
reports exactly 16 differing bytes in a 3.8 GB file.

Three things made this a controlled experiment rather than a fishing trip, and they are what
a successor should copy if they ever repeat it:

- **There is one variable.** `git diff ed222b3 HEAD` over the whole chain — the `Snakefile`,
  every rule file, `artifacts_mapping.py`, `process_spectra`, `calibrate`,
  `generate_ion_image`, `ome_tiff.py`, `multi_channel_image.py` — is imports, annotations and
  comment rewrapping. The rule files are byte-identical. So a difference in the output could
  only have come from the reader or the writer.
- **The baseline environment is pinned to this tree's lockfile.** `ed222b3` predates
  `uv.lock`, and the recipe below anticipated "resolution drift, record what resolves".
  Exporting the current lock as a constraints file instead removed the drift entirely: 198
  shared packages, **zero version mismatches**, with `pyimzml` the only meaningful package
  unique to the baseline. A difference would have had nowhere else to come from.
- **The imzML pairs were compared under both readers**, imzy's and the deleted parser's.
  Phase E recorded that the differential suite had lost the ability to catch *"a bug shared
  symmetrically by this writer and this reader"*; a single-reader comparison here would have
  reproduced that blind spot rather than closed it. Both readers report identical.

Two side findings worth keeping:

- **The baseline tree does not install as written**, for the reason Phase G found: its
  `snakemake_invoke` has no revision and today resolves to a HEAD that moved `SnakemakeInvoke`
  out of `__init__.py`. The constraints file pins it. Anyone reconstructing a pre-Phase-A
  environment will hit this first.
- **pyimzml wrote a dangling reference in every imzML this pipeline ever produced**:
  `instrumentConfigurationRef="instrumentConfiguration0"` on every spectrum, while the only
  `<instrumentConfiguration>` it declared had `id="IC1"`. It also used `cvRef="IMS"` without
  declaring the IMS ontology in `cvList`. imzy is correct on both counts. The one thing imzy
  does *worse* is writing the pixel count into `IMS:1000044`/`1000045` (max dimension x/y),
  which should be a physical length; nothing here reads them.

**What it does not cover** is in the report: one artifact, one calibration method,
`process_spectra` with no steps, no compressed input, no vendor format, two continuous 2D
acquisitions. And it says the current code agrees with the code it replaced — not that either
is correct.

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

- `src/depiction/tools/create_imzml_pool.py:67` has an orphaned
  `str(imzml_file.imzml_file.absolute())` whose result is discarded, leaving `@abs_path`
  undefined in the pandas query two lines below. Pre-existing; out of scope for the work
  above.
- ~~**`ImzmlReader.get_spectrum_n_points` returns bytes, not points.**~~ **Fixed by
  deletion in Phase E.** The legacy reader reported `IMS:1000104`, the encoded length, so
  for an uncompressed float32 array its answer was four times the truth; the imzy backend
  reports `IMS:1000103` and is correct. Note that this is a **behaviour change** for the
  one caller, `depiction.tools.experimental.msi_hdf5`, which has been getting four times
  the right number and is not covered by any test.
- **`WriteSpectraParallel` does not propagate dtypes.** Its chunk files are opened with
  `ImzmlWriteFile`'s defaults, so a float64 intensity array comes back as float32 after a
  parallel round trip. Pre-existing, pinned by `tests/differential/test_writer.py` rather
  than fixed.
- **A file with no declared pixel size silently gets 1 µm.** `Metadata.pixel_size` is a
  required `PixelSize`, but `ParseMetadata.pixel_size` returns `None` when the file declares
  no `IMS:1000046` — deliberately, since that was the whole reason Phase E kept the parser.
  So `proc_export_raw_metadata` takes its `ValidationError` branch, logs *"Failed to extract
  metadata"* (an overstatement: only the pixel size was missing), and substitutes a dummy
  1 µm that reaches the OME-TIFF. Found in Phase G, because the public fixture is the first
  file in this repository that declares no pixel size; the tonsil never reaches the branch.
  Pinned by `test_pixel_size_matches_the_exported_raw_metadata` rather than fixed — the fix
  is `PixelSize | None` through `Metadata`, `OmeTiff.write_image` and `OmeTiff.write`, which
  is a `depiction_io` API change and was outside Phase G.

---

## What a successor needs

### Phase D — Upstream contributions to imzy (not started)

**Five** gaps, not the three originally identified — and gap (3) turned out to be three problems wearing one coat. Each is independently useful; open
them as issues first, with the failing case. All five are worked around in-tree today,
behind `# upstream:` markers.

1. **No zlib support — was the one hard blocker; now worked around in full.**
   `_read_spectrum` is a bare `np.frombuffer(mz_bytes, dtype=self.mz_precision)`; nothing
   anywhere in imzy inspects `MS:1000574`, and `byte_offsets` stores array lengths, not
   encoded lengths. A zlib-compressed `.ibd` therefore **silently produces noise, not an
   error**. Reader half only — neither toolchain writes compressed output.

   Phase E closed this in-tree: `imzml_scan.py` collects `IMS:1000104` encoded lengths
   keyed by offset during the walk it was already doing, and `zlib_reader.py` subclasses
   `IMZMLReader` to inflate at the three places imzy turns bytes into floats. Upstream would
   want it differently — `process_spectrum` parsing `IMS:1000104` and the cache format
   carrying it — so the patch is not directly portable, but the failing case and the
   silent-corruption behaviour are still worth reporting as a bug in their own right.
2. **Readers are not picklable.** No `__getstate__`/`__setstate__`, and
   `WriteSpectraParallel` needs them. Note that `BaseReader._get_reader_kwargs()` returns
   `{}` and `IMZMLReader` does not override it, so for imzML this is simply "re-open by
   path" — simpler than it first appears.
3. **The `.icache` sidecar is unvalidated, unrelocatable and racy.** Three problems in one
   file, the first of which is the most dangerous thing found in the whole migration:
   - **No provenance check.** `_init` reloads `<stem>.icache` on sight — no size, mtime or
     UUID comparison — so a cache left from an earlier file at the same path makes imzy
     report *that* file's spectrum count and coordinates while slicing the *current*
     `.ibd`. Reproducible in four lines, and reachable by any pipeline that regenerates an
     output it has already read. Worked around on both sides: `ImzmlWriteFile` deletes the
     cache before writing, and `ImzyReader` deletes one that predates its `.imzML`. Storing
     the `.ibd` UUID in the cache would fix it properly.
   - **No `cache_dir`.** Read-only input trees degrade to a full re-parse on every open,
     and `_write_icache_safely` swallows the failure.
   - **A fixed temp name.** `write_icache` always writes `<stem>.icache.npz` before
     renaming, so concurrent first reads of one file — exactly what `ReadSpectraParallel`
     does — clobber each other. Benign in practice, since the `OSError` is swallowed and
     the rename is atomic, but it should be `mkstemp`-based.
4. **The writer silently drops empty spectra.** `IMZMLWriter.add_spectrum` catches
   `_EmptySpectrumError`, warns and returns `False` — *before* consulting its own
   `on_error="error"` setting. `filter_peaks` can emit empty spectra today and `pyimzml`
   writes them, so a naive writer flip would drop pixels and desynchronise `n_spectra`
   from `coordinates`. Worked around in `ImzmlWriter.add_spectrum`, which refuses an empty
   m/z array before delegating and treats a `False` return as an error.
5. **The writer always emits a z coordinate.** `_normalize_coordinates` appends `z = 1` to
   a 2D coordinate and `_add_scan_list` writes `IMS:1000052` unconditionally, where
   `pyimzml` writes it only for a 3-tuple. Found during Phase C and **not** in the original
   survey. It is the more dangerous of the two writer gaps: it would have added a z axis to
   every acquisition the pipeline produces, and since `reader.coordinates[i]` is handed
   straight back to `add_spectrum` in five tools, one round trip would have made the change
   permanent. Worked around in `DepictionIMZMLWriter`, which removes the cvParam again when
   no spectrum was given 3D coordinates — the one place where this codebase reaches into
   imzy's private methods, so it fails loudly if either method changes.

Assume upstream review is slow. **Do not block on merge:** carry each fix in the adapter
(or a pinned fork) and leave a `# upstream: <PR url>` marker at every carried patch.

### Phase E — done

See [above](#phase-e--imzy-as-the-default-custom-parser-deleted-). One item from the
original plan was **not** taken, and is still available:

- *Optional, skippable:* `GenericReader.imzml_mode` / `ImzmlModeEnum` is imzML-specific
  naming that becomes nonsense now that Bruker `.d` readers are reachable. Renaming to
  `spectra_mode` / `SpectraMode` is mechanical (33 sites) but pure churn.

### Phase G — done

See [above](#phase-g--a-public-fixture-the-system-tests-actually-run-on--pr-43). The four
steps as originally written are kept below, each annotated with what actually happened,
because the gap between the plan and the execution is the useful part.

Last, because it is independent of everything above and only worth doing once the I/O
layer has stopped moving. Phase F established that the system tests
([`system_tests/README.md`](../../system_tests/README.md)) cannot run in CI as written:
their only fixture is a 1.26 GB FGCZ-local acquisition, and the assertions
(`128 x 137` pixels, 118 channels, 10131 non-zero) are pinned to that one file. The test
therefore skips everywhere except one laptop, which is indistinguishable from not having
it.

The fix is a redistributable acquisition that the test suite fetches on demand:

1. **Find a public imzML/ibd pair** — **done, 2026-08-07**: an MIT-licensed, DOI-stable
   59 MB pair with checksums and measured geometry recorded in
   [`public-test-data.md`](public-test-data.md), along with what was ruled out and why.
   The criteria this had to meet are kept below as written.

   Find a pair that is small enough to download per CI run (target
   well under 100 MB — a single small tissue section or a cropped acquisition), openly
   licensed, and served from a stable, citable location. Candidate sources, in rough order
   of how likely they are to give a permanent URL:
   [METASPACE](https://metaspace2020.org), Zenodo, PRIDE, and the example data shipped by
   other MSI toolchains (`pyimzml`, `imzy`, Cardinal, SCiLS). Record the DOI/accession, the
   licence, and a SHA-256 next to the download code — an unpinned fixture is a test that
   changes underneath you.
   *If nothing suitable exists, cropping and re-publishing a slice of an in-house
   acquisition under CC-BY is a legitimate fallback, but it needs the data owner's sign-off.*
2. **Add a cached download**, not a committed file — **done**: `tests/real_data/fetch.py`
   downloads both public pairs into `.test-data/` (`DEPICTION_TEST_DATA_DIR` overrides it),
   verifies SHA-256 before renaming anything into place, and `tests/real_data/conftest.py`
   skips — never fails — when a file is absent.

   Two deviations from this step as originally written, both deliberate. The manifest is a
   typed module (`tests/real_data/datasets.py`) rather than an entry in
   `system_tests/inputs/inputs.yml`, because it also carries the *expected reading* for each
   file and nothing parses `inputs.yml` today; `system_tests` should import it rather than
   duplicating the URLs. And the cache is repo-local rather than under `XDG_CACHE_HOME`, so
   that 1.24 GB of fixtures is somewhere a person will find and delete. `system_tests`
   imports that manifest rather than repeating its URLs.
3. **Rewrite the assertions** so they hold for the fixture — **done**, and it was the
   smaller half rather than "the real work" this step predicted. All four constants turned
   out to be readable out of the work directory: `tonsil.imzML` declares
   `<spectrumList count="10131">`, which is `n_nonzero` exactly, because
   `SparseRepresentation.flat_to_spatial` derives `is_foreground` from the coordinate list
   and not from the values; `118` is the panel row count and `128 x 137` the coordinate
   bounding box.

   One deviation, in the direction of a stronger test. The pixel *count* became a set
   comparison — the foreground mask must equal the acquisition's coordinate set — because a
   count is satisfied by the right number of wrong pixels, a transposed or shifted image
   among them. Doing that surfaced a detail worth knowing: the OME-TIFF round trip drops the
   x/y coordinate labels (`OmeTiff.read` assigns only `c`), so the mask comes back indexed
   from zero and the acquisition's origin has to be added back.
4. **Enable the CI job** that Phase F replaced with a comment in
   `.github/workflows/pr-checks.yml` — **done**, as a separate job rather than another `nox`
   session, so a pipeline run never sits in front of the lint and unit-test feedback. The
   download is cached on `hashFiles('tests/real_data/datasets.py')`: keying on the manifest
   rather than on a checksum copied into the workflow means a URL change invalidates the
   cache too, not only a content change.

Keep the FGCZ tonsil path working alongside it — a large real acquisition is still the
better regression test, it just cannot be the *only* one.

### Cost to be aware of

`imzy` pulls roughly 40 transitive packages — matplotlib, h5py, hdf5plugin, mpire,
requests, numba, and the same author's `koyo` / `ims-utils` / `yoki5`. All licences are
permissive (BSD-3 / MIT / Apache-2.0), so `nox -s licensecheck` survives, but
`depiction_io`'s "minimal I/O package" property does not. **Paid in Phase C.** One knock-on
effect: imzy depends on `numba` with no lower bound, and a fresh resolve of `depiction_io`
alone then walks back to numba 0.53 (2021), which cannot build on Python 3.13 — hence the
explicit `numba>=0.61` floor in `pkgs/depiction_io/pyproject.toml`, which is not an import
dependency.

---

## Verification

```bash
uv sync --extra testing --extra dev
nox                                           # lint + both test suites + licensecheck + docs
uv run pytest tests/differential -v           # the reader against the corpus's ground truth
uv run pytest tests/unit/parallel_ops -v      # pickling across process boundaries
nox -s system_tests                           # real data, slow, skips without the local fixture
depiction-tools --help
```

Dependency isolation — the check that proves the split is real, not cosmetic:

```bash
uv pip show depiction    | grep -i 'pyimzml\|bioio'   # must be empty
uv pip show depiction_io | grep -i 'bioio\|tifffile'  # must be empty
uv pip list | grep -i pyimzml                         # must be empty since Phase C
```

`pytest tests/differential` was an A/B comparison between the two readers for the length of
the migration. There is one reader now, so it compares against the corpus's independently
held source arrays instead; see the note in Phase E about what that no longer catches.

---

## Risks

| Risk | Mitigation |
|---|---|
| Zlib-compressed input → imzy silently reads noise | **Closed.** `zlib_reader.py` inflates at every read site, `_ENCODED_READ_SITES` fails loudly if imzy grows another one, and the zlib twins run the whole parity suite. Mutation-checked both ways. Numpress still refuses rather than guesses |
| imzy's writer silently drops empty spectra | Adapter-side guard that raises before delegating and on a `False` return; upstream PR (4) |
| imzy PRs not merged quickly | Carry patches in-tree behind `# upstream: <url>` markers; never block a phase on review |
| imzy's writer adds a z axis to 2D files | Removed again in `DepictionIMZMLWriter`, pinned by `test_z_is_written_only_for_3d_input`; upstream PR (5) |
| `WriteSpectraParallel` chunk/merge breaks under the new writer | Flipped the writer while the reader was still known-good, so failures had one cause; a round-trip test now covers chunk-write → merge |
| Bruker readers unavailable on macOS | Expected — `imzy/plugins.py` disables them there. Local dev is imzML-only; the `.d` path is plumbed but untested, and `get_read_file` says so rather than failing obscurely |
| **No end-to-end run diffed against a pre-refactor baseline** | **Closed.** Both fixtures run through the `CALIB_IMAGES` chain in both trees produce identical output, with no tolerance, and the `.ibd` files differ only in their 16-byte UUID header. Two deviations from the recipe this row used to give made the result far stronger than it planned for: the baseline environment was pinned to this tree's lockfile instead of accepting resolution drift (198 shared packages, zero mismatches), and the imzML pairs were compared under *both* readers rather than only the new one. See [`baseline-diff.md`](baseline-diff.md) and [`system_tests/baseline/`](../../system_tests/baseline/README.md); the estimate of a day was about right |
| `get_spectrum_n_points` now returns points rather than bytes | A deliberate behaviour change, not a regression — the old answer was four times too large. Its only caller, `depiction.tools.experimental.msi_hdf5`, has no test and was never checked against the old value |
| `uv.lock` does not cover what CI actually installs | **Known, partly mitigated.** Every `nox` session installs with `uv pip install`, which resolves fresh rather than reading the lockfile, so the committed lock reproduces `.venv` but not CI. The only dependency where that could drift silently was the unpinned `snakemake_invoke`, now pinned to a revision (see Phase G); everything else is on PyPI with a version floor, so the exposure is ordinary upstream churn rather than an arbitrary git HEAD. Switching the sessions to `uv sync --frozen` would close it properly and was not done |
| The migration is abandoned mid-flight | No longer applicable. Phases A, B, C, E and F are done and green, there is one reader and one writer, and the parser that would have been the fourth half-finished refactoring is gone |
