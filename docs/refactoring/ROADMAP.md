# depiction: finish the aborted split, migrate I/O to `imzy`, leave it archivable

Status: **proposed**, 2026-08-06. Author: Leonardo Schwarz.

## Context

`depiction` has accumulated **three mutually inconsistent, never-completed refactorings** over ~14 months. None reached a green test run, and the debris is now actively confusing.

| Attempt | Where | State |
|---|---|---|
| `split-packages` (Apr 2025) | branch + `origin/split-packages` | 4-way split (`depiction_io` / `depiction_image_io` / `depiction_image_ops` / `depiction_tools`), 221 files, abandoned mid-move at "start moving image code". Left the untracked top-level `depiction_*/` egg-info dirs in the working tree. |
| `refactor_depiction_image.md` (Mar 2025) | was `.local/`, now [archive/2025-03-multi-channel-image-plan.md](archive/2025-03-multi-channel-image-plan.md) | `MultiChannelImage` → `MultiChannelImage` + `MultiChannelMaskedImage`, `AlphaChannel`→`MaskChannel`. Never started. Only trace is branch `better-bg-handling`. |
| `separate-depiction-io` (Mar 2026) | worktree `feats/separate-depiction-io`, plan archived as [archive/2026-03-separate-depiction-io-plan.md](archive/2026-03-separate-depiction-io-plan.md) | 2-way split to a uv workspace, 108 files, ~80% done. **Does not import**: the commits deleted `ome_tiff.py` and `hdf5_image_format.py` from `persistence/` but the replacement copies in `src/depiction/image/` were never `git add`ed. Those two files are rescued in [`archive/`](archive/). |

The branch `split-depiction-io` is an **empty placeholder** — identical to `dev`, zero commits. Attempt #3 never started.

Separately, [`vandeplaslab/imzy`](https://github.com/vandeplaslab/imzy) (BSD-3, v0.3.0 Jun 2026, actively maintained) covers most of what `depiction.persistence` does and adds what it lacks entirely: **Bruker `.d` (TSF/TDF/NeoFlex) and Waters `.raw`** readers, plus a materially better imzML writer than the `pyimzml` one currently in use.

**Goal:** land the two-package split, replace the hand-rolled imzML I/O with `imzy`, and leave the repo coherent, reproducible and green — suitable for going dormant.

### Decisions taken

- **End state: archive / dormant.** Nobody actively maintains it after handover; it must be reproducible and coherent, not feature-complete.
- **Split scope: two packages** — `depiction_io` + `depiction`, uv workspace. Not the 4-way split. The `MultiChannelImage` refactoring stays out of scope (archived, not executed).
- **imzy: adapter first, full replacement as the target.** Build the adapter behind the *existing* `GenericReader`/`GenericWriter` protocols, then close the three upstream gaps, then delete the custom parser. The adapter is deliberately shaped so the last step is a deletion, not a rewrite.

> One honest caveat on the archive goal: under "dormant", a package split earns less than it would under "publish to PyPI". It is still worth doing here for one specific reason — `depiction_io` is exactly the blast radius of the imzy migration, so the split *is* the seam that makes the migration reviewable and reversible. Anything beyond that boundary (the 4-way split) would be churn without payoff, and is excluded.

---

## What already exists and should be reused

The seam is already there and is the right one. Do **not** invent new abstractions.

- **`src/depiction/persistence/types.py`** — four Protocols: `GenericReadFile` (cheap, picklable file handle) / `GenericReader` (open handle) / `GenericWriteFile` / `GenericWriter`. Every consumer (86 files, 134 imports) already talks to these, not to imzML directly. imzy plugs in *underneath* this layer.
- **`src/depiction/persistence/imzml/imzml_reader.py`** — mmap + `np.frombuffer` + `zlib`, picklable via `__getstate__`/`__setstate__`. This is the behaviour the imzy adapter must match, and the reference implementation for the upstream PRs.
- **`src/depiction/persistence/ram/`** — `RamReadFile`/`RamReader`/`RamWriteFile`. imzy has **no equivalent**; ~80 unit tests depend on it. Keep as-is, move into `depiction_io`.
- **`src/depiction/tools/simulate/generate_synthetic_imzml.py`** — reuse to build the differential-test corpus rather than committing binary fixtures.
- **`src/depiction/parallel_ops/{read,write}_spectra_parallel.py`** — the multiprocessing chunk/merge machinery. The *only* consumer that cares about picklability; the acceptance test for the imzy pickle work.
- **[archive/2026-03-separate-depiction-io-plan.md](archive/2026-03-separate-depiction-io-plan.md)** — its Phase 1 (workspace config) and its circular-dependency resolution are correct and get reused verbatim.

### imzy: verified capability assessment

Read directly from a clone of `main` (~10.2k LOC).

**Strictly better than what depiction has:**

- `IMZMLWriter` (`src/imzy/_writers/_imzml.py`) — dtype control, `ibd_mode` auto/continuous/processed, polarity, pixel size, image shape, coordinate origin, SHA-1 over the ibd, `on_error` policy, context manager. `pyimzml` offers none of this.
- Bruker TSF/TDF/NeoFlex readers via the Bruker SDK (`CDLL`). Linux + Windows only — **not macOS** (`src/imzy/plugins.py` hard-disables them on Mac). Fine for FGCZ Linux; local Mac dev gets imzML only.
- `get_ion_image(s)`, `get_tic`, `get_normalizations`, `to_hdf5`/`to_zarr` centroid extraction.

**Three concrete gaps vs. the current reader — these are the full-replacement blockers:**

1. **No zlib support — the one hard blocker.** `_read_spectrum` is a bare `np.frombuffer(mz_bytes, dtype=self.mz_precision)`; nothing anywhere in imzy inspects `MS:1000574`. A zlib-compressed `.ibd` **silently produces noise, not an error**. `depiction`'s reader handles `Compression.Zlib` (`imzml_reader.py:143-154`, flag parsed in `parse_spectra.py:107`).

   Compressed input is rare here but has happened. Rare + silent is the worst combination for a repo about to go dormant: a successor hits it in 2028 and gets plausible-looking noise with no diagnostic. This is a read-side concern only — `depiction`'s writer calls `pyimzml.ImzMLWriter` without compression args, and imzy's writer only emits `MS:1000576 "no compression"`, so nothing in either toolchain *produces* compressed output.
2. **Readers are not picklable.** No `__getstate__`/`__setstate__`. There *is* `BaseReader._get_reader_kwargs()`, so a pickle path is ~30 LOC — but it does not exist today, and `WriteSpectraParallel` needs it.
3. **Writes a `.icache` sidecar next to the input `.imzML`.** Read-only input directories degrade to re-parsing on every open (`_write_icache_safely` swallows the failure). No way to redirect the cache.

Minor: seek/read rather than mmap; single global `mz_precision`/`int_precision` (the same assumption depiction already makes); transitive deps on the same author's `koyo` / `ims-utils` / `yoki5` / `pluggy` ecosystem.

---

## Roadmap — 2 weeks

Work on a fresh branch off `dev`. `split-depiction-io` is empty; either reuse it or cut `io-refactor` from `dev`.

### Day 0 — Rescue before anything else ✅ done

Two files existed **only** as untracked files on disk inside a gitignored worktree. All four are now in [`archive/`](archive/):

```
archive/ome_tiff.py.rescued                       # load-bearing
archive/hdf5_image_format.py.rescued              # load-bearing
archive/2026-03-separate-depiction-io-plan.md
archive/2025-03-multi-channel-image-plan.md
```

Commit these before `git worktree remove feats/separate-depiction-io`.

### Phase A — Consolidate and de-risk (Days 1–2)

Goal: one clean starting point, and a safety net that makes every later step verifiable.

1. **Clear the debris.** Delete untracked `depiction_io/`, `depiction_image_io/`, `depiction_image_ops/`, `depiction_spectrum_ops/`, `depiction_tools/` (egg-info residue only, no source). Delete branches `split-packages`, `separate-depiction-io`, `refactor-snakemake-invoke`, `separate-app-runner`, `tmp/20241105_del1`, `dev-deploy-batch` (all zero-ahead of `dev` or superseded) — after the Day-0 rescue is committed. Push the deletion of `origin/split-packages`.
2. **Get CI honest.** `.github/workflows/pr-checks.yml` runs `nox` (lint + tests + licensecheck) on py3.13. Enable the `ruff` pre-commit hook currently commented out in `.pre-commit-config.yaml`, fix the fallout, and set `[tool.ruff] target-version = "py313"` (currently `py39`, contradicting `requires-python >= 3.13`). Add the refactor branch to the CI trigger list.
3. **Build the differential-test harness.** `tests/differential/` — a parametrised corpus generated by `generate_synthetic_imzml.py` covering: continuous × processed, float32 × float64 m/z and intensity, zlib × uncompressed, 2D × 3D coordinates, empty spectra. For each file, assert that every `GenericReader` implementation returns byte-identical `get_spectrum_mz/int` and identical `coordinates`, `n_spectra`, `imzml_mode`.

   **This harness is the load-bearing deliverable of the whole plan.** Everything after it is "swap a backend and watch the harness stay green."
4. **Capture a zlib specimen.** Compressed input is rare but confirmed to occur, so gap (1) is a blocker, not a footnote — the task is evidence, not a go/no-go. Sweep for a real one:
   ```
   grep -lc 'MS:1000574' <dataset>/*.imzML   # zlib compression cvParam
   ```
   Add whatever turns up (or a synthetic equivalent, if no real file is still around) to the differential corpus as a permanent fixture. A synthetic file is enough to drive the work; a real one is better because it also pins down which vendor/export path produces them.

**Exit criteria:** `nox` green on a clean tree; differential harness green against the existing reader; one branch, no ghost dirs.

### Phase B — The two-package split (Days 3–4)

Reuse the archived plan's Phases 1–5, with the two corrections its own execution got wrong.

**Layout** — uv workspace, `[tool.uv.workspace] members = [".", "pkgs/*"]`:

```
pkgs/depiction_io/src/depiction_io/
├── types.py            # GenericRead/Write{File,er} protocols
├── file_checksums.py
├── imzml_zip.py
├── pixel_size.py
├── imzml/              # reader, writer, mode enum, compression, parser/
└── ram/                # RamReadFile / RamReader / RamWriteFile
```

`depiction_io` owns `pyimzml` (→ later `imzy`), `numpy`, `xarray`, `pydantic`, `loguru`, `tqdm`. The root `depiction` drops them.

**The two knots**, resolved the way both prior attempts concluded:

- `Hdf5ImageFormat` ↔ `MultiChannelImage` is genuinely circular → `Hdf5ImageFormat` moves to `src/depiction/image/`, not into `depiction_io`.
- `OmeTiff.read_image/write_image` take/return `MultiChannelImage` → also moves to `src/depiction/image/`. **Consequently drop `bioio`, `bioio-ome-tiff`, `tifffile` from `depiction_io`'s deps** — the prior attempt left them stale, and left the `depiction_io` README advertising OME-TIFF support it no longer had.

**Corrections to the prior execution:**

- **Keep `src/depiction/persistence/__init__.py` as a deprecation shim** re-exporting from `depiction_io` with a `DeprecationWarning`. The prior attempt deleted it and rewrote all ~90 call sites in one commit, which is why nothing was independently reviewable. Rewrite call sites in a *separate* commit, then drop the shim in Phase F.
- Watch the `mv` target-exists trap that produced `pkgs/depiction_io/tests/unit/imzml/imzml/` and `.../ram/ram/` last time.

Split `noxfile.py` into `tests_depiction` / `tests_depiction_io` sessions; update CI to run both.

**Exit criteria:** `uv sync` clean; both test suites green; `uv pip show depiction | grep pyimzml` empty; `depiction-tools --help` works.

### Phase C — imzy adapter behind the existing protocols (Days 5–7)

New module `pkgs/depiction_io/src/depiction_io/imzy_backend/`:

- `ImzyReadFile(GenericReadFile)` — holds a path, cheap and picklable; `get_reader()` opens an `imzy` reader. Maps `n_spectra`←`n_pixels`, `coordinates`←`xyz_coordinates`, `pixel_size`←`(x_pixel_size, y_pixel_size)`.
- `ImzyReader(GenericReader)` — wraps `imzy.BaseReader`. `get_spectrum_mz/int` over `_read_spectrum`; `imzml_mode` derived from whether m/z arrays are shared. Implements `__getstate__`/`__setstate__` **in the adapter** via `_get_reader_kwargs()` + path — this is the prototype for upstream PR #2.
- `ImzyWriteFile(GenericWriteFile)` / `ImzyWriter(GenericWriter)` — wraps `imzy.IMZMLWriter`. Maps `ImzmlModeEnum` → `ibd_mode`, forwards `pixel_size`.
- Backend selection: a `DEPICTION_IO_BACKEND` env var / `get_read_file(path)` dispatcher, defaulting to the legacy reader. `.d` paths route to imzy unconditionally (nothing else can read them).

Run the differential harness with both backends. **Expected failures at this point: zlib files.** Record them as `xfail` pointing at the upstream issue — do not paper over them.

**Install the zlib guard in the same commit.** Before Phase D lands anything, `ImzyReadFile` must detect `MS:1000574` on open and `raise NotImplementedError` with a pointer to the upstream issue. Silent noise is not an acceptable interim state, and this guard is the thing that makes the rest of the migration safe to leave half-finished if the two weeks run out. Assert it in the differential harness so it cannot regress.

Then **flip the writer first**: make `ImzmlWriteFile` use `imzy.IMZMLWriter` by default and drop `pyimzml` from `depiction_io`. The writer has no gaps and is strictly better, and `WriteSpectraParallel`'s chunk-then-`MergeImzml` path is the sharp edge — validate it here, while the reader is still the known-good one.

**Exit criteria:** differential harness green for both backends on uncompressed files; `pyimzml` gone from the dependency tree; `ReadSpectraParallel`/`WriteSpectraParallel` green against the imzy backend.

### Phase D — Upstream contributions to imzy (Days 8–10)

Three PRs to `vandeplaslab/imzy`, each independently useful, each removing one blocker. Open them as issues first, with the failing-case description.

1. **zlib decompression — required, do this one first.** Parse the compression `cvParam` per binary-array group (`MS:1000574` zlib / `MS:1000576` none), store the flag in the `.icache` (bump cache version so stale caches regenerate), decompress in `_read_spectrum`/`_read_spectra`. Reference: `imzml_reader.py:138-154`, `parse_spectra.py:107`, `compression.py`. **Reader half only** — neither toolchain writes compressed output, so `IMZMLWriter` support is out of scope and would only slow review. Worth reporting the silent-corruption behaviour as a bug in its own right, independent of the fix.
2. **Picklable readers** — `__getstate__`/`__setstate__` on `BaseReader` built on `_get_reader_kwargs()`, dropping open file handles and lazily reopening. Reference: `imzml_reader.py:56-84`.
3. **Configurable `.icache` location** — a `cache_dir` argument and/or `IMZY_CACHE_DIR` env, so read-only input trees work without silently falling back to a full re-parse on every open.

Assume upstream review is slow. **Do not block on merge:** carry each fix in the adapter (or a pinned fork) so Phase E can proceed, and leave a `# upstream: <PR url>` marker at every carried patch so a successor can delete them when the PRs land. If PR (1) turns out unnecessary per the Phase-A/4 data check, drop it and say so.

### Phase E — Flip the default, delete the custom parser (Days 11–13)

1. Make imzy the default backend. Run the full suite + `system_tests` + at least one real end-to-end `depiction_targeted_preproc` pipeline run.
2. **Delete** `depiction_io/imzml/parser/` (`parse_spectra.py`, `parse_metadata.py`, `cv_params.py` — 368 LOC), `imzml_reader.py`, `compression.py`, `imzml_alignment_tracker.py`, and their tests. Retarget the integration tests under `tests/integration/imzml_parser/` at the imzy backend rather than deleting them — they encode real cvParam edge cases (cf. commit `46f3f56 "handle weird cvParam entries"`) worth keeping as regression coverage.
3. **Keep** `ram/` (no imzy equivalent), `file_checksums.py`, `imzml_zip.py`, `pixel_size.py`, `types.py`.
4. *Optional, last, skippable:* `GenericReader.imzml_mode` / `ImzmlModeEnum` is imzML-specific naming that becomes nonsense once Bruker `.d` readers exist. Rename to `spectra_mode` / `SpectraMode` with a deprecated alias. Mechanical (33 sites) but pure churn — cut it if anything above slipped.

Net effect: roughly **-1,100 LOC of custom parsing**, `pyimzml` gone, Bruker `.d` support gained for free.

### Phase F — Archive hygiene (Day 14)

Under the dormant end state this is the part that actually matters. Do not let it get squeezed.

- Drop the `depiction.persistence` deprecation shim; single import path.
- **README:** it currently claims *"Python 3.12 is required, 3.13 is not compatible yet"* — contradicted by `requires-python >= 3.13`, `.python-version`, and CI. Fix, and document the workspace layout.
- **Sphinx docs:** `docs/modules/persistence/image_data.md` autodoc-references `depiction.persistence.format_ome_tiff.OmeTiff`, a path that has not existed for some time — autodoc would fail. Retarget all of `docs/modules/persistence/` at `depiction_io`; fill or delete the two empty stub headings (`### Format: RAM`, `### Format: NetCDF4`).
- Commit `uv.lock` (currently untracked, as is `pylock.toml`) so the environment is reproducible from a cold clone. **The single highest-value archive action.**
- Extend this document with what was actually done, what was deliberately not done, and which imzy patches are carried locally against which upstream PRs.
- Delete or mark stale the remaining branches (`extract-physical`, `imzml-zip-pipeline`, `tic-image`, `specify-dtype`, `better-bg-handling`, `tiff-orientation`, `update-deps`, `dev-calibration`, `optional-package`) — merge what's worth merging, delete the rest. An archived repo with 15 dangling branches is not archived.
- Enable the `system_tests` CI job (currently fully commented out with *"TODO add this later with synthetic or shared test data"*) using the Phase-A synthetic corpus.

---

## Critical files

**Config / structure**

- `pyproject.toml` → workspace root; drop `pyimzml`, `bioio*`, `tifffile`, `h5netcdf`
- `pkgs/depiction_io/pyproject.toml` → new
- `noxfile.py`, `.github/workflows/pr-checks.yml`, `.pre-commit-config.yaml`

**Moves (24 files)** — `src/depiction/persistence/{imzml,ram}/`, `types.py`, `file_checksums.py`, `imzml_zip.py`, `image/pixel_size.py` → `pkgs/depiction_io/src/depiction_io/`

**Moves (2 files, circular-dependency fix)** — `persistence/image/{ome_tiff,hdf5_image_format}.py` → `src/depiction/image/`. Rescued copies are in [`archive/`](archive/); do not re-derive them.

**New** — `pkgs/depiction_io/src/depiction_io/imzy_backend/`, `tests/differential/`

**Import rewrite (86 files, 134 statements)** — mechanical `depiction.persistence.X` → `depiction_io.X`, heaviest in `src/depiction/tools/` (27), `src/depiction_targeted_preproc/workflow/` (9), `tests/`

**Untested today, and now load-bearing** — `types.py`, `imzml_zip.py`, `compression.py`, `imzml_alignment_tracker.py`, `hdf5_image_format.py`, `pixel_size.py` have no unit tests. Add coverage for `types.py` and `compression.py` in Phase A; the rest can ride on the differential harness.

---

## Verification

Run at the end of every phase, not just at the end:

```bash
uv sync                                       # workspace resolves
nox                                           # lint + tests_depiction + tests_depiction_io + licensecheck
uv run pytest tests/differential -v           # both backends, byte-identical
uv run pytest tests/unit/parallel_ops -v      # pickling across process boundaries
nox -s system_tests                           # real data, slow
depiction-tools --help && depiction-tools imzml --help
```

Backend-parity check (Phase C onward):

```bash
DEPICTION_IO_BACKEND=legacy uv run pytest tests/ -q
DEPICTION_IO_BACKEND=imzy   uv run pytest tests/ -q   # identical results
```

Dependency isolation (Phase B onward):

```bash
uv pip show depiction    | grep -i 'pyimzml\|bioio'   # must be empty
uv pip show depiction_io | grep -i imzy               # must be present
```

End-to-end: one full `depiction_targeted_preproc` pipeline run on a real dataset, diffing the `.ome.tiff` and QC outputs against a pre-refactor baseline. This is the only check that covers the snakemake glue, and the one that would catch a writer regression.

---

## Risks

| Risk | Mitigation |
|---|---|
| Zlib-compressed input → imzy silently reads noise. Confirmed to occur here, if rarely | Three layers: hard guard raising `NotImplementedError` (Phase C, lands before any default flip); zlib fixture permanently in the differential corpus (Phase A/4); upstream PR #1 (Phase D, first PR). The guard is what makes the migration safe to abandon mid-flight |
| imzy PRs not merged in 2 weeks | Carry patches in-tree behind `# upstream: <url>` markers; never block a phase on review |
| `WriteSpectraParallel` chunk/merge breaks under the new writer | Flip the writer first (Phase C), while the reader is still known-good, so failures have one cause |
| Bruker readers unavailable on macOS | Expected — `imzy/plugins.py` disables them on Mac. Local dev is imzML-only; gate Bruker tests on platform |
| Two weeks runs out mid-migration, leaving attempt #4 | Every phase ends green and shippable. Phases A+B alone (4 days) already leave the repo better than today. Phase F is non-negotiable — cut Phase E's optional rename, not the archive hygiene |
