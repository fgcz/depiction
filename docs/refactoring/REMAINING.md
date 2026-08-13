# What is still open

The refactoring this directory tracked is finished: the two-package split landed, imzML I/O
went to `imzy` and the hand-rolled parser is gone, the system tests run on a public fixture in
CI, and the pre-refactor baseline diff came back identical on two real acquisitions. That
history is in the git log; the design rationale that outlived it is in
[`../modules/depiction_io/imzy_backend.md`](../modules/depiction_io/imzy_backend.md) and
[`../test-data.md`](../test-data.md).

What is left is below. Nothing here blocks anything else, and the repository is coherent and
green without it — this is a list for whoever picks it up, not a plan in flight.

## 1. Report the five imzy gaps upstream

Not started. All five are described, with their failing cases, in
[`../modules/depiction_io/imzy_backend.md`](../modules/depiction_io/imzy_backend.md), and all
five are worked around in-tree behind `# upstream:` markers. Open them as issues first, with
the failing case. Gap (1)'s silent-corruption behaviour and gap (3)'s missing cache provenance
are the two worth reporting even if no patch follows.

Assume upstream review is slow. **Do not block on merge:** carry each fix in the adapter or a
pinned fork, and leave a `# upstream: <PR url>` marker at every carried patch.

## 2. Known, still unfixed

- **`WriteSpectraParallel` does not propagate dtypes.** Its chunk files are opened with
  `ImzmlWriteFile`'s defaults, so a float64 intensity array comes back as float32 after a
  parallel round trip. Pre-existing, and pinned by `tests/differential/test_writer.py` rather
  than fixed.
- **Some pipeline steps use a lot of RAM.** Two places account for the peak: the sparse-aware
  spatial smoothing (`image/smoothing/spatial_smoothing_sparse_aware.py`) works one channel at
  a time but needs that channel's whole image resident while it does — `scipy.signal.convolve`
  over a float64 copy, twice, with no offloading — and parsing the input file's metadata.
  **Never profiled:** this is the reported shape of the problem, not a measurement, and anyone
  who wants to fix it should start by measuring rather than by trusting this entry. Filed as
  #32, closed on the way into dormancy because it was a diagnosis request with no acceptance
  criterion, not because the RAM use went away.
- **`get_spectrum_n_points` has no in-repo consumer.** It is part of the `GenericReader`
  protocol and both backends implement it, and `tests/real_data/` checks its answer against the
  array it describes. Read it as protocol surface, not as a method whose behaviour anything
  depends on. Its one former caller reported four times the truth and was covered by no test.

## 3. Fixtures that do not exist

See [`../test-data.md`](../test-data.md) for the full reasoning. In short: no third-party
zlib-compressed specimen, no `processed`-mode real acquisition, and the Bruker `.d` and Waters
`.raw` readers cannot run on macOS and have no fixture. None of these is closed by finding
more of the same kind of data.

## 4. The A355 app-interface move

`depiction` no longer knows what a workunit is — the app-runner glue for
A355_MSI_Targeted_PreprocBatch left the library, because it belongs with the app.

**It has not landed anywhere yet.** The code is preserved at
[`fgcz/depiction@043390f`](https://github.com/fgcz/depiction/tree/043390f99d39cb7a9e850876da5ca1ab93ca7741/src/depiction_targeted_preproc/app_interface),
and the file-by-file map of what moves where is tracked as an issue in the internal
slurmworker repository. Production A355 is unaffected — its pinned `0.1.12` installs a wheel
built before the removal — but the `devel` version of `app_A355.yml` points at a live checkout
and **breaks until that move is done.**

`app_interface/process_chunk.py` stayed: it imports no `bfabric` module, it is the snakemake
entry point, and the system tests drive the pipeline through it.

## 5. Optional: a second large fixture that anyone can run

`system_tests` has two fixtures, and only the 59 MB public one runs in CI. The other is the
1.26 GB FGCZ tonsil, which is not redistributable and neither is its 118-marker PC-MT panel, so
it skips on any machine without the files.

`RAPIFLEX_CTRLS` (`tests/real_data/datasets.py`) would close half of that gap without any new
download infrastructure: CC0, already SHA-256-pinned, 1.17 GB, the same SCiLS Lab / Bruker
Container header conventions as the tonsil, and — unlike the mouse kidney — it **declares a
pixel size**, which is the one assertion the CI fixture cannot exercise. It would need a panel
derived from its own mean spectrum, the way
`system_tests/panels/make_mouse_kidney_panel.py` already does. At 1.17 GB it still could not
run in CI.

The tonsil itself is worth revisiting once the acquisition's paper publishes; until then
neither the data nor the panel is ours to publish.

## 6. Available, not taken

- **`GenericReader.imzml_mode` / `ImzmlModeEnum` → `spectra_mode` / `SpectraMode`.** The
  current naming is imzML-specific and becomes nonsense now that Bruker `.d` readers are
  reachable. Mechanical across 33 sites, and pure churn.
- **Splitting `MultiChannelImage` into masked and unmasked variants**, and renaming
  `AlphaChannel` → `MaskChannel`. Planned in March 2025, **never started**, and deliberately
  kept out of scope for the work that followed — it is an image-layer change, and the
  migration was an I/O-layer one. The plan is in git history at
  [`a46bdd6:docs/refactoring/archive/2025-03-multi-channel-image-plan.md`](https://github.com/fgcz/depiction/blob/a46bdd67f98d0306a2a99db7281c0273d1cc790f/docs/refactoring/archive/2025-03-multi-channel-image-plan.md),
  and the only other trace is the tag `archive/better-bg-handling`. Anyone reviving it should
  read the dimension and geometry conventions in
  [`../modules/image/multi_channel_image.md`](../modules/image/multi_channel_image.md) first.

## 7. Deliberate non-goals

These are decisions, not oversights. A successor should not "fix" them without a reason that
did not apply before:

- **`ANN` and `D103` are not enforced by ruff.** ~210 pre-existing violations (missing
  annotations, missing docstrings on public functions). What is enforced can actually be kept
  green, which is the point.
- **`TC` is not enforced.** Moving an import into a `TYPE_CHECKING` block breaks anything that
  resolves annotations at runtime, and this project does that in two places at once — pydantic
  models and cyclopts CLIs. 56 violations, all in that blast radius.
- **`UP042` (`class Foo(str, Enum)` → `StrEnum`) is ignored.** Not cosmetic: the two differ in
  `str()` and f-string interpolation (`"Foo.X"` vs `"x"`). All six occurrences are config enums
  flowing through pydantic models and snakemake YAML, so converting them is a serialization
  change needing its own commit and tests.
- **`system_tests` cannot run in CI in full** — one of its two fixtures is 1.26 GB and not
  redistributable. The public one does run, on every PR.
- **No `depiction.persistence` deprecation shim.** It would have lived about two weeks before
  the archive-hygiene pass deleted it again, leaving a successor with two import paths and no
  reason for either.

## Environment gotchas

- **`uv sync --all-extras` does not work on macOS arm64**, and did not before the split:
  `ms-peak-picker` has no wheel and its sdist build fails against the current numpy. Use
  `uv sync --extra dev`.
- **`system_tests/baseline/constraints.txt` has a hand-maintained line.** The pre-refactor tree
  genuinely needs upstream `snakemake_invoke`, but here it is a workspace member, so re-running
  the documented `uv export --no-emit-workspace` recipe drops the pin without failing. Both the
  file's header and [`../../system_tests/baseline/README.md`](../../system_tests/baseline/README.md)
  say so.
- **`findmfpy` is the `findmf` extra**, not a hard dependency: a C++ extension with cp313-only
  wheels, so requiring it required a compiler everywhere else. `get_peak_picker` turns the
  missing import into a message naming the extra. The `testing` extra still pulls it in, so CI
  keeps covering that branch.
