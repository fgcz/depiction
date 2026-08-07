# The pre-refactor baseline diff

Run 2026-08-07 by Leonardo Schwarz. Reproduce with
[`system_tests/baseline/`](../../system_tests/baseline/README.md).

**Result: the migration changed nothing in the data.** The `depiction_targeted_preproc`
`CALIB_IMAGES` chain, run on two real acquisitions in this tree and in the tree as it was
before the `imzy` migration, produces identical output. Every spectrum's m/z and intensity
array, every coordinate, every image value, every channel name, the calibration
coefficients, the OME-TIFF and the SpatialData zarr — all equal, exactly, no tolerance.

The `.ibd` files go further than that: they are **byte-identical after their 16-byte UUID
header**. `cmp -l` reports exactly 16 differing bytes in a 3.8 GB file. pyimzml and imzy
wrote the same bytes.

This closes the risk the [roadmap](ROADMAP.md) recorded as *"open, and now the largest
remaining one by some distance"*.

## Why this was the last open question

Everything else built during the migration checks the new code against itself or against
ground truth: the differential corpus, `tests/real_data/`, the Phase G system tests. All of
them would have stayed green if imzy simply produced *different* output from pyimzml, as long
as it produced it consistently. Nothing compared against the code that was replaced.

## Method

The baseline is **`ed222b3`** ("Spatial dist plot", #31) — the last commit before Phase A,
still on `pyimzml`, still with `depiction.persistence` inside the parent package.

**There is one variable.** `git diff ed222b3 HEAD` over the entire chain —
`workflow/Snakefile`, `rules/rules_proc.smk`, `rules/rules_vis.smk`,
`pipeline_config/artifacts_mapping.py`, `tools/process_spectra/`, `tools/calibrate/`,
`tools/generate_ion_image.py`, `tools/cli/cli_generate_ion_images.py`,
`image/multi_channel_image.py`, `parallel_ops/write_spectra_parallel.py` — is imports, type
annotations and comment rewrapping. The rule files and `artifacts_mapping.py` are
byte-identical; `image/ome_tiff.py` differs from its baseline counterpart by one import line.
So the pipeline either side of the I/O layer is the same code, and any difference in the
output could only have come from the reader or the writer.

**The environments differ by one package.** `ed222b3` predates the committed `uv.lock`, so
installing it today would resolve ~380 packages at whatever is current and an imzy difference
could not be told apart from a scipy bump. Instead the baseline was installed against this
tree's lock exported as a constraints file
([`system_tests/baseline/constraints.txt`](../../system_tests/baseline/constraints.txt)):

| | baseline | current |
|---|---|---|
| Python | 3.13.7 | 3.13.7 |
| shared packages | 198, **0 version mismatches** | |
| unique to baseline | `pyimzml` 1.5.5, `wheezy-template` (its dependency) | — |
| unique to current | — | `imzy` 0.3.0 and its subtree, plus the dev/testing extras |

`snakemake-invoke` was pinned to `e7e3c33` in both. The baseline's own `pyproject.toml`
declares it with no revision, so installed as written it resolves the git HEAD, where
`SnakemakeInvoke` has been moved out of `__init__.py` — the bug Phase G found on `dev`, which
would otherwise have made the baseline tree unrunnable today.

**Both fixtures.** `mouse_kidney` (public, 1581 spectra, a 20-mass generated panel) and
`tonsil` (FGCZ, 10131 spectra, the real 118-marker panel). Work directories staged by the same
code from `system_tests/calibration/configs/`; only the interpreter differs.

**Comparison is by value, and the imzML pairs are read under both readers.** A single reader
cannot distinguish "the two files agree" from "the reader makes the same mistake on both" —
which is exactly the check Phase E recorded losing when the second XML parser was deleted. So
`compare_imzml.py` runs once under imzy and once under the pre-refactor parser. Both report
identical.

**Exact equality was the assertion, decided before any number was seen.** No tolerance was
used, and none was needed.

## Results

Every cell below is "identical", for both fixtures, with no tolerance.

| Artifact | Compared | mouse_kidney | tonsil |
|---|---|---|---|
| `panels/{unstandardized_full,full,full_visualize,calibration}.csv` | text | ✅ | ✅ |
| `raw_metadata.json`, `config/*.yml` | text | ✅ | ✅ |
| `processed.imzML` / `.ibd` | all m/z + intensity arrays, dtypes, coordinates, mode | ✅ 1581 spectra | ✅ 10131 spectra |
| `calibrated.imzML` / `.ibd` | the same | ✅ | ✅ |
| `images_default.hdf5` | values, channel names, foreground mask | ✅ | ✅ |
| `calib_data.hdf5` — `features_raw`, `features_processed`, `model_coefs` | the same | ✅ | ✅ |
| `images_default.ome.tiff` | the same, plus physical pixel size | ✅ | ✅ |
| `images_default.sd.zarr` | image array, channel coordinates | ✅ | ✅ |

`processed.imzML` is the tightest of these: `process_spectra` runs with `steps: []`, so it is
a read→write round trip on a real acquisition (which also rewrites the continuous input as
processed). `calibrated.imzML` is the same after real computation.

**The comparators were mutation-checked before the result was believed.** A copy of one work
directory was perturbed in six places — one text value, one image pixel, one channel name,
the pixel size, one byte of the `calibrated.ibd`, one calibration coefficient — and the
downstream artifacts regenerated with the pipeline's own code. All ten resulting problems were
reported, each naming its artifact, and both scripts exited non-zero; the two untouched
`calib_data` groups were correctly reported as unchanged. The single flipped `.ibd` byte
surfaced as `spectrum 1580 int: 1/9013 values differ, max abs diff 1.4013e-45`.

## The differences that do exist

None of these are differences in the data. They are listed because "identical" above means
*by value*, and a successor comparing files directly will see them.

**In the `.ibd`:** the 16-byte UUID header, and nothing else.

**In the imzML XML**, beyond attribute quoting, self-closing-tag whitespace and omitted
`value=""`:

- `IMS:1000080` (UUID) and `IMS:1000091` (ibd SHA-1) — different by construction.
- `<software>` / `<dataProcessing>` identify the writer: pyimzml (whose tag says 1.5.4 even
  though the installed package is 1.5.5) versus imzy 0.3.0, and imzy adds a
  `userParam name="imzy export timestamp"`. **That timestamp is the one thing preventing a
  byte-reproducible imzML**, which is worth knowing for anyone who wants one.
- The per-spectrum summary cvParams are written at lower precision by imzy —
  `1220.0382205078124` becomes `1220.03822051`, `198.0` becomes `198`. These are the
  descriptive `lowest observed m/z` / `base peak` / `total ion current` values, not the data;
  the arrays they describe are byte-identical.
- `unitName` for intensity is `number of detector counts` under imzy and `number of counts`
  under pyimzml. imzy's is the MS ontology's name.

Three differences are imzy being **more correct** than what it replaced:

- pyimzml wrote `cvList count="2"` and used `cvRef="IMS"` throughout without ever declaring
  the IMS ontology. imzy declares it.
- pyimzml wrote `instrumentConfigurationRef="instrumentConfiguration0"` on every spectrum
  while the only `<instrumentConfiguration>` it declared had `id="IC1"` — a dangling reference
  in every imzML this pipeline has ever produced. imzy writes `IC1`.
- imzy adds `IMS:1000044` / `IMS:1000045` (max dimension x/y), which pyimzml omitted. Note
  that it writes the *pixel count* into them rather than a physical dimension, which is
  dimensionally wrong; nothing in this codebase reads them, so it is inert here.

**Sidecars:** the current tree leaves `raw.icache` / `processed.icache` / `calibrated.icache`
next to the imzML files. These are imzy's parse caches, not pipeline artifacts, and they are
not in `get_result_files_new`.

## What this does not cover

The diff is one path through the pipeline, run twice:

- **One artifact**, `CALIB_IMAGES`, and therefore one branch of the rule graph. The QC plots,
  the clustering rules and `RAW_TIC` were not exercised.
- **One calibration method**, `ConstantGlobalShift`.
- **`process_spectra` with `steps: []`** — no peak picking, no baseline correction, no
  smoothing. Those tools use the same reader and writer, but through code paths this run did
  not take.
- **imzML only.** The Bruker `.d` and Waters `.raw` readers imzy adds cannot run on macOS and
  have no fixture; see the roadmap.
- **No compressed input.** Both fixtures are uncompressed. The zlib path has the differential
  corpus and `tests/integration/imzml_parser/`, not this.
- **Two acquisitions**, both continuous, both 2D, both from a single vendor lineage.

It also does not say the *current* code is correct — only that it agrees with the code it
replaced, which is a different and narrower claim.
