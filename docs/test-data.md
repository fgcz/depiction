# Public imzML test data

The two redistributable acquisitions this repository tests against. Both are downloaded by
[`tests/real_data/fetch.py`](../tests/real_data/fetch.py) into a gitignored cache, and every
measured number below is pinned in
[`tests/real_data/datasets.py`](../tests/real_data/datasets.py) — see
[How to fetch and check](#how-to-fetch-and-check).

They serve two different jobs, and no single file is good at both:

- **A real vendor acquisition.** Large, with the header conventions a SCiLS Lab export actually
  produces, which a synthetic corpus does not. That is Candidate 1.
- **A CI fixture.** Well under 100 MB, openly licensed, at a stable citable URL with a recorded
  checksum, so `system_tests` runs the whole `depiction_targeted_preproc` pipeline on every PR
  instead of skipping everywhere but one laptop. That is Candidate 2 — see
  [`system_tests/README.md`](../system_tests/README.md).

Both are other groups' published work, reused here under their stated licences. Anything
derived from them should carry the citations below. Every number in the tables was measured
locally through this repository's own reader, not quoted from the deposit.

## Candidate 1 — rapifleX acquisition

**[PRIDE PXD048809](https://www.ebi.ac.uk/pride/archive/projects/PXD048809)** — *Spatial
multi-omics in skeletal muscle unravel complex myofiber architecture*. Murine tibialis
anterior; metabolites and phospholipids. Published in Communications Biology:
[10.1038/s42003-024-06949-1](https://doi.org/10.1038/S42003-024-06949-1).

**Licence: CC0 (Creative Commons Public Domain)** — redistributable, including derivatives,
without a permission request. Citing it is courtesy rather than obligation under CC0, and we
should do it anyway.

```
https://ftp.pride.ebi.ac.uk/pride/data/archive/2024/09/PXD048809/
  20230130_rf1_msi2023001_slide001_ctrls.imzML
  20230130_rf1_msi2023001_slide001_ctrls.ibd
```

| | |
|---|---|
| Size | 8,903,764 B (imzML) + 1,169,811,216 B (ibd) = **1.18 GB** |
| SHA-256 imzML | `706a42243e95dca0d84f650840fd94d6a781c971857e96e244b10adb6ce03bb8` |
| SHA-256 ibd | `acbccad504442682c8e5426af718e3de8f9e613992a32ccf2d80265cd2f8fe02` |
| Mode | `continuous` + `profile spectrum`, no compression |
| Spectra | 3887 |
| m/z | 79.985 – 1000.008, 75200 bins, float64 m/z / float32 intensity |
| Geometry | declares 240 × 668 max pixels, 20 µm; coordinate bbox 240 × 668 |
| Exported by | SCiLS Lab 11.02.14724, Bruker Container nativeID format |

**Why this one.** It is structurally close to the FGCZ tonsil acquisition
(`system_tests/inputs/tonsil.imzML`): same SCiLS Lab export route, same Bruker Container
nativeID format, same continuous + profile + uncompressed combination, comparable raster
(20 µm) and comparable size (1.18 GB vs 1.26 GB). It is the only fixture here that exercises
those header conventions, and the only one that **declares a pixel size** — Candidate 2 does
not, so any assertion about physical size has to come from this file.

**On the rapifleX attribution.** The project declares two instruments, `solariX` and
`rapifleX`, and does not map files to instruments. This file is almost certainly the rapifleX:
its header names *"Bruker Daltonics flex series"* (a solariX acquisition surfaces as *"solarix
series"*), and the filename prefix `rf1` is consistent. That is inference from the header plus
the filename, not a statement from the depositors — worth confirming with them if it ever
becomes load-bearing.

**Low ion counts, which is the useful part.** Per-pixel maximum intensities across sampled
spectra range from 1 to 900 counts; spectrum 2500 peaks at 1 count. Sparse and near-empty
spectra are common here rather than exceptional. That is expected for MALDI-TOF at 20 µm in
this m/z range, and it is exactly what we want to test against: it exercises the peak-free path
added in commit `8a1f0db` ("drop peak-free spectra in `filter_peaks` rather than aborting") on
data that produces it naturally, instead of on a constructed edge case.

**Larger siblings, if a load test is ever wanted.** The same acquisition has three tissue
measurements in the same directory: `..._slide001_m1` (119.3 MB + 15.6 GB), `_m2` (110.0 MB +
14.4 GB), `_m3` (87.3 MB + 11.4 GB). `ctrls` is the small one.

## Candidate 2 — peptide TOF with documented calibrants

**[Zenodo 10.5281/zenodo.1560646](https://doi.org/10.5281/zenodo.1560646)** — *MALDI imaging of
mouse kidney peptides — test dataset*. FFPE mouse kidney, tryptic peptides. Prepared and
published as the training fixture for the
[Galaxy MSI tutorial](https://galaxyproject.github.io/training-material/topics/proteomics/tutorials/mass-spectrometry-imaging-loading-exploring-data/tutorial.html),
i.e. it was made to be reused for exactly this kind of purpose.

**Licence: MIT**, with a Zenodo DOI — a stable, citable URL.

```
https://zenodo.org/records/1560646/files/mouse_kidney_cut.imzML?download=1
https://zenodo.org/records/1560646/files/mouse_kidney_cut.ibd?download=1
```

| | |
|---|---|
| Size | 2,276,557 B (imzML) + 57,034,280 B (ibd) = **59 MB** |
| SHA-256 imzML | `779fa4cb718cc8b19a11c9e8ddeb90e3e7fef421852ccbf7e049a8f6d61aa1dc` |
| SHA-256 ibd | `8740034a9734a2713cce0de78b3ce03e49cba524e9614e14e9304ace45ce015b` |
| Mode | `continuous` + `profile spectrum` |
| Spectra | 1581 |
| m/z | 1220.0382 – 1624.9987, 9013 bins, **float32** m/z / float32 intensity |
| Geometry | coordinate bbox 31 × 51 (x 20–50, y 25–75); **the file declares no pixel size** |
| Instrument | Applied Biosystems/MDS SCIEX 4800 MALDI TOF/TOF, reflector positive |

**Why this one, beyond its size.** The deposit documents which calibrants were sprayed onto the
section — Angiotensin I, Substance P, [Glu1]-Fibrinopeptide B, ACTH 18-39, plus Bombesin
digestion-control spots — so there is an externally known set of masses to test against rather
than values recorded from one of our own runs. It is also the only fixture with **float32 m/z**;
the differential corpus parametrises over that, but no other real file covers it.

The `system_tests` panel is *not* built from those calibrants, for the reason in
[Measured mass offset](#measured-mass-offset--read-the-caveat): the peaks sit consistently below
their theoretical values, so a panel of theoretical masses would be testing that offset rather
than the pipeline. It uses the twenty strongest peaks of the fixture's own mean spectrum
instead — see `system_tests/panels/make_mouse_kidney_panel.py`. The calibrants remain the better
basis for a *calibration-accuracy* assertion, which is not what that test makes.

**The 150 µm raster is not in the file.** Its `scanSettings` carries only `max count of pixel x`
= 50 and `max count of pixel y` = 75 — no `IMS:1000046`/`IMS:1000047` anywhere. The 150 µm
figure is the deposit's description of the acquisition, not something the imzML states, and
conflating the two is easy to do.

This is a live constraint rather than trivia. Every spatial assertion in `system_tests` is
written in pixels because of it, and it is what surfaced a real defect: `Metadata.pixel_size`
was not optional, so `proc_export_raw_metadata` rejected the parsed metadata outright and
substituted a dummy 1 µm that reached the OME-TIFF. The tonsil never reached that branch, which
is why it went unnoticed until there was a file declaring no pixel size. Since fixed — the
field is `PixelSize | None` and the exporters omit a physical size they do not have.

### Measured mass offset — read the caveat

Peaks sit consistently *below* the theoretical monoisotopic (M+H)⁺ values. Mean spectrum over
159 pixels, apex bin and an intensity-weighted centroid over ±2 bins around it:

| Calibrant | Theoretical (M+H)⁺ | Apex | Centroid | Offset |
|---|---|---|---|---|
| Angiotensin I | 1296.6853 | 1296.6346 | 1296.6344 | −39 ppm |
| Substance P | 1347.7354 | 1347.7083 | 1347.7013 | −25 ppm |
| [Glu1]-Fibrinopeptide B | 1570.6768 | 1570.6172 | 1570.6181 | −37 ppm |
| Bombesin | 1619.8223 | 1619.8025 | 1619.7937 | −18 ppm |
| ACTH 18-39 | 2465.1989 | — | — | outside the trimmed m/z range |

**The caveat matters more than the numbers.** The profile bin spacing is 0.042–0.048 Da, which
is **30–33 ppm per bin** at these masses. So the per-peak magnitudes above are determined to
roughly one bin at best, and the differences between them are not resolved by this method. What
is robust is the *sign and rough scale*: a systematic negative offset of a few tens of ppm, in
the same direction for all four peaks.

**This is not a criticism of the deposit.** An offset of this size is unremarkable for a
reflector-mode TOF of this generation, the file is a deliberately trimmed teaching export
rather than a calibration reference, and part of the apparent offset may be our own crude peak
localisation on coarse profile bins. It is listed because a known, consistent, correctable
offset is *useful* to us — it gives calibration assertions an external reference — not because
anything is wrong with the data.

**Two real limitations for our purposes.** The instrument is a SCIEX 4800, not Bruker, and the
file was exported from Cardinal rather than SCiLS, so it will not exercise Bruker-specific
header handling. And it is pre-cropped (m/z 1220–1625, roughly half the kidney plus one control
spot), so it is a trimmed export rather than a full acquisition. Good fixture, weak "real
acquisition" evidence — hence Candidate 1.

## How to fetch and check

```bash
uv run python -m tests.real_data.fetch --list     # what is defined, and what is present
uv run python -m tests.real_data.fetch --all      # both, 1.24 GB
uv run pytest tests/real_data -v
```

The download goes to `.test-data/`, which is gitignored; `DEPICTION_TEST_DATA_DIR` points it
somewhere else, and pointing it at an empty directory is how the tests are made to skip — which
is what CI and a fresh clone do. Nothing is written under its final name until its SHA-256
matches, so a partial download cannot be mistaken for a complete file.

**The pinned values came from two independent readers**, measured while both the imzy backend
and the hand-rolled parser still existed, and they agreed on all of them. That is the point of
recording them: a future disagreement is evidence about imzy, not about how the expectations
were derived. `tests/real_data/` asserts them rather than quoting them, and adds what a
one-off comparison could not — the declared `IMS:1000091` checksum validating (the only
real-file exercise `parse_metadata.py` has), `scan_imzml` on genuine vendor headers, the
z-column decision checked against the XML, batched reads agreeing with per-spectrum reads, and
a `ReadSpectraParallel` round trip through worker processes. 38 tests, 8 s once the files are
local.

## Other sources considered

None of the following are deficient datasets; they are simply not what is needed here, which is
the narrow and unusual combination of *uncompressed, continuous, profile-mode* imzML under a
licence that permits redistribution.

| Source | Why it is not used here |
|---|---|
| [METASPACE](https://metaspace2020.eu) | A large and valuable resource — 19,772 datasets — but oriented to annotation rather than redistribution. Its ingest expects centroided data, so profile-mode files are rare; of the datasets surveyed only 7 are tagged `TOF (Rapiflex)`. More decisively, the download metadata exposes no licence field, so redistribution rights would have to be established per dataset with each submitter, and anonymous bulk downloads are rate-limited (a reasonable protection on a free service, but incompatible with a CI fixture). Better used interactively than as a pinned test input. |
| [PRIDE PXD049325](https://www.ebi.ac.uk/pride/archive/projects/PXD049325) | CC0, rapifleX + timsTOF fleX, amyloid plaque analysis. The peptide files are centroided and cover m/z 1999–16051 in linear mode — a different acquisition regime from the one this pipeline processes. |
| [PRIDE PXD047820](https://www.ebi.ac.uk/pride/archive/projects/PXD047820) | CC0, rapifleX + Orbitrap Exploris 480, head and neck. `Sample3_cal33_6_1_0_15.imzML` is centroided. |
| [GigaDB 100909](https://doi.org/10.5524/100909) (M²aia multi-modal mouse brain), [100131](https://doi.org/10.5524/100131) (Oetjen et al. 3D benchmark) | **Could not be assessed from this environment**, so they are neither endorsed nor excluded. Both are plausible on paper, and M²aia's example data is rapifleX. We could not retrieve a file listing: the dataset pages render client-side, and the mirrors we tried did not respond usefully to scripted access from this network. The Oetjen files are additionally named `-centroid.imzML`. Worth one more attempt via a working mirror, or simply by asking, before ruling out. |

Centroided data dominates public MSI deposits because that is what most downstream annotation
workflows consume — a sensible choice by depositors, and the reason only two candidates fit the
profile-mode requirement.

## What is still not covered

Neither candidate closes these, and finding more data will not close them either:

- **Zlib-compressed imzML.** Still no *specimen* — neither candidate is compressed, and nothing
  in either toolchain can *produce* one, so a real-world example likely has to be hand-built
  rather than found. The *code* gap is closed: `imzy_backend/zlib_reader.py` handles it and
  `corpus.compress_case` builds synthetic zlib twins. What is missing is a compressed file
  somebody else's software wrote, which is the only thing that would test the assumption that
  they compress each binary array whole.
- **`processed` mode.** Both candidates are `continuous`. Two of the three imzML variants
  remain fixture-free on real data.
- **Bruker `.d` (TSF/TDF/NeoFlex).** Reachable through `get_read_file`, exercised by nothing,
  cannot run on macOS. Unaffected by anything here.

Everything cited here was checked on 2026-08-07. Licence fields and URLs come from the PRIDE
and Zenodo APIs; sizes, checksums, geometry and spectral values were measured locally on the
downloaded files and are now asserted by `tests/real_data/datasets.py`.
