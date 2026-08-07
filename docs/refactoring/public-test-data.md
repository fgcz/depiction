# Public imzML test data: verified candidates

Status: **candidates identified and verified**, 2026-08-07. Author: Leonardo Schwarz.
No repository code has been changed — this is the input-gathering step, done ahead of the
work that consumes it.

## Why this exists

Two phases of [`ROADMAP.md`](ROADMAP.md) are waiting on redistributable data, for different
reasons:

- **Phase E** ("flip the default, delete the custom parser") is *"blocked on inputs only the
  maintainer can provide: a real dataset plus pre-refactor baseline outputs"*. The imzy
  reader is not the default because it *"has not been checked against a real acquisition"*.
  What Phase E needs is a **real acquisition**, and it may be large.
- **Phase G** ("a downloadable public fixture for the system tests") needs the opposite: a
  pair **well under 100 MB**, openly licensed, at a stable citable URL, with a recorded
  checksum, because `system_tests/` currently has exactly one fixture — a 1.26 GB
  FGCZ-local file — so the suite skips everywhere except one laptop.

No single file is good at both jobs. The two below split them. Both were downloaded and read
through this repository's own readers; the numbers in the tables are measured locally, not
quoted from the deposit.

Both are other groups' published work, reused here under their stated licences. Anything
derived from them should carry the citations given below.

## Candidate 1 — rapifleX acquisition, for Phase E

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

**Why this one.** It is structurally close to `system_tests/inputs/tonsil.imzML`: same SCiLS
Lab export route, same Bruker Container nativeID format, same continuous + profile +
uncompressed combination, comparable raster (20 µm) and comparable size (1.18 GB vs
1.26 GB). That is the property Phase E needs — it exercises the header conventions a
vendor export actually produces, which a synthetic corpus does not.

**On the rapifleX attribution.** The project declares two instruments, `solariX` and
`rapifleX`, and does not map files to instruments. This file is almost certainly the
rapifleX: its header names *"Bruker Daltonics flex series"* (a solariX acquisition surfaces
as *"solarix series"*), and the filename prefix `rf1` is consistent. That is inference from
the header plus the filename, not a statement from the depositors — worth confirming with
them if it ever becomes load-bearing.

**Low ion counts, which is the useful part.** Per-pixel maximum intensities across sampled
spectra range from 1 to 900 counts; spectrum 2500 peaks at 1 count. Sparse and near-empty
spectra are common here rather than exceptional. That is expected for MALDI-TOF at 20 µm in
this m/z range, and it is exactly what we want to test against: it exercises the peak-free
path added in commit `8a1f0db` ("drop peak-free spectra in `filter_peaks` rather than
aborting") on data that produces it naturally, instead of on a constructed edge case.

**Larger siblings, if a load test is ever wanted.** The same acquisition has three tissue
measurements in the same directory: `..._slide001_m1` (119.3 MB + 15.6 GB),
`_m2` (110.0 MB + 14.4 GB), `_m3` (87.3 MB + 11.4 GB). `ctrls` is the small one.

## Candidate 2 — peptide TOF with documented calibrants, for Phase G

**[Zenodo 10.5281/zenodo.1560646](https://doi.org/10.5281/zenodo.1560646)** — *MALDI imaging
of mouse kidney peptides — test dataset*. FFPE mouse kidney, tryptic peptides. Prepared and
published as the training fixture for the
[Galaxy MSI tutorial](https://galaxyproject.github.io/training-material/topics/proteomics/tutorials/mass-spectrometry-imaging-loading-exploring-data/tutorial.html),
i.e. it was made to be reused for exactly this kind of purpose.

**Licence: MIT**, with a Zenodo DOI — a stable, citable URL, which is what Phase G asks for.

```
https://zenodo.org/records/1560646/files/mouse_kidney_cut.imzML?download=1
https://zenodo.org/records/1560646/files/mouse_kidney_cut.ibd?download=1
```

| | |
|---|---|
| Size | 2,276,557 B (imzML) + 57,034,280 B (ibd) = **59 MB — within Phase G's <100 MB target** |
| SHA-256 imzML | `779fa4cb718cc8b19a11c9e8ddeb90e3e7fef421852ccbf7e049a8f6d61aa1dc` |
| SHA-256 ibd | `8740034a9734a2713cce0de78b3ce03e49cba524e9614e14e9304ace45ce015b` |
| Mode | `continuous` + `profile spectrum` |
| Spectra | 1581 |
| m/z | 1220.0382 – 1624.9987, 9013 bins, **float32** m/z / float32 intensity |
| Geometry | coordinate bbox 31 × 51 (x 20–50, y 25–75), 150 µm raster |
| Instrument | Applied Biosystems/MDS SCIEX 4800 MALDI TOF/TOF, reflector positive |

**Why this one, beyond its size.** The deposit documents which calibrants were sprayed onto
the section — Angiotensin I, Substance P, [Glu1]-Fibrinopeptide B, ACTH 18-39, plus Bombesin
digestion-control spots — so there is an externally known set of masses to test against
rather than values recorded from one of our own runs.

It is also **float32 m/z**. The differential corpus parametrises over that, but no real
fixture currently covers it.

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

**The caveat matters more than the numbers.** The profile bin spacing is 0.042–0.048 Da,
which is **30–33 ppm per bin** at these masses. So the per-peak magnitudes above are
determined to roughly one bin at best, and the differences between them are not resolved by
this method. What is robust is the *sign and rough scale*: a systematic negative offset of a
few tens of ppm, in the same direction for all four peaks.

**This is not a criticism of the deposit.** An offset of this size is unremarkable for a
reflector-mode TOF of this generation, the file is a deliberately trimmed teaching export
rather than a calibration reference, and part of the apparent offset may be our own crude
peak localisation on coarse profile bins. It is listed because a known, consistent,
correctable offset is *useful* to us — it gives calibration assertions an external
reference — not because anything is wrong with the data.

**Two real limitations for our purposes.** The instrument is a SCIEX 4800, not Bruker, and
the file was exported from Cardinal rather than SCiLS, so it will not exercise
Bruker-specific header handling. And it is pre-cropped (m/z 1220–1625, roughly half the
kidney plus one control spot), so it is a trimmed export rather than a full acquisition.
Good fixture, weak "real acquisition" evidence — hence Candidate 1.

## Verification performed

Both pairs were downloaded and read through both backends. `tests/differential/` was not
modified; this was run directly against `ImzmlReadFile` and `ImzyReadFile`:

```
mouse_kidney_cut   mode CONTINUOUS | CONTINUOUS   n_spectra 1581 | 1581
  spectra 0, 5, 790, 1580        → m/z and intensity arrays byte-equal, coordinates equal
rf1_ctrls          mode CONTINUOUS | CONTINUOUS   n_spectra 3887 | 3887
  spectra 0, 1, 17, 1000, 2500, 3886 → m/z and intensity arrays byte-equal, coordinates equal
```

**This is the first time the imzy reader has been checked against a real third-party
acquisition, which is the stated reason it is not the default.** It agrees exactly on both.
That is evidence for Phase E, not a completion of it: Phase E also requires routing the 46
`ImzmlReadFile(...)` call sites through `get_read_file`, and an end-to-end
`depiction_targeted_preproc` run diffed against a baseline. Reader parity on two files does
not substitute for either.

## Other sources considered

None of the following are deficient datasets; they are simply not what these two phases
need, which is the narrow and unusual combination of *uncompressed, continuous, profile-mode*
imzML under a licence that permits redistribution.

| Source | Why it is not used here |
|---|---|
| [METASPACE](https://metaspace2020.eu) | A large and valuable resource — 19,772 datasets — but oriented to annotation rather than redistribution. Its ingest expects centroided data, so profile-mode files are rare; of the datasets surveyed only 7 are tagged `TOF (Rapiflex)`. More decisively for us, the download metadata exposes no licence field, so redistribution rights would have to be established per dataset with each submitter, and anonymous bulk downloads are rate-limited (a reasonable protection on a free service, but incompatible with a CI fixture). Better used interactively than as a pinned test input. |
| [PRIDE PXD049325](https://www.ebi.ac.uk/pride/archive/projects/PXD049325) | CC0, rapifleX + timsTOF fleX, amyloid plaque analysis. The peptide files are centroided and cover m/z 1999–16051 in linear mode — a different acquisition regime from the one this pipeline processes. |
| [PRIDE PXD047820](https://www.ebi.ac.uk/pride/archive/projects/PXD047820) | CC0, rapifleX + Orbitrap Exploris 480, head and neck. `Sample3_cal33_6_1_0_15.imzML` is centroided. |
| [GigaDB 100909](https://doi.org/10.5524/100909) (M²aia multi-modal mouse brain), [100131](https://doi.org/10.5524/100131) (Oetjen et al. 3D benchmark) | **Could not be assessed from this environment**, so they are neither endorsed nor excluded. Both are plausible on paper, and M²aia's example data is rapifleX. We could not retrieve a file listing here: the dataset pages render client-side, and the mirrors we tried did not respond usefully to scripted access from this network. The Oetjen files are additionally named `-centroid.imzML`. Worth one more attempt via a working mirror, or simply by asking, before ruling out. |

Centroided data dominates public MSI deposits because that is what most downstream
annotation workflows consume — a sensible choice by depositors, and the reason only two
candidates fit our profile-mode requirement.

## What is still not covered

Neither candidate closes these, and finding more data will not close them either:

- **Zlib-compressed imzML.** Still no *specimen* — neither candidate is compressed, and per
  Phase A nothing in either toolchain can *produce* one, so a real-world example likely has
  to be hand-built rather than found. Note that the reader-side half of Phase D gap (1) was
  in flight as uncommitted work (`imzy_backend/zlib_reader.py`) when this was written, so
  check its state before assuming compressed input is unsupported. Data and code are
  separate gaps; only the data one is this document's subject.
- **`processed` mode.** Both candidates are `continuous`. Two of the three imzML variants
  remain fixture-free on real data.
- **Bruker `.d` (TSF/TDF/NeoFlex).** Plumbed in Phase C, exercised by nothing, cannot run on
  macOS. Unaffected by anything here.

## Suggested next steps

Ordered by what unblocks the most for the least work:

1. **Take Candidate 2 as the Phase G fixture.** It meets every criterion Phase G lists —
   under 100 MB, MIT, DOI-stable, checksum recorded above. The download fixture is the easy
   half; per Phase G step 3 the real work is rewriting the `128 x 137` / 118-channel /
   10131-non-zero assertions so they derive from the input rather than from one recorded
   output.
   Note the fixture has **no PC-MT panel**, so `panel.csv` needs a companion target list —
   the calibrants above are the obvious basis for one. If a calibration assertion is built
   on them, give it a tolerance consistent with the ~30 ppm bin spacing rather than the
   point estimates in the table.
2. **Take Candidate 1 as the Phase E acquisition**, keeping the tonsil path working alongside
   both, per Phase G's closing note. At 1.18 GB it is too large for CI, so it belongs in the
   same on-demand cache, marked slow and kept off the fast path.
3. **Do not read the parity result above as more than it is.** It covers the reader on two
   files. The writer, the 46 call sites, and the end-to-end baseline diff are all still open.

Everything cited here was checked on 2026-08-07. Licence fields and URLs come from the PRIDE
and Zenodo APIs; sizes, checksums, geometry and spectral values were measured locally on the
downloaded files.
