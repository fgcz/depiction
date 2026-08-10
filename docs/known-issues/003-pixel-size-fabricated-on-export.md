# Pixel size is dropped on write, then fabricated as 1 µm on export

Severity: **high** | Status: open | Found: 2026-08-10
Files: `pkgs/depiction_io/src/depiction_io/imzml/metadata.py:7`,
`pkgs/depiction_io/src/depiction_io/imzml/imzml_writer.py:34`,
`src/depiction_targeted_preproc/workflow/proc/export_raw_metadata.py:19-29`

## Symptom

Exported OME-TIFF and OME-NGFF images declare a physical pixel size of **1 µm × 1 µm**
regardless of the acquisition's real raster. Any downstream measurement in physical units —
distances, areas, scale bars, registration against another modality — is wrong by whatever the
real pixel pitch was (typically 20–100 µm, i.e. off by one to two orders of magnitude).

## Why it happens

Three pieces compound:

1. `Metadata.pixel_size` is a **non-optional** field, so `ParseMetadata.parse()` raises on any
   file that declares no `IMS:1000046` / `IMS:1000047`.
2. `ImzmlWriter.open` emits no pixel-size cvParams, so **every file this package writes** is
   such a file. A read → write → read round trip loses the pixel size.
3. `export_raw_metadata.py` catches the resulting failure and substitutes
   `PixelSize(1, 1, "micrometer")`, which `vis/images_ome_tiff.py:20` and
   `vis/images_ome_ngff.py:20` then write as the image's **physical** pixel size.

The invented value is indistinguishable from a genuine 1 µm acquisition in the output.

## How to reproduce

Write any file with `ImzmlWriter` and read its metadata back; `ParseMetadata.parse()` raises
on the missing pixel size. Then run the pipeline to `CALIB_IMAGES` on an acquisition with a
known raster and inspect the OME-TIFF's `PhysicalSizeX` / `PhysicalSizeY`.

## Fix sketch

Minimum honest fix — stop inventing a number:

- make it `pixel_size: PixelSize | None = None` on `Metadata`,
- let `None` propagate through `OmeTiff.write_image` so the exporter simply omits the physical
  size rather than asserting a false one,
- unpin `system_tests/calibration/test_pipeline_calibration_only.py:161`, which currently
  asserts the fabricated 1 µm and so locks the bug in place.

The larger, optional half is forwarding a real pixel size through `ImzmlWriteFile` /
`copy_spectra` so it survives processing. If you only do one, do the first: an absent pixel
size is recoverable, a confidently wrong one is not.

## Notes

Pre-existing, **not** a regression from the imzy migration — the previous pyimzml-based writer
dropped it too (`git show a46bdd6:pkgs/depiction_io/.../imzml_writer.py`).

Already written up under "Known, still unfixed" at `docs/refactoring/ROADMAP.md:497-506`. This
file exists because that entry undersells it: the ROADMAP frames it as a metadata gap, but the
consequence is wrong physical units in the exported images.
