# depiction_io

Reading and writing mass-spectrometry imaging data, split out of `depiction` so that the
I/O layer has a boundary of its own.

## What is here

- `types.py` — the `GenericReader` / `GenericReadFile` / `GenericWriter` /
  `GenericWriteFile` protocols. This is the seam: everything else in `depiction` talks to
  these rather than to a file format, so a backend can be swapped underneath without the
  callers noticing.
- `imzml/` — the imzML reader (memory-mapped, picklable), the writer, and the XML/cvParam
  parser.
- `ram/` — in-memory implementations of the same protocols, used heavily in tests.
- `imzml_zip.py`, `file_checksums.py`, `pixel_size.py` — supporting pieces.

## What is deliberately not here

OME-TIFF and HDF5 image formats live in `depiction.image`, not in this package. They take
and return `MultiChannelImage`, so putting them here would make `depiction_io` depend on
`depiction` and defeat the split. That is also why `bioio`, `bioio-ome-tiff` and
`tifffile` are not dependencies of this package.

## Status

See [`docs/refactoring/ROADMAP.md`](../../docs/refactoring/ROADMAP.md) in the repository
root. The medium-term intent is for the imzML reading and writing here to be replaced by
[`imzy`](https://github.com/vandeplaslab/imzy) behind the protocols in `types.py`; the
differential tests in `tests/differential/` exist to make that swap verifiable.
