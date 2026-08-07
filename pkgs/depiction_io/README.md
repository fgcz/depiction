# depiction_io

Reading and writing mass-spectrometry imaging data, split out of `depiction` so that the
I/O layer has a boundary of its own.

## What is here

- `types.py` — the `GenericReader` / `GenericReadFile` / `GenericWriter` /
  `GenericWriteFile` protocols. This is the seam: everything else in `depiction` talks to
  these rather than to a file format, so a backend can be swapped underneath without the
  callers noticing.
- `imzml/` — the legacy imzML reader (memory-mapped, picklable), the writer, and the
  XML/cvParam parser.
- `imzy_backend/` — the same protocols implemented on top of
  [`imzy`](https://github.com/vandeplaslab/imzy), plus the corrections imzy needs before it
  can be trusted with this data. Each carries an `# upstream:` marker.
- `ram/` — in-memory implementations of the same protocols, used heavily in tests.
- `backend.py`, `imzml_zip.py`, `file_checksums.py`, `pixel_size.py` — supporting pieces.

## Which backend reads, which backend writes

**Writing** goes through imzy unconditionally: `pyimzml` is gone.

**Reading** defaults to the legacy parser. `get_read_file(path)` returns an
`ImzmlReadFile` unless `DEPICTION_IO_BACKEND=imzy` is set or a backend is passed
explicitly, and a non-imzML path (a Bruker `.d`) always goes to imzy because nothing else
can open one. The default stays legacy because the imzy reader has not been checked against
a real acquisition, and because it cannot read zlib-compressed files at all — it refuses
them rather than returning the noise it would otherwise produce.

Note that the tools in `depiction` still construct `ImzmlReadFile` directly instead of
calling `get_read_file`, so the environment variable does not currently redirect them.
Routing those call sites through the seam is the first step of the roadmap's Phase E.

## What is deliberately not here

OME-TIFF and HDF5 image formats live in `depiction.image`, not in this package. They take
and return `MultiChannelImage`, so putting them here would make `depiction_io` depend on
`depiction` and defeat the split. That is also why `bioio`, `bioio-ome-tiff` and
`tifffile` are not dependencies of this package.

## Status

See [`docs/refactoring/ROADMAP.md`](../../docs/refactoring/ROADMAP.md) in the repository
root. The writer has been replaced and the reader exists but is not the default; deleting
the hand-rolled parser is Phase E and is blocked on a real dataset. The differential tests
in `tests/differential/` run every assertion against both readers, which is what makes the
remaining half of the swap verifiable.
