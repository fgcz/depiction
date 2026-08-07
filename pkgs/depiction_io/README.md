# depiction_io

Reading and writing mass-spectrometry imaging data, split out of `depiction` so that the
I/O layer has a boundary of its own.

## What is here

- `types.py` — the `GenericReader` / `GenericReadFile` / `GenericWriter` /
  `GenericWriteFile` protocols. This is the seam: everything else in `depiction` talks to
  these rather than to a file format, so a backend can be swapped underneath without the
  callers noticing.
- `imzy_backend/` — the protocols implemented on top of
  [`imzy`](https://github.com/vandeplaslab/imzy), plus the corrections imzy needs before it
  can be trusted with this data. Each carries an `# upstream:` marker.
- `imzml/` — the writer, the mode enum, and `parser/parse_metadata.py`. The last is what is
  left of a hand-rolled parser that used to do the reading: imzy parses no checksums and
  reports a pixel size of `1` where a file declares none, so metadata still comes from here.
- `ram/` — in-memory implementations of the same protocols, used heavily in tests.
- `backend.py`, `imzml_zip.py`, `file_checksums.py`, `pixel_size.py` — supporting pieces.

## Reading and writing

Both go through imzy. `pyimzml` and the hand-rolled parser are gone.

Construct read files with **`get_read_file(path)`** rather than naming a class. There is one
implementation behind it today, but it is the seam that made replacing the parser a one-line
change, and the choice it makes is still real: a non-imzML path (a Bruker `.d`) can only be
served by imzy, and imzy disables its Bruker readers on macOS — which `get_read_file` says
plainly instead of failing somewhere further in.

zlib-compressed files are read. imzy cannot do this on its own — it has the block offsets
but never parses `IMS:1000104`, so it would read a compressed `.ibd` as raw floats and
return noise — so `imzml_scan.py` collects the encoded lengths and `zlib_reader.py` inflates.
Numpress is refused rather than guessed at.

## What is deliberately not here

OME-TIFF and HDF5 image formats live in `depiction.image`, not in this package. They take
and return `MultiChannelImage`, so putting them here would make `depiction_io` depend on
`depiction` and defeat the split. That is also why `bioio`, `bioio-ome-tiff` and
`tifffile` are not dependencies of this package.

## Status

See [`docs/refactoring/ROADMAP.md`](../../docs/refactoring/ROADMAP.md) in the repository
root. The migration to imzy is complete: one reader, one writer, and roughly 1,900 lines of
custom parsing removed across Phases C and E.

The differential tests in `tests/differential/` were an A/B comparison between the two
readers for the length of that migration and are what made it safe. With one reader left
they assert against the corpus's independently held source arrays instead — still useful,
but no longer a second opinion. The roadmap says what that stops catching.
