# depiction_io

Reading and writing mass-spectrometry imaging data, split out of `depiction` so that the
I/O layer has a boundary of its own.

## Depending on it from another application

Depend on this package rather than on `depiction`, unless you also need the processing,
image and pipeline code. `depiction` carries the Snakemake orchestration, the image stack
and the calibration and QC tools, which is the difference between a ~50-package install and
a ~200-package one.

```toml
[tool.uv.sources]
depiction_io = { git = "https://github.com/fgcz/depiction.git", subdirectory = "pkgs/depiction_io", rev = "..." }
```

Pin a `rev`. Without one the dependency tracks the default branch, and the only thing
between you and a surprise is your own lockfile.

Code written before the split imports from `depiction.persistence`, which no longer exists.
The reading side changed shape at the same time, so it is not only a rename:

| was | now |
|---|---|
| `ImzmlReadFile(path)` | `get_read_file(path)` |
| `read_file.get_reader()` | `with read_file.reader() as reader:` |
| `ImzmlReader` (as a type) | `GenericReader` |
| `ImzmlWriteFile` | unchanged |

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

The migration to imzy is complete: one reader, one writer, and roughly 1,900 lines of custom
parsing removed. The gaps imzy still has, the workarounds they get here, and the pieces of the
old parser that were kept on purpose are documented in
[`docs/modules/depiction_io/imzy_backend.md`](https://github.com/fgcz/depiction/blob/dev/docs/modules/depiction_io/imzy_backend.md).

The differential tests in `tests/differential/` were an A/B comparison between the two
readers for the length of that migration and are what made it safe. With one reader left
they assert against the corpus's independently held source arrays instead — still useful,
but no longer a second opinion. That page says what it stops catching.
