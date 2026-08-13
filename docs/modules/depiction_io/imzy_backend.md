# The imzy backend

All imzML reading and writing goes through [imzy](https://github.com/vandeplaslab/imzy),
behind the `GenericReadFile` / `GenericReader` / `GenericWriter` protocols in
`depiction_io.types`. There is one reader and one writer; the hand-rolled etree parser and the
`pyimzml` writer that preceded them are gone, along with roughly 1,900 lines of custom parsing.

This page is the rationale for the shape of `depiction_io.imzy_backend`: the gaps imzy has
that it works around, the pieces of the old parser that were kept on purpose, and the
behaviour a caller inherits from imzy. It is not a changelog — see
[`docs/refactoring/REMAINING.md`](https://github.com/fgcz/depiction/blob/dev/docs/refactoring/REMAINING.md)
for what is still open.

## The five gaps

Each is worked around in-tree and carries an `# upstream:` marker at the workaround naming the
gap by the number it has here. None has been reported upstream yet. Assume upstream review is
slow: **do not block on merge.** Carry each fix in the adapter or a pinned fork, and leave a
`# upstream: <PR url>` marker at every carried patch.

### (1) No zlib support

`_read_spectrum` is a bare `np.frombuffer(mz_bytes, dtype=self.mz_precision)`. Nothing in imzy
inspects the compression cvParam, and `byte_offsets` stores *array* lengths (`IMS:1000103`)
rather than *encoded* lengths (`IMS:1000104`). A zlib-compressed `.ibd` therefore **does not
fail — it yields plausible-looking noise.** Reader half only; neither toolchain writes
compressed output.

Worked around in full. `imzml_scan.py` collects the `IMS:1000104` encoded lengths during the
streaming walk it already performs, keyed by the block's offset — so the reader never has to
work out which `binaryDataArray` was the m/z one, and a continuous file's shared m/z block
collapses to one entry. `zlib_reader.py` subclasses `IMZMLReader` and inflates at the three
places imzy turns bytes into floats; `_ENCODED_READ_SITES` pins that list so an imzy upgrade
adding a fourth fails loudly rather than silently returning noise again. Numpress stays
refused — undoing it needs a codec, not a length.

Upstream would want this differently (`process_spectrum` parsing `IMS:1000104`, and the cache
format carrying it), so the patch is not directly portable. The silent-corruption behaviour is
worth reporting as a bug in its own right, independent of any fix.

### (2) Readers are not picklable

No `__getstate__` / `__setstate__`, and `WriteSpectraParallel` needs them. Simpler than it
first appears: `BaseReader._get_reader_kwargs()` returns `{}` and `IMZMLReader` does not
override it, so for imzML this is just "re-open by path". `ImzyReader` implements both, and
carries the encoded lengths through the pickle rather than re-deriving them — recovering them
means another walk of the XML, which is the expensive half of opening a large file.

### (3) The `.icache` sidecar is unvalidated, unrelocatable and racy

Three problems in one file. The first is the most dangerous thing found in the whole migration.

- **No provenance check.** `_init` reloads `<stem>.icache` on sight — no size, mtime or UUID
  comparison — so a cache left from an earlier file at the same path makes imzy report *that*
  file's spectrum count and coordinates while slicing the *current* `.ibd`. Reproducible in
  four lines, and reachable by any pipeline that regenerates an output it has already read.
  Worked around on both sides: `ImzmlWriteFile` deletes the cache before writing, and
  `ImzyReader._discard_stale_icache` deletes one that predates its `.imzML`. Storing the
  `.ibd` UUID in the cache would fix it properly.
- **No `cache_dir`.** A read-only input tree degrades to a full re-parse on every open, and
  `_write_icache_safely` swallows the failure.
- **A fixed temp name.** `write_icache` always writes `<stem>.icache.npz` before renaming, so
  concurrent first reads of one file — exactly what `ReadSpectraParallel` does — clobber each
  other. Benign in practice, since the `OSError` is swallowed and the rename is atomic, but it
  should be `mkstemp`-based.

### (4) The writer silently drops empty spectra

`IMZMLWriter.add_spectrum` catches its own `_EmptySpectrumError`, warns and returns `False` —
*before* consulting the `on_error="error"` setting it was given. `filter_peaks` can emit an
empty spectrum today and `pyimzml` wrote them, so a naive writer flip would drop pixels and
desynchronise `n_spectra` from `coordinates`. Worked around in `ImzmlWriter.add_spectrum`,
which refuses an empty m/z array up front and treats a `False` return as an error.

### (5) The writer always emits a z coordinate

`_normalize_coordinates` appends `z = 1` to a 2D coordinate and `_add_scan_list` writes
`IMS:1000052` unconditionally, where `pyimzml` writes it only for a 3-tuple. The more
dangerous of the two writer gaps: it would have added a z axis to every acquisition the
pipeline produces, and since `reader.coordinates[i]` is handed straight back to `add_spectrum`
in five tools, one round trip would have made the change permanent. Worked around in
`DepictionIMZMLWriter`, which removes the cvParam again when no spectrum was given 3D
coordinates — the one place this codebase reaches into imzy's private methods, so it fails
loudly if either method changes.

## Kept deliberately

A successor should **not** "finish the job" by deleting these:

- **`imzml/parser/parse_metadata.py`** — what is left of the hand-rolled parser. It is how
  `ImzyReadFile` gets checksums and a pixel size that is `None` rather than `1` when the file
  declares no `IMS:1000046`. imzy parses no checksums and reports `1` in that case, which is a
  fabricated physical size, not a missing one.
- **`imzml_alignment_tracker.py`** — still used by the writer.

`get_read_file` also stays, even though there is only one backend: the choice it makes for a
vendor format on macOS is still real, and it reports plainly when a format cannot be read on
this platform. Construct read files through it rather than naming a class.

## Behaviour a caller inherits

Two consequences of imzy refusing to write nothing:

- **An empty spectrum is refused by the writer.** `pyimzml` wrote it; imzy warns and drops it,
  which would desynchronise the pixel count; this adapter raises. The one caller that can
  produce one, `filter_peaks`, drops the pixel explicitly and logs it — which is what its
  sibling `pick_peaks` has always done for the same situation. **The policy sits in the tool,
  where it is visible, rather than in the I/O layer, where it was silent.**
- **Closing a writer with no spectra raises.** Reachable from `SubsampleImzml` with a ratio
  that rounds to zero, and from `CutoutRectangularRegion` with an empty selection — both of
  which used to produce a file, though a malformed one. The error does not mask a failure
  raised inside the `with` body.

Also: imzy rewrites the output suffix to `.imzML` rather than using the path it was given, so
`ImzmlWriter.open` rejects any other spelling instead of quietly writing somewhere else.

## Why imzy is capped at `<0.4`

`zlib_reader.py` overrides three *private* `IMZMLReader` methods and asserts they exist at
import, so a future imzy that renames one becomes an `ImportError` for every downstream
consumer, with nobody here to cut a release. The full reasoning is inline at
`pkgs/depiction_io/pyproject.toml`, next to the pin. Lift the cap once someone has verified a
newer imzy against `tests/differential/`.

## What the written imzML looks like

These are differences from the `pyimzml` output this repository produced for years. None of
them is a difference in the data — the m/z and intensity arrays are byte-identical — but a
successor comparing files directly will see them.

- **`userParam name="imzy export timestamp"` is the one thing preventing a byte-reproducible
  imzML.** Worth knowing for anyone who wants one.
- **`IMS:1000044` / `IMS:1000045` (max dimension x/y) carry the pixel count**, not a physical
  length. That is dimensionally wrong, and it is imzy's one regression against `pyimzml`,
  which omitted them. Nothing in this codebase reads them, so it is inert here.
- **`unitName` for intensity is `number of detector counts`**, where `pyimzml` wrote `number of
  counts`. imzy's is the MS ontology's name.
- **The per-spectrum summary cvParams are written at lower precision** —
  `1220.0382205078124` becomes `1220.03822051`, `198.0` becomes `198`. These are the
  descriptive `lowest observed m/z` / `base peak` / `total ion current` values, not the data.
- **`.icache` sidecars are left next to the imzML files.** They are imzy's parse caches, not
  pipeline artifacts, and are not in `get_result_files_new`.

Three where imzy is **more correct** than what it replaced:

- `pyimzml` wrote `cvList count="2"` and used `cvRef="IMS"` throughout without ever declaring
  the IMS ontology. imzy declares it.
- `pyimzml` wrote `instrumentConfigurationRef="instrumentConfiguration0"` on every spectrum
  while the only `<instrumentConfiguration>` it declared had `id="IC1"` — **a dangling
  reference in every imzML this pipeline has ever produced.** imzy writes `IC1`.
- `pyimzml`'s writer tag claimed version 1.5.4 while the installed package was 1.5.5.

That the migration changed nothing in the data was established by running the `CALIB_IMAGES`
chain on two real acquisitions in this tree and in the tree before the migration: every
artifact identical with no tolerance, and the `.ibd` files byte-identical after their 16-byte
UUID header. The harness is in
[`system_tests/baseline/`](https://github.com/fgcz/depiction/blob/dev/system_tests/baseline/README.md).

## What is not exercised

- **Bruker `.d` (TSF/TDF/NeoFlex) and Waters `.raw`** are plumbed and reachable through
  `get_read_file`, but exercised by nothing: they cannot run on macOS — `imzy/plugins.py`
  disables them — and there is no fixture.
- **zlib** has the code path, mutation-checked, and synthetic twins from
  `tests/differential/corpus.py` (`compress_case`). What is missing is a compressed file
  somebody else's software wrote, which is the only thing that would test the assumption that
  they compress each binary array whole. Nothing in either toolchain can *produce* one.
- **`processed` mode** has no real-acquisition fixture; both public fixtures are `continuous`.
  See [`docs/test-data.md`](https://github.com/fgcz/depiction/blob/dev/docs/test-data.md).
- **The differential suite is no longer an A/B comparison.** It keeps most of its value
  because `corpus.Case` carries the source arrays independently of any reader, so the
  assertions are backend-against-ground-truth, and `RamReadFile` still cross-checks. What is
  genuinely gone is a second *XML parser* to disagree with: a bug shared symmetrically by this
  writer and this reader would not show up there. Only real acquisitions catch that class of
  thing.
