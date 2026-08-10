# `RamReader` / `RamReadFile` do not implement the protocols the README promises

Severity: **medium** | Status: open | Found: 2026-08-10
Files: `pkgs/depiction_io/src/depiction_io/ram/ram_reader.py:14`,
`pkgs/depiction_io/src/depiction_io/ram/ram_read_file.py:16`

## Symptom

`pkgs/depiction_io/README.md:42` advertises the RAM backend as "in-memory implementations of
the same protocols". They are not. Passing one where a `GenericReader` / `GenericReadFile` is
expected raises `AttributeError` on the first protocol method the caller touches — including
`GenerateIonImage.generate_ion_images_for_file`, which calls `coordinates_array_2d`.

This matters more than its severity suggests: `depiction_io` is the package the root README
now tells downstream applications to depend on, so this fails for exactly the audience being
courted. The failure is at least loud and immediate, not silent.

## Why it happens

Neither class declares a base, so neither inherits the protocol default implementations:

```
RamReadFile  bases=['object']
   missing vs GenericReadFile: compact_metadata, coordinates_array_2d, is_checksum_valid,
                               pixel_size, print_summary, summary
RamReader    bases=['object']
   missing vs GenericReader:   coordinates_array_2d, get_spectrum_coordinates,
                               get_spectrum_with_coords
```

## How to reproduce

```python
import numpy as np
from depiction_io.ram.ram_write_file import RamWriteFile
from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum

wf = RamWriteFile(imzml_mode=ImzmlModeEnum.PROCESSED)
with wf.writer() as w:
    w.add_spectrum(np.array([100.0, 200.0]), np.array([1.0, 2.0]), (1, 1, 1))
wf.to_read_file().coordinates_array_2d
# AttributeError: 'RamReadFile' object has no attribute 'coordinates_array_2d'
```

The generic diff above is reproducible with:

```python
from depiction_io import types
from depiction_io.ram.ram_reader import RamReader

sorted(
    {n for n in dir(types.GenericReader) if not n.startswith("_")}
    - {n for n in dir(RamReader) if not n.startswith("_")}
)
```

## Fix sketch

Declare the bases — `class RamReader(GenericReader)`, `class RamReadFile(GenericReadFile)` —
and check that the inherited defaults actually work against the in-memory backing store (they
are `Protocol`s with default implementations, so this may be nearly free, but verify rather
than assume).

Then add a conformance test that loops over every public protocol member × every concrete
backend. That single test is what would have caught this, and it also guards the imzy and
imzML backends.

If the subclassing turns out non-trivial, the honest alternative is to downgrade the README
sentence to say the RAM backend implements only the write path.

## Notes

`RamWriteFile.to_read_file()` itself works fine — an earlier draft of this finding claimed it
raised `TypeError`, which is wrong. The construction succeeds; only the protocol surface is
missing.
