# `depiction_io` depends on `imzy` with no upper bound while overriding its private methods

Severity: **low** | Status: **fixed** | Found: 2026-08-10
File: `pkgs/depiction_io/pyproject.toml:24`

## Symptom

`imzy>=0.3.0` is unbounded. `depiction_io.imzy_backend.zlib_reader` overrides three private
`IMZMLReader` methods (`_read_spectrum`, `_read_spectra`, `_estimate_centroid_mass_range`) to
add zlib support that imzy lacks. If a future imzy renames or removes one of them, every
downstream application that depends on `depiction_io` fails at import, and there is nobody
here to cut a release.

## What is already handled

The design is deliberate and **fails loudly by construction** — this is not a silent-corruption
risk. `zlib_reader.py` pins the override list and asserts it at import:

```python
_ENCODED_READ_SITES = (
    "_read_spectrum",
    "_read_spectra",
    "_estimate_centroid_mass_range",
)
_missing = [name for name in _ENCODED_READ_SITES if not hasattr(IMZMLReader, name)]
if _missing:
    raise ImportError(...)
```

The module docstring explains why: if imzy grows a fourth read site, this raises rather than
silently reading compressed bytes as raw floats again. An earlier draft of this finding claimed
the monkeypatching was unguarded; that was wrong.

So the residual issue is narrow: the guard converts a silent-wrongness failure into a loud one,
but a dormant package still cannot *fix* the loud one.

## Fix

A judgment call, one line:

```toml
"imzy>=0.3.0,<0.4",
```

A cap means a future resolver picks a compatible imzy instead of installing a combination that
raises at import. The cost is that downstream users must lift the cap themselves once someone
verifies a newer imzy — which, for an unmaintained package, is the right default.

## Notes

Upstreaming the compression support is Phase D gap (1) in `docs/refactoring/ROADMAP.md`, the
only roadmap phase never started. If that ever happens, this whole module and the cap go away.
