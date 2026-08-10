# `evaluate_bins` always computes in float32 — the float64 guard can never be true

Severity: **low** (but see the warning below) | Status: open | Found: 2026-08-10
File: `src/depiction/spectrum/evaluate_bins.py:41`

## Symptom

```python
is_f64 = isinstance(mz_arr.dtype, (float, np.float64)) or isinstance(
    int_arr.dtype, (float, np.float64)
)
dtype = np.float64 if is_f64 else np.float32
```

`mz_arr.dtype` is a `numpy.dtype` **object**, never an instance of `float` or `np.float64`, so
`is_f64` is always `False` and every binning operation silently downcasts to float32 —
including float64 input, where the intent was clearly to preserve precision.

```pycon
>>> import numpy as np
>>> isinstance(np.arange(5, dtype=np.float64).dtype, (float, np.float64))
False
```

The correct test is `mz_arr.dtype == np.float64` or `np.issubdtype(mz_arr.dtype, np.float64)`.

## Why this is *not* filed as a straightforward fix

Fixing it changes the numeric output of **every mean spectrum and every binned intensity** in
the pipeline. Those outputs are what `docs/refactoring/baseline-diff.md` pinned when it
established that the imzy migration was output-identical. Correcting the guard would break that
equivalence for a reason unrelated to the migration, and there would be nobody around to
explain the difference.

## Fix sketch — pick one, do not half-do it

- **Fix and re-baseline:** correct the guard, re-run the baseline diff on both acquisitions,
  and record the new expected output in `baseline-diff.md` with a note saying why it changed.
- **Leave and document:** add a comment at line 41 stating that the guard is inert, that
  binning is float32 in practice, and that this is deliberate for baseline stability.

The one thing to avoid is fixing it quietly, which turns a known quirk into an unexplained
regression in the archive.

## Notes

Precision impact is modest for typical MSI intensities but not nil for m/z, where float32 has
~7 significant digits and m/z values run to 5-6 digits before the decimal point.
