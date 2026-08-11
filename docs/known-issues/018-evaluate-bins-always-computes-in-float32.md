# `evaluate_bins` always computes in float32 — the float64 guard can never be true

Severity: **low** (but see the warning below) | Status: **accepted — documented in the code**, 2026-08-11
File: `src/depiction/spectrum/evaluate_bins.py`, the `is_f64` guard in `EvaluateBins.evaluate`

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

## Decision taken

Of the two arms this file originally offered — fix and re-baseline, or leave and document — the
**second was taken**, on 2026-08-11. The guard is unchanged and now carries a comment at the
`is_f64` line saying that it never fires, that binning is float32 in practice, and that correcting
it means re-running `docs/refactoring/baseline-diff.md` and recording the new expected output.

Nothing about the analysis below has changed; this entry stays open in the sense that the
precision question is still live. What is closed is the risk the file was written about: a reader
of `evaluate_bins.py` no longer has to rediscover that the guard is dead.

Whoever picks up the first arm should: correct the test to
`np.issubdtype(mz_arr.dtype, np.float64)`, re-run the baseline diff on both acquisitions, and
record the new expected output in `baseline-diff.md` with a note saying why it changed.

The one thing to avoid is fixing it quietly, which turns a known quirk into an unexplained
regression in the archive.

## Notes

Precision impact is modest for typical MSI intensities but not nil for m/z, where float32 has
~7 significant digits and m/z values run to 5-6 digits before the decimal point.
