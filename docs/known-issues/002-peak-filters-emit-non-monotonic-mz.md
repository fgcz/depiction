# `filter_index_peaks` returns peaks in intensity order, not m/z order

Severity: **high** | Status: open | Found: 2026-08-10
File: `src/depiction/spectrum/peak_filtering/filter_n_highest_intensity.py:27`

## Symptom

With `peak_picker_type: BasicUninterpolated` and any n-highest-intensity peak filter, the
written peak imzML has a **non-monotonic m/z array**. That violates the imzML expectation of
ascending m/z and silently breaks every downstream consumer that assumes sorted input —
`np.searchsorted`, binning, ion-image extraction, and any external tool reading the file.

## Why it happens

`FilterNHighestIntensity` has two methods that should agree, and don't:

- `filter_peaks` (line 49) sorts the selected indices back into m/z order:
  `np.sort(np.argsort(peak_int_arr)[-self.max_count:])`
- `filter_index_peaks` (line 27) does **not**: `peak_idx_arr[sorted_idx[-self.max_count:]]`

`FilterNHighestIntensityPartitioned` inherits the defect: it delegates per partition to
`FilterNHighestIntensity.filter_index_peaks` and `extend`s the results
(`filter_n_highest_intensity_partitioned.py:65`), so partitions are ascending but the peaks
within each partition are not.

`BasicPeakPicker` (`peak_picking/basic_peak_picker.py:64`) is the only picker that calls
`filter_index_peaks`; the other three call `filter_peaks` and are unaffected.

## How to reproduce

```python
import numpy as np
from depiction.spectrum.peak_filtering import FilterNHighestIntensity

mz = np.arange(100, 200, 10.0)
inten = np.array([10, 50, 90, 30, 70, 20, 80, 40, 60, 15.0])
f = FilterNHighestIntensity(max_count=3)
idx = np.array([1, 2, 4, 6, 8])
print(mz[f.filter_index_peaks(mz, inten, idx)])  # [140. 160. 120.]  <- not ascending
print(
    f.filter_peaks(mz, inten, mz[idx], inten[idx])[0]
)  # [120. 140. 160.]  <- ascending
```

Partitioned variant, same result:

```python
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
from depiction.spectrum.peak_filtering.filter_n_highest_intensity_partitioned import (
    FilterNHighestIntensityPartitionedConfig as C,
)

f = FilterNHighestIntensityPartitioned(config=C(max_count=4, n_partitions=2))
# -> mz [110. 140. 190. 170.], ascending? False
```

## Fix sketch

One line — sort before returning:

```python
return np.sort(peak_idx_arr[sorted_idx[-self.max_count :]])
```

Add a test asserting ascending m/z from **both** `filter_index_peaks` and `filter_peaks`, for
both the plain and the partitioned filter. That is the assertion whose absence let the two
methods drift apart.

## Notes

The shipped presets do not hit this: they use the FindMF picker, which routes through the
correctly-sorted `filter_peaks`. It is reachable only via the opt-in `BasicUninterpolated`
picker plus a configured `peak_filtering` — a supported combination, just not a default one.

Fixing this changes numeric output for that configuration; re-run
`docs/refactoring/baseline-diff.md` if you rely on it.
