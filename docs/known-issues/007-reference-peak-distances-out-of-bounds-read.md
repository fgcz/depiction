# Out-of-bounds read in the numba kernel fabricates calibration distances

Severity: **medium** | Status: open | Found: 2026-08-10
File: `src/depiction/calibration/spectrum/reference_peak_distances.py:45`

## Symptom

When no peak lies within the search window of a reference mass, the function should report
`nan`. Instead it reads whatever is adjacent in memory and may report a small, plausible-looking
distance, which then feeds the calibration model fit as if it were a real observation.

## Why it happens

```python
if i_left < i_right:
    i_max = i_left + np.argmax(peak_int_arr[i_left:i_right])
    s_dist_mz = peak_mz_arr[i_max] - mz_ref
else:
    s_dist_mz = peak_mz_arr[i_left] - mz_ref  # <- no peak in window
```

The `else` branch runs precisely when the window is empty. For a reference mass above the last
peak, `np.searchsorted` returns `i_left == len(peak_mz_arr)`, so `peak_mz_arr[i_left]` indexes
one past the end. The function is `@njit` with bounds checking off, so instead of an
`IndexError` it reads adjacent memory. The result then passes the
`if abs(s_dist_mz) <= max_distance_mz` gate whenever that garbage happens to be close.

## How to reproduce

Call with a reference mass above the largest peak m/z. The observed value depends on what is
adjacent in memory, so it differs between a standalone array and a view — that
nondeterminism is itself the tell. In practice the fabricated value is usually still rejected
by the distance gate (the adjacent garbage is far away), so the realistic impact is occasional
spurious observations rather than a systematically biased fit.

## Fix sketch

Delete the `else` branch. No candidate peak means the entry should stay `nan`, which is what
the surrounding code already expects. Highest value-per-minute item on the whole list.

Add a test with a reference mass above the last peak asserting `nan`.

## Notes

Because the current behaviour is memory-dependent, fixing it can change calibration output
slightly on inputs that previously got a lucky-but-accepted garbage distance. Re-run the
baseline diff if that matters.
