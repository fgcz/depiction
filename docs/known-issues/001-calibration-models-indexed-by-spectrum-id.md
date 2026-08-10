# Calibration models are assigned to the wrong pixels

Severity: **critical** | Status: open | Found: 2026-08-10
File: `src/depiction/calibration/apply/apply_models.py:63`

## Symptom

Calibration produces silently wrong m/z values. Every spectrum is calibrated, no error is
raised and no warning is logged, but the model applied to a given spectrum may belong to a
different pixel. The output file looks entirely normal.

## Why it happens

`ApplyModels.calibrate_spectra` reads spectrum `spectrum_id` from the file and then takes its
coefficients positionally:

```python
mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
features = all_model_coefs.data_flat.isel(i=spectrum_id)
```

`data_flat` (`src/depiction/image/multi_channel_image.py:174`) is

```python
return self._data.stack(i=("y", "x")).isel(i=self.fg_mask_flat)
```

so `i` enumerates **foreground pixels in row-major (y, x) order**, while `spectrum_id`
enumerates **spectra in file order**. The two agree only when both of these hold:

1. the imzML is lexsorted by `(y, x)`, and
2. every pixel is foreground, so `fg_mask_flat` drops nothing.

Neither is checked or documented. Break either and the assignment silently shifts. Note that
condition 2 makes this sharper than "non-row-major files are broken": a single background
pixel offsets every subsequent model by one.

The code already knows: line 61 carries a pre-existing
`# TODO sanity check the usage of i as spectrum_id (i.e. check the coords!)`.

## How to reproduce

Any acquisition whose spectra are not in row-major order. `SubsampleImzml` with
`mode=randomized` (`src/depiction/tools/subsample_imzml.py`) emits exactly such a file, as do
serpentine/meander raster acquisitions.

Read the code path directly:

```
sed -n '60,67p' src/depiction/calibration/apply/apply_models.py
sed -n '172,175p' src/depiction/image/multi_channel_image.py
```

## Fix sketch

Index by coordinate rather than by position. `get_spectrum_with_coords` already returns
`coords`, so use them:

```python
features = all_model_coefs.data_spatial.sel(
    y=coords[1], x=coords[0]
)  # check axis order
```

If that is more surgery than you want, the cheap version is still a large improvement: assert
at the top of `calibrate_spectra` that the file's `coordinates_array_2d` is lexsorted by
`(y, x)` and that the foreground mask is full, and raise otherwise. Failing loudly on an
unsupported layout beats silently mis-calibrating it.

## Notes

Whichever fix is chosen, re-run the pipeline baseline diff
(`docs/refactoring/baseline-diff.md`) afterwards — this changes numeric output for any input
that currently triggers the bug, and is indistinguishable from a regression without it.
