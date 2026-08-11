# `horizontal_concat` crashes on images of different heights — the case it documents supporting

Severity: **low** | Status: **fixed** | Found: 2026-08-10
File: `src/depiction/image/horizontal_concat.py:27`

## Symptom

```
ValueError: cannot reindex or align along dimension 'y' because the (pandas) index has
duplicate values
```

The function computes `ymax` across all inputs and pads each image up to it, so handling
unequal heights is clearly the intent — it just doesn't work.

## Why it happens

```python
data = data.pad(y=(0, ymax - data.y.values.max()), constant_values=0)
```

`constant_values=0` fills the padded **data**, but xarray fills the new **coordinate labels**
with `NaN`. The shorter image then carries several `y = NaN` labels, and the
`xarray.concat(concat, dim="x")` at line 35 aligns on `y`, where the repeated `NaN`s are
duplicate index values.

## How to reproduce

```python
import numpy as np, xarray
from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.horizontal_concat import horizontal_concat


def img(h, w, val):
    data = xarray.DataArray(
        np.full((h, w, 1), val, float),
        dims=("y", "x", "c"),
        coords={"y": np.arange(h), "x": np.arange(w), "c": ["ch"]},
    )
    fg = xarray.DataArray(
        np.ones((h, w), bool),
        dims=("y", "x"),
        coords={"y": np.arange(h), "x": np.arange(w)},
    )
    return MultiChannelImage(data, fg)


horizontal_concat([img(3, 2, 1.0), img(5, 2, 2.0)])  # ValueError
```

xarray also emits a `FutureWarning` here about `join` defaulting from `outer` to `exact`,
which will turn this into a different error in a future xarray.

## Fix

Re-label the y axis after padding, and pad by height rather than by largest y label:

```python
ymax = max(image.data_spatial.sizes["y"] for image in images)  # was: y.values.max()
...
data = data.pad(y=(0, ymax - data.sizes["y"]), constant_values=0)
data = data.assign_coords(y=np.arange(data.sizes["y"]))
```

The `sizes` change was not in the original finding. Re-labelling alone fixes the crash, but
leaves the padding amount derived from the largest y *label*, and the two only agree for
0-based contiguous coordinates. Given an image with an offset y origin the old expression
invents rows that no pixel occupies — with the relabel in place that produced a silently wrong
result instead of the previous crash, which is worse. `sizes` is identical for every input
this repo actually produces and coherent for the rest.

Tests cover unequal heights and an offset y origin; the pre-existing tests all used
equal-height 0-based inputs, which is why this survived.

## Notes

An earlier draft of this file claimed the only in-repo caller is `depiction_cluster_sandbox`.
That is wrong: `image/multi_channel_image_concatenation.py:88` calls it too, from
`MultiChannelImageConcatenation.concat_images`. Every test of that path passes images of
equal height, which is why the crash stayed hidden.

Images built with bare `from_spatial` and no `y`/`x` coordinate labels concatenate fine; the
crash needs labelled coordinates, which is what `from_flat` and `read_hdf5` produce.
