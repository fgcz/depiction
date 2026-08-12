# MultiChannelImage

This class makes our code a bit more standardized on how multi-channel images are handled internally.
We use Xarray for the representation of the data and basically provide a wrapper around it providing reusable blocks of common functionality.

The channels have coordinates in the DataArray, describing the channel names; this is required, and
the constructor raises if they are missing or not strings.

## Dimension conventions

Dimensions are always named, so prefer `.transpose(...)` and `.sel(...)` over positional indexing.
Where the order below matters, it is because something downstream reads `.values` directly.

**Spatial (dense) form: `(y, x, c)`.** This is the one convention that is actually enforced rather
than merely documented — `MultiChannelImage.__init__` transposes every incoming array to
`("y", "x", "c")` (and the mask to `("y", "x")`), so an instance has that layout regardless of what
the caller passed. `(c, y, x)` appears only as a deliberate transpose at a format boundary, in the
OME-TIFF and OME-NGFF writers, and never inside the class.

**Flat (sparse) form: `(i, c)` going in, `(c, i)` coming out.** `MultiChannelImage.from_flat` and
`SparseRepresentation` take `(i, c)`, but `data_flat` returns **`("c", "i")`**, because
`stack(i=("y", "x"))` appends the stacked dimension. Callers that need `(i, c)` transpose
explicitly — `workflow/proc/cluster_kmeans.py` does, and several call sites use `.values.T`. The
`i` dimension carries a pandas MultiIndex named `("y", "x")`.

**Coordinates: `(i, d)` with `d = ["x", "y"]`.** This is validated, not just assumed:
`_validate_coordinates` transposes, sorts on `d` and raises unless the result is `["x", "y"]`, and
`SparseRepresentation.flat_to_spatial` checks the same thing.

**The y axis increases downward**, as in an image, not as in a plot. An imzML y coordinate maps
directly to the array row index and nothing in the chain flips it: not the reader, not
`SparseRepresentation`, not the OME-TIFF or OME-NGFF writers, and not the plotting helpers — which
pass `origin="upper"` or `yincrease=False` to say so explicitly. There is no `flipud`,
`origin="lower"` or reversed image slice anywhere in `src/`, `pkgs/` or `tests/`.
`system_tests/calibration/test_pipeline_calibration_only.py` pins this end to end: it reconstructs
acquisition coordinates from the foreground mask after a real OME-TIFF round trip and asserts set
equality against the imzML coordinate list, which would fail if any stage flipped the axis.

### Known exceptions

These are the places that do not follow the above. Each carries a matching `TODO` in the code:

- `MultiChannelImage.coordinates_flat` returns `(d, i)` with `d = ["y", "x"]` — reversed on both
  axes. The conforming version is in the file directly above it, commented out. It does not cause
  bugs because `_validate_coordinates` sorts on `d`, which alphabetically restores `["x", "y"]` and
  carries the values with it.
- `MultiChannelImage.data_flat`'s own docstring claims `(i, c)`; it returns `(c, i)`, and the tests
  pin `(c, i)`.
- `MultiChannelImageConcatenation`'s public `min_coords` is `(y, x)`, because it is broadcast
  against `coordinates_flat` above.

## Mask-based / sparse representation

An image is stored either dense (`(y, x, c)`, a full grid) or sparse (`(i, c)` values plus
`(i, d)` coordinates, listing only acquired pixels). `SparseRepresentation` converts between the
two, and `is_foreground` is what records which grid positions were acquired — it is a required
constructor argument, so a `MultiChannelImage` always knows its own foreground.

The `y` and `x` coordinates are *not* always present. Built through `from_flat` they are, and they
carry the acquisition's own origin rather than starting at zero. Built from a bare spatial array
they are absent, and the OME-TIFF round trip drops them — it restores only `c`.

## Flat representation

`data_flat` drops background pixels: it stacks `y` and `x` into `i` and then selects on the
foreground mask, so its length is the number of acquired pixels, not `y * x`. The `i` index is a
MultiIndex over `("y", "x")`, which is what lets `unstack("i")` reconstruct the grid — and why an
operation that rebuilds a flat array from scratch has to restore that index before unstacking.

## API Reference

```{eval-rst}
.. autoclass:: depiction.image.MultiChannelImage
    :members:
    :undoc-members:
    :show-inheritance:
```
