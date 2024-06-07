from __future__ import annotations

from typing import Callable

import numpy as np
import xarray
from xarray import DataArray

from depiction.image.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware


def _apply_on_spatial_view(array: DataArray, fn: Callable[[DataArray], DataArray]) -> DataArray:
    # adjust indexing for the rest of the function
    array_flat = array.drop("i").set_xindex(["x", "y"])

    # perform the computation on 2d view
    array_2d = array_flat.unstack("i").transpose("y", "x", "c")
    array_2d = fn(array_2d)

    # trick: concatenate an additional channel that will indicate everything that was present before, including
    #        fully nan elements because they should not disappear like an actual background
    # this almost works but is broken:
    is_nan_before = (
        array_flat.isnull().all("c").astype(array_2d.dtype).expand_dims("c").unstack("i").transpose("y", "x", "c")
    )
    array_2d = xarray.concat([array_2d, is_nan_before], dim="c")

    # stack back the 2d view
    result = array_2d.stack(i=("x", "y")).dropna("i", how="all")

    # remove the additional channel
    result = result.isel(c=slice(0, -1))

    # revert the indexing
    x, y = result.x.values, result.y.values
    return result.drop_vars(["i", "x", "y"]).assign_coords(x=("i", x), y=("i", y), i=np.arange(len(result.i)))


# TODO should be refactored later

# TODO test this case : a spectrum is all nan before, but present in the flat repr, it should not disappear like an actual background


def smooth_image_features(all_features: DataArray, kernel_size: int, kernel_std: float) -> DataArray:
    def fn(array_2d: DataArray) -> DataArray:
        smoother = SpatialSmoothingSparseAware(
            kernel_size=kernel_size,
            kernel_std=kernel_std,
        )
        return smoother.smooth(array_2d, bg_value=np.nan)

    return _apply_on_spatial_view(all_features, fn)
