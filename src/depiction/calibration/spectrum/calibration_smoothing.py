from __future__ import annotations

import numpy as np
from xarray import DataArray

from depiction.image.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware


# TODO should be refactored later


def smooth_image_features(all_features: DataArray, kernel_size: int, kernel_std: float) -> DataArray:
    features_flat = all_features.drop("i").set_xindex(["x", "y"])
    features_2d = features_flat.unstack("i").transpose("y", "x", "c")
    smoother = SpatialSmoothingSparseAware(
        kernel_size=kernel_size,
        kernel_std=kernel_std,
    )
    result = smoother.smooth(features_2d, bg_value=np.nan)
    # TODO a bit ugly...
    result = result.stack(i=("x", "y")).dropna("i", how="all")
    x, y = result.x.values, result.y.values
    return result.drop_vars(["i", "x", "y"]).assign_coords(x=("i", x), y=("i", y), i=np.arange(len(result.i)))
