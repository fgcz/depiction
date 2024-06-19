from __future__ import annotations

import numpy as np
from xarray import DataArray

from depiction.image.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware
from depiction.image.xarray_helper import XarrayHelper


# TODO should be refactored later

# TODO test this case : a spectrum is all nan before, but present in the flat repr, it should not disappear like an actual background


def smooth_image_features(all_features: DataArray, kernel_size: int, kernel_std: float) -> DataArray:
    def fn(array_2d: DataArray) -> DataArray:
        smoother = SpatialSmoothingSparseAware(
            kernel_size=kernel_size,
            kernel_std=kernel_std,
        )
        return smoother.smooth(array_2d, bg_value=np.nan)

    return XarrayHelper.apply_on_spatial_view(all_features, fn)
    # return _apply_on_spatial_view(all_features, fn)
