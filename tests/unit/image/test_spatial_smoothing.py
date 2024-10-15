import unittest

import numpy as np
import pytest
import xarray as xr
from sparse import GCXS
from xarray import DataArray

from depiction.image.smoothing.spatial_smoothing import SpatialSmoothing


def _convert_array(arr: DataArray, variant: str) -> DataArray:
    if variant == "dense":
        return arr
    elif variant == "sparse":
        values = GCXS.from_numpy(arr)
        return DataArray(values, dims=arr.dims, coords=arr.coords, attrs=arr.attrs, name=arr.name)


@pytest.mark.parametrize("variant", ["dense", "sparse"])
def test_smooth(variant: str) -> None:
    smoothing = SpatialSmoothing(sigma=100)
    values = DataArray(np.array([[[1.0, 1, 1], [1, 0, 1]], [[5, 5, 5], [5, 5, 5]]]), dims=["c", "y", "x"])
    values = _convert_array(values, variant)

    smoothed_values = smoothing.smooth(values)
    expected_values = DataArray(np.stack([np.full((2, 3), 5 / 6), np.full((2, 3), 5)], axis=2), dims=["y", "x", "c"])
    xr.testing.assert_allclose(expected_values, smoothed_values)


def test_fill_background_when_nearest() -> None:
    b = 0
    arr = DataArray(
        [
            [
                [b, b, b, b, b, b],
                [b, 4, 4, 4, b, b],
                [b, 5, 5, 5, b, b],
                [b, 6, 7, 8, b, b],
                [b, b, b, b, b, b],
            ]
        ],
        dims=("c", "y", "x"),
    )
    smoothing = SpatialSmoothing(sigma=1.0, background_fill_mode="nearest")
    filled_arr = smoothing._fill_background(values=arr)
    expected_arr = DataArray(
        [
            [[4], [4], [4], [4], [4], [4]],
            [[4], [4], [4], [4], [4], [4]],
            [[5], [5], [5], [5], [5], [5]],
            [[6], [6], [7], [8], [8], [8]],
            [[6], [6], [7], [8], [8], [8]],
        ],
        dims=("y", "x", "c"),
    ).transpose("c", "y", "x")
    np.testing.assert_array_equal(expected_arr, filled_arr)


if __name__ == "__main__":
    unittest.main()
