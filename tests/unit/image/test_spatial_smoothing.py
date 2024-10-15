import unittest

import numpy as np
import xarray as xr
from sparse import GCXS
from xarray import DataArray

from depiction.image.smoothing.spatial_smoothing import SpatialSmoothing


class TestSpatialSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinates = np.array([[1, 2], [1, 3], [1, 4], [2, 2], [2, 4]])
        self.coordinates_shifted = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]])
        self.full_values_spatial = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 0, 5]])
        self.values_flat = np.array([1, 2, 3, 4, 5])
        self.values_spatial = np.array([[1, 2, 3], [4, 0, 5]])

    def _convert_array(self, arr: DataArray, variant: str) -> DataArray:
        if variant == "dense":
            return arr
        elif variant == "sparse":
            values = GCXS.from_numpy(arr)
            return DataArray(values, dims=arr.dims, coords=arr.coords, attrs=arr.attrs, name=arr.name)

    def _test_variants(self, *variants):
        for variant in variants:
            with self.subTest(variant=variant):
                yield variant

    def test_smooth(self) -> None:
        for variant in self._test_variants("dense", "sparse"):
            smoothing = SpatialSmoothing(sigma=100)
            values = DataArray(np.array([[[1.0, 1, 1], [1, 0, 1]], [[5, 5, 5], [5, 5, 5]]]), dims=["c", "y", "x"])
            values = self._convert_array(values, variant)

            smoothed_values = smoothing.smooth(values)
            expected_values = DataArray(
                np.stack([np.full((2, 3), 5 / 6), np.full((2, 3), 5)], axis=2), dims=["y", "x", "c"]
            )
            xr.testing.assert_allclose(expected_values, smoothed_values)

    def test_fill_background_when_nearest(self) -> None:
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
