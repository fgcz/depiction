import unittest
from datetime import timedelta
from functools import cached_property

import numpy as np
from hypothesis import given, strategies, settings
from sparse import GCXS
from xarray import DataArray

from depiction.image.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware
from depiction.misc.integration_test_utils import IntegrationTestUtils


class TestSpatialSmoothingSparseAware(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_kernel_size = 5
        self.mock_kernel_std = 1.0
        self.mock_use_interpolation = False
        IntegrationTestUtils.treat_warnings_as_error(self)

    @cached_property
    def mock_smooth(self) -> SpatialSmoothingSparseAware:
        return SpatialSmoothingSparseAware(
            kernel_size=self.mock_kernel_size,
            kernel_std=self.mock_kernel_std,
            use_interpolation=self.mock_use_interpolation,
        )

    def _convert_array(self, arr: DataArray, variant: str) -> DataArray:
        # TODO duplicated with SpatialSmoothing test
        if variant == "dense":
            return arr
        elif variant == "sparse":
            values = GCXS.from_numpy(arr)
            return DataArray(values, dims=arr.dims, coords=arr.coords, attrs=arr.attrs, name=arr.name)

    def _test_variants(self, *variants, subtest=True):
        # TODO duplicated with SpatialSmoothing test (but this one is extended)
        for variant in variants:
            if subtest:
                with self.subTest(variant=variant):
                    yield variant
            else:
                yield variant

    def test_smooth_when_unchanged(self) -> None:
        for variant in self._test_variants("dense", "sparse"):
            image = DataArray(np.concatenate([np.ones((5, 5, 1)), np.zeros((5, 5, 1))], axis=0), dims=("y", "x", "c"))
            image = self._convert_array(image, variant)
            smoothed = self.mock_smooth.smooth(image, bg_value=0)
            np.testing.assert_array_almost_equal(1.0, smoothed.values[:5, :, 0], decimal=8)
            np.testing.assert_array_almost_equal(0.0, smoothed.values[-5:, :, 0], decimal=8)
            self.assertEqual(("y", "x", "c"), smoothed.dims)

    @given(strategies.floats(min_value=0, max_value=1e12, allow_subnormal=False))
    @settings(deadline=timedelta(seconds=1))
    def test_smooth_preserves_values(self, fill_value) -> None:
        for variant in self._test_variants("dense", "sparse", subtest=False):
            dense_image = DataArray(np.full((2, 5, 1), fill_value=fill_value), dims=("y", "x", "c"))
            image = self._convert_array(dense_image, variant)
            smoothed = self.mock_smooth.smooth(image, bg_value=0)
            np.testing.assert_allclose(dense_image.values, smoothed.values, rtol=1e-8)

    def test_smooth_when_bg_nan(self) -> None:
        for variant in self._test_variants("dense", "sparse"):
            dense_image = DataArray(
                np.concatenate([np.full((5, 5, 1), np.nan), np.ones((5, 5, 1)), np.zeros((5, 5, 1))]),
                dims=("y", "x", "c"),
            )
            image = self._convert_array(dense_image, variant)
            smoothed = self.mock_smooth.smooth(image, bg_value=np.nan)
            np.testing.assert_array_equal(np.nan, smoothed.values[:5, 0])
            np.testing.assert_array_almost_equal(1.0, smoothed.values[5:8, :, 0], decimal=8)
            np.testing.assert_array_almost_equal(0.0, smoothed.values[-3:, :, 0], decimal=8)
            smoothed_part = smoothed.values[8:12, :, 0]
            for i_value, value in enumerate([0.94551132, 0.70130997, 0.29869003, 0.05448868]):
                np.testing.assert_array_almost_equal(value, smoothed_part[i_value, :], decimal=8)

    def test_smooth_casts_when_integer(self) -> None:
        for variant in self._test_variants("dense", "sparse"):
            image_dense = DataArray(np.full((2, 5, 1), fill_value=10, dtype=int), dims=("y", "x", "c"))
            image = self._convert_array(image_dense, variant)
            res_values = self.mock_smooth.smooth(image=image)
            self.assertEqual(np.float64, res_values.dtype)
            np.testing.assert_allclose(image_dense.values, res_values.values, rtol=1e-8)

    def test_smooth_dense_when_use_interpolation(self) -> None:
        self.mock_use_interpolation = True
        mock_image = np.full((9, 5), fill_value=3.0)
        mock_image[4, 2] = np.nan
        smoothed = self.mock_smooth._smooth_dense(image_2d=mock_image, bg_value=np.nan)
        self.assertEqual(0, np.sum(np.isnan(smoothed)))
        self.assertAlmostEqual(3, smoothed[4, 2], places=6)

    def test_gaussian_kernel(self) -> None:
        self.mock_kernel_size = 3
        self.mock_kernel_std = 1.0
        expected_arr = np.array(
            [
                [0.075114, 0.123841, 0.075114],
                [0.123841, 0.20418, 0.123841],
                [0.075114, 0.123841, 0.075114],
            ]
        )
        np.testing.assert_allclose(expected_arr, self.mock_smooth.gaussian_kernel, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
