from datetime import timedelta
from functools import cached_property

from hypothesis import given, strategies, settings

from ionmapper.misc.integration_test_utils import IntegrationTestUtils
from ionmapper.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware
import numpy as np
import unittest


class TestSpatialSmoothingSparseAware(unittest.TestCase):
    def setUp(self):
        self.mock_kernel_size = 5
        self.mock_kernel_std = 1.0
        self.mock_use_interpolation = False
        IntegrationTestUtils.treat_warnings_as_error(self)

    @cached_property
    def mock_target(self) -> SpatialSmoothingSparseAware:
        return SpatialSmoothingSparseAware(
            kernel_size=self.mock_kernel_size,
            kernel_std=self.mock_kernel_std,
            use_interpolation=self.mock_use_interpolation,
        )

    def test_smooth_sparse(self):
        sparse_values = np.concatenate([np.ones(25), np.zeros(25)])
        coordinates = np.indices((10, 5)).reshape(2, -1).T
        smoothed = self.mock_target.smooth_sparse(sparse_values=sparse_values, coordinates=coordinates)
        self.assertTupleEqual((50,), smoothed.shape)
        np.testing.assert_array_almost_equal(1.0, smoothed[:15], decimal=8)
        np.testing.assert_array_almost_equal(0.0, smoothed[-15:], decimal=8)
        smoothed_part = smoothed[15:35]
        for i_value, value in enumerate([0.94551132, 0.70130997, 0.29869003, 0.05448868]):
            np.testing.assert_array_almost_equal(value, smoothed_part[i_value * 5 : (i_value + 1) * 5], decimal=8)

    @given(strategies.floats(min_value=0, max_value=1e12, allow_subnormal=False))
    @settings(deadline=timedelta(seconds=1))
    def test_smooth_sparse_preserves_values(self, fill_value):
        mock_sparse = np.full(10, fill_value=fill_value)
        mock_coordinates = np.indices((2, 5)).reshape(2, -1).T
        smoothed = self.mock_target.smooth_sparse(sparse_values=mock_sparse, coordinates=mock_coordinates)
        np.testing.assert_allclose(mock_sparse, smoothed, rtol=1e-8)

    @given(strategies.floats(min_value=0, max_value=1e12, allow_subnormal=False))
    @settings(deadline=timedelta(seconds=1))
    def test_smooth_sparse_non_square(self, fill_value):
        mock_sparse = np.full(20, fill_value=fill_value)
        mock_coordinates = np.array([[i // 3, i % 3] for i in range(20)])
        smoothed = self.mock_target.smooth_sparse(sparse_values=mock_sparse, coordinates=mock_coordinates)
        np.testing.assert_allclose(mock_sparse, smoothed, rtol=1e-8)

    def test_smooth_sparse_casts_when_integer(self):
        mock_sparse = np.full(10, fill_value=10)
        mock_coordinates = np.array([[i // 5, i % 5] for i in range(10)])
        smoothed = self.mock_target.smooth_sparse(sparse_values=mock_sparse, coordinates=mock_coordinates)
        self.assertEqual(np.float64, smoothed.dtype)
        np.testing.assert_allclose(mock_sparse, smoothed, rtol=1e-8)

    def test_smooth_dense(self):
        mock_image_2d = np.ones((15, 5))
        mock_image_2d[:5, :] = np.nan
        mock_image_2d[-5:, :] = 0
        smoothed = self.mock_target.smooth_dense(image_2d=mock_image_2d)

        expected_nan = np.zeros((15, 5))
        expected_nan[:5, :] = 1
        np.testing.assert_array_equal(expected_nan, np.isnan(smoothed))

        np.testing.assert_array_equal(np.ones((3, 5)), smoothed[5:8, :])
        np.testing.assert_array_equal(np.zeros((3, 5)), smoothed[-3:, :])
        smoothed_part = smoothed[8:12, :]
        np.testing.assert_allclose(0, smoothed_part.T - smoothed_part[:, 0], atol=1e-8)
        expected = [0.94551132, 0.70130997, 0.29869003, 0.05448868]
        np.testing.assert_array_almost_equal(expected, smoothed_part[:, 0], decimal=8)

    def test_smooth_dense_preserves_values(self):
        mock_image = np.full((5, 10), fill_value=2.5)
        smoothed = self.mock_target.smooth_dense(image_2d=mock_image)
        np.testing.assert_allclose(mock_image, smoothed, rtol=1e-8)

    def test_smooth_dense_when_use_interpolation(self):
        self.mock_use_interpolation = True
        mock_image = np.full((9, 5), fill_value=3.0)
        mock_image[4, 2] = np.nan
        smoothed = self.mock_target.smooth_dense(image_2d=mock_image)
        self.assertEqual(0, np.sum(np.isnan(smoothed)))
        self.assertAlmostEqual(3, smoothed[4, 2], places=6)

    def test_smooth_sparse_multi_channel(self):
        sparse_values = np.stack([np.full(10, fill_value=10.0), np.full(10, fill_value=20.0)], axis=1)
        coordinates = np.array([[i // 3, i % 3] for i in range(10)])
        smoothed = self.mock_target.smooth_sparse_multi_channel(sparse_values, coordinates)
        np.testing.assert_allclose(sparse_values, smoothed, rtol=1e-8)

    def test_smooth_dense_multi_channel(self):
        image_2d = np.stack(
            [np.full((5, 10), fill_value=10.0), np.full((5, 10), fill_value=20.0)],
            axis=2,
        )
        smoothed = self.mock_target.smooth_dense_multi_channel(image_2d)
        np.testing.assert_allclose(image_2d, smoothed, rtol=1e-8)

    def test_gaussian_kernel(self):
        self.mock_kernel_size = 3
        self.mock_kernel_std = 1.0
        expected_arr = np.array(
            [
                [0.075114, 0.123841, 0.075114],
                [0.123841, 0.20418, 0.123841],
                [0.075114, 0.123841, 0.075114],
            ]
        )
        np.testing.assert_allclose(expected_arr, self.mock_target.gaussian_kernel, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
