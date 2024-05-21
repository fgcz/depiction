import unittest

import numpy as np

from ionplotter.spatial_smoothing import SpatialSmoothing


class TestSpatialSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinates = np.array([[1, 2], [1, 3], [1, 4], [2, 2], [2, 4]])
        self.full_values_spatial = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 0, 5]])
        self.values_flat = np.array([1, 2, 3, 4, 5])
        self.values_spatial = np.array([[1, 2, 3], [4, 0, 5]])

    def test_smooth_values(self) -> None:
        smoothing = SpatialSmoothing(sigma=1000)
        smoothed_values = smoothing.smooth_sparse(sparse_values=np.ones(5), coordinates=self.coordinates)
        np.testing.assert_allclose(5 / 6, smoothed_values)
        self.assertTupleEqual((5,), smoothed_values.shape)

    def test_flat_to_grid_when_2d(self) -> None:
        values_spatial = SpatialSmoothing(sigma=1.0).flat_to_grid(
            sparse_values=self.values_flat, coordinates=self.coordinates, background_value=0
        )
        np.testing.assert_array_equal(self.values_spatial, values_spatial)

    def test_grid_to_flat_when_2d(self) -> None:
        values_flat = SpatialSmoothing.grid_to_flat(values_grid=self.values_spatial, coordinates=self.coordinates)
        np.testing.assert_array_equal(self.values_flat, values_flat)

    def test_fill_background_when_nearest(self) -> None:
        for b in [0, np.nan]:
            with self.subTest(bg=b):
                arr = np.array(
                    [
                        [b, b, b, b, b, b],
                        [b, 4, 4, 4, b, b],
                        [b, 5, 5, 5, b, b],
                        [b, 6, 7, 8, b, b],
                        [b, b, b, b, b, b],
                    ]
                )
                smoothing = SpatialSmoothing(sigma=1.0, background_fill_mode="nearest", background_value=b)
                filled_arr = smoothing.fill_background(values_spatial=arr)
                expected_arr = np.array(
                    [
                        [4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 4, 4],
                        [5, 5, 5, 5, 5, 5],
                        [6, 6, 7, 8, 8, 8],
                        [6, 6, 7, 8, 8, 8],
                    ]
                )
                np.testing.assert_array_equal(expected_arr, filled_arr)


if __name__ == "__main__":
    unittest.main()
