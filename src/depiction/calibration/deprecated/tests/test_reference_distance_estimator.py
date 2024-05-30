import unittest
from functools import cached_property

import numpy as np

from depiction.calibration.deprecated.reference_distance_estimator import (
    ReferenceDistanceEstimator,
)


class TestReferenceDistanceEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_reference_mz = np.array([50.0, 75, 100, 110])
        self.mock_n_candidates = 3

    @cached_property
    def mock_estimator(self) -> ReferenceDistanceEstimator:
        return ReferenceDistanceEstimator(
            reference_mz=self.mock_reference_mz,
            n_candidates=self.mock_n_candidates,
        )

    def test_n_targets(self) -> None:
        self.assertEqual(4, self.mock_estimator.n_targets)

    def test_n_candidates(self) -> None:
        self.assertEqual(3, self.mock_estimator.n_candidates)

    def test_reference_mz(self) -> None:
        np.testing.assert_array_equal(
            np.array([50.0, 75, 100, 110]),
            self.mock_estimator.reference_mz,
        )

    def test_closest_index(self) -> None:
        self.assertEqual(1, self.mock_estimator.closest_index)

    def test_compute_distances_for_peaks(self) -> None:
        mz_peaks = np.array([50.0, 75, 100, 110, 120, 130, 140])
        distances, closest_indices = self.mock_estimator.compute_distances_for_peaks(mz_peaks)
        np.testing.assert_array_equal(
            np.array(
                [
                    [-np.inf, 0, 25],
                    [-25, 0, 25],
                    [-25, 0, 10],
                    [-10, 0, 10],
                ]
            ),
            distances,
        )
        np.testing.assert_array_equal(
            np.array([0, 1, 2, 3]),
            closest_indices,
        )

    def test_compute_distances_for_peaks_when_left_out_of_bounds(self) -> None:
        self.mock_reference_mz = np.array([9.0, 10.0, 20.0, 20.5])
        mz_peaks = np.array([9.0, 18.0, 18.5, 19.0, 21.0, 25.0])
        distances, closest_indices = self.mock_estimator.compute_distances_for_peaks(mz_peaks)
        np.testing.assert_array_equal([0, 0, 3, 4], closest_indices)
        np.testing.assert_array_equal([-np.inf, 0, 9], distances[0])
        np.testing.assert_array_equal([-np.inf, -1, 8], distances[1])
        np.testing.assert_array_equal([-1.5, -1.0, 1], distances[2])
        np.testing.assert_array_equal([-1.5, 0.5, 4.5], distances[3])

    def test_compute_distances_for_peaks_when_right_out_of_bounds(self) -> None:
        self.mock_reference_mz = np.array([10.0, 20.0, 29.0, 30.0])
        mz_peaks = np.array([8.0, 9.0, 19, 28.0, 28.5, 29.0])
        distances, closest_indices = self.mock_estimator.compute_distances_for_peaks(mz_peaks)
        np.testing.assert_array_equal([1, 2, 5, 5], closest_indices)
        np.testing.assert_array_equal([-2.0, -1, 9], distances[0])
        np.testing.assert_array_equal([-11.0, -1, 8], distances[1])
        np.testing.assert_array_equal([-0.5, 0, np.inf], distances[2])
        np.testing.assert_array_equal([-1.5, -1, np.inf], distances[3])


if __name__ == "__main__":
    unittest.main()
