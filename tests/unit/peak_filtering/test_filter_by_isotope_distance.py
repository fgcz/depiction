import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from ionmapper.peak_filtering.filter_by_isotope_distance import (
    FilterByIsotopeDistance,
)


class TestFilterByIsotopeDistance(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_min_distance = 0.9
        self.mock_max_distance = 1.1
        self.mock_far_distance = 3.0

    @cached_property
    def mock_filter(self) -> FilterByIsotopeDistance:
        return FilterByIsotopeDistance(
            min_distance=self.mock_min_distance,
            max_distance=self.mock_max_distance,
            far_distance=self.mock_far_distance,
        )

    def test(self) -> None:
        spectrum_mz_arr = np.arange(0, 200, 0.1)
        peak_idx_arr = np.array([10, 20, 30, 70, 75, 90, 100, 110, 150, 155])
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr")
        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )
        np.testing.assert_array_equal(np.array([10, 20, 30, 100, 110]), indices)

    def test_when_empty_input(self) -> None:
        mz_arr = np.array([])
        int_arr = np.array([])
        peak_idx_arr = np.array([])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal(np.array([]), indices)

    def test_when_empty_result(self) -> None:
        mz_arr = np.array([2, 4])
        int_arr = np.array([1, 1])
        peak_idx_arr = np.array([0, 1])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal(np.array([]), indices)


if __name__ == "__main__":
    unittest.main()
