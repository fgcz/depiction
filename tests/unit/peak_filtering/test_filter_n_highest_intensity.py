import unittest
from functools import cached_property
from unittest.mock import MagicMock, patch

import numpy as np

from ionmapper.peak_filtering.filter_n_highest_intensity import (
    FilterNHighestIntensity,
)


class TestfilterNHighestIntensity(unittest.TestCase):
    def setUp(self):
        self.mock_max_count = 3
        self.mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")

    @cached_property
    def mock_filter(self) -> FilterNHighestIntensity:
        return FilterNHighestIntensity(max_count=self.mock_max_count)

    def test_filter_index_peaks_when_actually_filtering(self):
        spectrum_int_arr = np.array([1, 2, 0, 0, 3, 4, 0, 0, 7, 8, 1])
        peak_idx_array = np.array([0, 1, 4, 5, 8, 9, 10])
        remaining_indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=self.mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_array,
        )
        np.testing.assert_array_equal([5, 8, 9], remaining_indices)

    def test_filter_peaks_when_actually_filtering(self):
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr", spec=[])
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr", spec=[])
        peak_mz_arr = np.array([1, 2, 3, 5, 8, 13])
        peak_int_arr = np.array([1, 2, 0, 0, 8, 7])
        mz_arr, int_arr = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal([2, 8, 13], mz_arr)
        np.testing.assert_array_equal([2, 8, 7], int_arr)

    def test_filter_index_peaks_when_exactly_all_peaks_are_valid(self):
        self.mock_max_count = 4
        spectrum_int_arr = np.array([1, 2, 0, 3, 4])
        peak_idx_array = np.array([0, 1, 3, 4])
        remaining_indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=self.mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_array,
        )
        np.testing.assert_array_equal([0, 1, 3, 4], remaining_indices)

    def test_filter_peaks_when_exactly_all_peaks_are_valid(self):
        self.mock_max_count = 4
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr", spec=[])
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr", spec=[])
        peak_mz_arr = np.array([1, 2, 3, 4])
        peak_int_arr = np.array([1, 2, 3, 4])
        mz_arr, int_arr = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal([1, 2, 3, 4], mz_arr)
        np.testing.assert_array_equal([1, 2, 3, 4], int_arr)

    def test_filter_index_peaks_when_not_enough(self):
        self.mock_max_count = 5
        spectrum_int_arr = np.array([1, 2, 0, 3, 4])
        peak_idx_array = np.array([0, 1, 3, 4])
        remaining_indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=self.mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_array,
        )
        np.testing.assert_array_equal([0, 1, 3, 4], remaining_indices)

    def test_filter_peaks_when_not_enough(self):
        self.mock_max_count = 5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr", spec=[])
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr", spec=[])
        peak_mz_arr = np.array([1, 2, 3, 4])
        peak_int_arr = np.array([1, 2, 3, 4])
        mz_arr, int_arr = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal([1, 2, 3, 4], mz_arr)
        np.testing.assert_array_equal([1, 2, 3, 4], int_arr)

    def test_filter_index_peaks_when_empty_input(self):
        mz_arr = np.array([])
        int_arr = np.array([])
        peak_idx_arr = np.array([])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal(np.array([]), indices)

    def test_filter_peaks_when_empty_input(self):
        mz_arr = np.array([])
        int_arr = np.array([])
        peak_mz_arr = np.array([])
        peak_int_arr = np.array([])

        mz_arr, int_arr = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )

        np.testing.assert_array_equal(np.array([]), mz_arr)
        np.testing.assert_array_equal(np.array([]), int_arr)


if __name__ == "__main__":
    unittest.main()
