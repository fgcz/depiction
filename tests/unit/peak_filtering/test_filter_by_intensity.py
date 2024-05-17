import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from ionmapper.peak_filtering.filter_by_intensity import FilterByIntensity


class TestFilterByIntensity(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_intensity_threshold = 25
        self.mock_normalization = None

    @cached_property
    def mock_filter(self) -> FilterByIntensity:
        return FilterByIntensity(
            min_intensity=self.mock_intensity_threshold,
            normalization=self.mock_normalization,
        )

    def test_filter_index_peaks_when_no_normalization(self) -> None:
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_idx_arr = np.array([2, 5, 45, 48])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )
        np.testing.assert_array_equal(np.array([45, 48]), indices)

    def test_filter_peaks_when_no_normalization(self) -> None:
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_mz_arr = np.array([10, 20, 30, 40])
        peak_int_arr = spectrum_int_arr[[2, 5, 45, 48]]

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([30, 40]), result_mz)

    def test_filter_index_peaks_when_tic_normalization(self) -> None:
        self.mock_normalization = "tic"
        self.mock_intensity_threshold = 0.02
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_idx_arr = np.array([2, 5, 45, 48])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )
        np.testing.assert_array_equal(np.array([45, 48]), indices)

    def test_filter_peaks_when_tic_normalization(self) -> None:
        self.mock_normalization = "tic"
        self.mock_intensity_threshold = 0.02
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_mz_arr = np.array([10, 20, 30, 40])
        peak_int_arr = spectrum_int_arr[[2, 5, 45, 48]]

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([30, 40]), result_mz)

    def test_filter_index_peaks_when_median_normalization(self) -> None:
        self.mock_normalization = "median"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_idx_arr = np.array([2, 5, 45, 48])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )
        np.testing.assert_array_equal(np.array([45, 48]), indices)

    def test_filter_peaks_when_median_normalization(self) -> None:
        self.mock_normalization = "median"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_mz_arr = np.array([10, 20, 30, 40])
        peak_int_arr = spectrum_int_arr[[2, 5, 45, 48]]

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([30, 40]), result_mz)

    def test_filter_index_peaks_when_median_normalization_and_zero(self) -> None:
        self.mock_normalization = "median"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.array([0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0])
        peak_idx_arr = np.array([2, 5, 8])
        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )
        np.testing.assert_array_equal(np.array([2, 5, 8]), indices)

    def test_filter_peaks_when_median_normalization_and_zero(self) -> None:
        self.mock_normalization = "median"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.array([0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0])
        peak_mz_arr = np.array([10, 20, 30])
        peak_int_arr = np.array([10, 10, 10])
        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([10, 20, 30]), result_mz)
        np.testing.assert_array_equal(np.array([10, 10, 10]), result_int)

    def test_filter_peaks_when_vec_norm_normalization(self) -> None:
        self.mock_normalization = "vec_norm"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.linspace(10, 100, 50)
        peak_mz_arr = np.array([10, 20, 30, 40])
        peak_int_arr = spectrum_int_arr[[2, 5, 45, 48]]

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([30, 40]), result_mz)

    def test_filter_peaks_when_vec_norm_normalization_and_zero(self) -> None:
        self.mock_normalization = "vec_norm"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.array([0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0])
        peak_mz_arr = np.array([10, 20, 30])
        peak_int_arr = np.array([10, 10, 10])

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([10, 20, 30]), result_mz)
        np.testing.assert_array_equal(np.array([10, 10, 10]), result_int)

    def test_filter_peaks_when_vec_norm_normalization_and_all_zero(self) -> None:
        self.mock_normalization = "vec_norm"
        self.mock_intensity_threshold = 0.5
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        spectrum_int_arr = np.zeros(50)
        peak_mz_arr = np.array([10, 20, 30])
        peak_int_arr = np.array([0, 0, 0])

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        np.testing.assert_array_equal(np.array([]), result_mz)
        np.testing.assert_array_equal(np.array([]), result_int)

    def test_filter_index_peaks_when_empty_input(self) -> None:
        mz_arr = np.array([])
        int_arr = np.array([])
        peak_idx_arr = np.array([])

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal(np.array([]), indices)

    def test_filter_peaks_when_empty_input(self) -> None:
        mz_arr = np.array([])
        int_arr = np.array([])
        peak_mz_arr = np.array([])
        peak_int_arr = np.array([])

        result_mz, result_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )

        np.testing.assert_array_equal(np.array([]), result_mz)
        np.testing.assert_array_equal(np.array([]), result_int)

    def test_filter_index_peaks_when_empty_result(self) -> None:
        mz_arr = np.array([2])
        int_arr = np.array([1])
        peak_idx_arr = np.array([0])
        self.mock_intensity_threshold = 100

        indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal(np.array([]), indices)

    def test_filter_peaks_when_empty_result(self) -> None:
        mz_arr = np.array([2])
        int_arr = np.array([1])
        peak_mz_arr = np.array([2])
        peak_int_arr = np.array([1])
        self.mock_intensity_threshold = 100

        peak_mz, peak_int = self.mock_filter.filter_peaks(
            spectrum_mz_arr=mz_arr,
            spectrum_int_arr=int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )

        np.testing.assert_array_equal(np.array([]), peak_mz)
        np.testing.assert_array_equal(np.array([]), peak_int)


if __name__ == "__main__":
    unittest.main()
