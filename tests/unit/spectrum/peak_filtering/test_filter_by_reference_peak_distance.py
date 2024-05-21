import unittest
from functools import cached_property

import numpy as np

from ionplotter.spectrum.peak_filtering import (
    FilterByReferencePeakDistance,
)


class TestFilterByReferencePeakDistance(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_max_distance = 3.0
        self.mock_reference_mz = np.array([100.0, 150.0, 2000.0])
        self.mock_baseline_value = 0.2
        self.mock_peak_value = 10.0
        self.mock_peak_mz = np.array(
            [
                # isotopic peaks of ref 1 (idx 0-2)
                99.5,
                100.5,
                101.5,
                # unrelated peak (idx 4)
                105.0,
                # isotopic peaks of ref 2 (idx 5-7)
                150.7,
                151.7,
                151.9,
                # unrelated peak (idx 8)
                200.5,
                # isotopic peaks of ref 3 (idx 9-11)
                2000.1,
                2001.1,
                2002.1,
            ]
        )
        self.mock_actual_peak_mz = np.setdiff1d(self.mock_peak_mz, [105.0, 200.5])

    @cached_property
    def mock_data(self) -> dict[str, np.ndarray]:
        # this is a remainder of a previous iteration, it could probably be simplified
        mz_range_min = np.min(self.mock_peak_mz)
        mz_range_max = np.max(self.mock_peak_mz)

        spectrum_idx_arr = np.unique(
            np.concatenate(
                [
                    np.linspace(mz_range_min, mz_range_max, 200),
                    [mz for mz in self.mock_peak_mz if mz_range_min < mz < mz_range_max],
                ]
            )
        )
        peak_idx_arr = np.where(np.isin(spectrum_idx_arr, self.mock_peak_mz))[0]
        spectrum_int_arr = np.full_like(spectrum_idx_arr, self.mock_baseline_value)
        spectrum_int_arr[peak_idx_arr] = self.mock_peak_value
        return {
            "spectrum_mz_arr": spectrum_idx_arr,
            "spectrum_int_arr": spectrum_int_arr,
            "peak_idx_arr": peak_idx_arr,
        }

    @cached_property
    def mock_filter(self) -> FilterByReferencePeakDistance:
        return FilterByReferencePeakDistance(max_distance=self.mock_max_distance, reference_mz=self.mock_reference_mz)

    def test(self) -> None:
        candidates = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=self.mock_data["spectrum_mz_arr"],
            spectrum_int_arr=self.mock_data["spectrum_int_arr"],
            peak_idx_arr=self.mock_data["peak_idx_arr"],
        )
        np.testing.assert_array_equal(
            np.where(np.isin(self.mock_data["spectrum_mz_arr"], self.mock_actual_peak_mz))[0],
            candidates,
        )

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


if __name__ == "__main__":
    unittest.main()
