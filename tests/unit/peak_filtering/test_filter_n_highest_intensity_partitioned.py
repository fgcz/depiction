import unittest
from functools import cached_property
from unittest.mock import MagicMock, call, ANY, patch

import numpy as np

from ionplotter.peak_filtering.filter_n_highest_intensity_partitioned import (
    FilterNHighestIntensityPartitioned,
)


class TestfilterNHighestIntensityPartitioned(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_max_count = 5
        self.mock_n_partitions = 2
        self.mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")

    @cached_property
    def mock_filter(self) -> FilterNHighestIntensityPartitioned:
        return FilterNHighestIntensityPartitioned(max_count=self.mock_max_count, n_partitions=self.mock_n_partitions)

    @patch("ionplotter.peak_filtering.filter_n_highest_intensity_partitioned.FilterNHighestIntensity")
    def test_filter_index_peaks(self, mock_filter_n_highest_intensity) -> None:
        mock_filter_n_highest_intensity.return_value.filter_index_peaks.side_effect = [
            np.array([10, 20, 30]),
            np.array([200, 210]),
        ]
        spectrum_mz_arr = np.linspace(5, 210, 250)
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr")
        peak_idx_arr = np.array([10, 20, 30, 40, 190, 195, 197, 200, 210])

        peak_indices = self.mock_filter.filter_index_peaks(
            spectrum_mz_arr=spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        np.testing.assert_array_equal([10, 20, 30, 200, 210], peak_indices)
        self.assertListEqual(
            [
                call(max_count=2),
                call().filter_index_peaks(
                    spectrum_mz_arr=ANY,
                    spectrum_int_arr=mock_spectrum_int_arr,
                    peak_idx_arr=ANY,
                ),
                call().filter_index_peaks(
                    spectrum_mz_arr=ANY,
                    spectrum_int_arr=mock_spectrum_int_arr,
                    peak_idx_arr=ANY,
                ),
            ],
            mock_filter_n_highest_intensity.mock_calls,
        )
        np.testing.assert_array_equal(
            [10, 20, 30, 40],
            mock_filter_n_highest_intensity.return_value.filter_index_peaks.call_args_list[0][1]["peak_idx_arr"],
        )
        np.testing.assert_array_equal(
            [190, 195, 197, 200, 210],
            mock_filter_n_highest_intensity.return_value.filter_index_peaks.call_args_list[1][1]["peak_idx_arr"],
        )

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


if __name__ == "__main__":
    unittest.main()
