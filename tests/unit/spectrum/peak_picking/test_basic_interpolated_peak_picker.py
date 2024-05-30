import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np

from depiction.spectrum.peak_picking import BasicPeakPicker, BasicInterpolatedPeakPicker


class TestBasicInterpolatedPeakPicker(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_min_prominence = MagicMock(name="mock_min_prominence")
        self.mock_min_distance = MagicMock(name="mock_min_distance")
        self.mock_min_distance_unit = MagicMock(name="mock_min_distance_unit")
        self.mock_peak_filtering = MagicMock(name="mock_peak_filtering")

    @cached_property
    def basic_interpolated_peak_picker(self) -> BasicInterpolatedPeakPicker:
        return BasicInterpolatedPeakPicker(
            min_prominence=self.mock_min_prominence,
            min_distance=self.mock_min_distance,
            min_distance_unit=self.mock_min_distance_unit,
            peak_filtering=self.mock_peak_filtering,
        )

    def test_interpolate_max_mz_and_intensity_when_success(self) -> None:
        pass

    def test_interpolate_max_mz_and_intensity_when_failure(self) -> None:
        pass

    @patch.object(BasicPeakPicker, "get_min_distance_indices")
    def test_find_local_maxima_indices(self, mock_get_min_distance_indices) -> None:
        int_arr = np.array([0, 10, 0, 5, 0, 5, 0, 10, 0])
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        mock_get_min_distance_indices.return_value = 3
        self.mock_min_prominence = 1.0

        local_maxima = self.basic_interpolated_peak_picker._find_local_maxima_indices(mock_mz_arr, int_arr)
        np.testing.assert_array_equal([1, 7], local_maxima)

        mock_get_min_distance_indices.assert_called_once_with(
            min_distance=self.mock_min_distance, min_distance_unit=self.mock_min_distance_unit, mz_arr=mock_mz_arr
        )


if __name__ == "__main__":
    unittest.main()
