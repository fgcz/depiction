import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np
from logot import Logot, logged
from logot.loguru import LoguruCapturer

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

    def test_interpolate_max_mz_and_intensity_when_success_and_exact(self) -> None:
        """In this very simple case, interpolation should return the same value as the intensity on both sides is
        symmetric."""
        local_max_index = 2
        mz_arr = np.array([1.0, 2, 3, 4, 5])
        int_arr = np.array([0.0, 0, 10, 0, 0])

        interpolated_mz, interpolated_int = self.basic_interpolated_peak_picker._interpolate_max_mz_and_intensity(
            local_max_index, mz_arr, int_arr
        )

        self.assertAlmostEqual(3, interpolated_mz)
        self.assertAlmostEqual(10, interpolated_int)

    def test_interpolate_max_mz_and_intensity_when_success_and_not_exact(self) -> None:
        local_max_index = 2
        mz_arr = np.array([1.0, 2, 3, 4, 5])
        int_arr = np.array([0.0, 0, 10, 5, 0])

        interpolated_mz, interpolated_int = self.basic_interpolated_peak_picker._interpolate_max_mz_and_intensity(
            local_max_index, mz_arr, int_arr
        )

        self.assertAlmostEqual(3.16667, interpolated_mz, places=5)
        self.assertAlmostEqual(10.20833, interpolated_int, places=5)

    def test_interpolate_max_mz_and_intensity_when_failure(self) -> None:
        local_max_index = 2
        mz_arr = np.array([1.0, 2, 3, 4, 5])
        int_arr = np.array([0.0, 10, 10, 10, 0])

        with Logot().capturing(capturer=LoguruCapturer) as logot:
            interpolated_mz, interpolated_int = self.basic_interpolated_peak_picker._interpolate_max_mz_and_intensity(
                local_max_index, mz_arr, int_arr
            )
        self.assertIsNone(interpolated_mz)
        self.assertIsNone(interpolated_int)
        logot.assert_logged(logged.warning("Error: %d roots found for local maximum at index 2; %s"))

    # TODO test interpolate when there are 2 roots because of numerical issues, 3 is apparently also possible

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
