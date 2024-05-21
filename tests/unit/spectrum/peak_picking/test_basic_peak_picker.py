import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np

from ionplotter.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker


class TestBasicPeakPicker(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_smooth_sigma = 0.1
        self.mock_min_prominence = 2.0
        self.mock_min_distance = None
        self.mock_min_distance_unit = "index"

        self.mock_mz_arr = np.arange(10, 20, 0.1)
        self.mock_int_arr = np.full(self.mock_mz_arr.shape, 0.5)
        self.mock_int_arr[::7] = np.linspace(10, 15, len(self.mock_int_arr[::7]))

    @cached_property
    def mock_picker(self) -> BasicPeakPicker:
        return BasicPeakPicker(
            smooth_sigma=self.mock_smooth_sigma,
            min_prominence=self.mock_min_prominence,
            min_distance=self.mock_min_distance,
            min_distance_unit=self.mock_min_distance_unit,
        )

    @patch("scipy.ndimage.gaussian_filter1d")
    def test_get_smoothed_intensities(self, mock_gaussian_filter1d) -> None:
        self.mock_smooth_sigma = 5
        mock_int_arr = MagicMock(name="mock_int_arr", spec=[])
        mock_mz_arr = np.array([5, 7, 9, 11, 11.5, 15])
        int_arr_smooth = self.mock_picker.get_smoothed_intensities(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        self.assertEqual(mock_gaussian_filter1d.return_value, int_arr_smooth)
        mock_gaussian_filter1d.assert_called_once_with(mock_int_arr, sigma=2.5)

    def test_get_smoothed_intensities_when_sigma_is_none(self) -> None:
        self.mock_smooth_sigma = None
        mock_int_arr = MagicMock(name="mock_int_arr", spec=[])
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        smoothed = self.mock_picker.get_smoothed_intensities(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        self.assertEqual(mock_int_arr, smoothed)

    def test_pick_peaks_index(self) -> None:
        indices = self.mock_picker.pick_peaks_index(self.mock_mz_arr, self.mock_int_arr)
        np.testing.assert_array_equal(
            np.array([7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91]),
            indices,
        )

    def test_pick_peaks_index_when_min_distance(self) -> None:
        self.mock_min_distance = 1.0
        self.mock_min_distance_unit = "mz"
        indices = self.mock_picker.pick_peaks_index(self.mock_mz_arr, self.mock_int_arr)
        np.testing.assert_array_equal([14, 28, 42, 56, 70, 84], indices)

    @patch.object(BasicPeakPicker, "pick_peaks_index")
    def test_pick_peaks_mz(self, method_pick_peaks_index) -> None:
        mock_mz_array = np.array([5.0, 6, 7, 8, 9, 10])
        mock_int_array = MagicMock(name="mock_int_array", spec=[])
        method_pick_peaks_index.return_value = np.array([1, 3, 5])

        peaks_mz = self.mock_picker.pick_peaks_mz(
            mock_mz_array,
            mock_int_array,
        )

        np.testing.assert_array_equal(np.array([6.0, 8, 10]), peaks_mz)

    def test_clone(self) -> None:
        copy = self.mock_picker.clone()
        self.assertIsNot(self.mock_picker, copy)
        self.assertEqual(self.mock_picker.smooth_sigma, copy.smooth_sigma)
        self.assertEqual(self.mock_picker.min_prominence, copy.min_prominence)

    def test_get_min_distance_indices_when_none(self) -> None:
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        self.assertIsNone(
            BasicPeakPicker.get_min_distance_indices(min_distance=None, min_distance_unit=None, mz_arr=mock_mz_arr)
        )

    def test_get_min_distance_indices_when_unit_index(self) -> None:
        mock_min_distance = 3
        mock_min_distance_unit = "index"
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        self.assertEqual(
            3,
            BasicPeakPicker.get_min_distance_indices(
                min_distance=mock_min_distance, min_distance_unit=mock_min_distance_unit, mz_arr=mock_mz_arr
            ),
        )

    def test_get_min_distance_indices_when_unit_mz(self) -> None:
        mock_min_distance = 0.5
        mock_min_distance_unit = "mz"
        self.assertEqual(
            5,
            BasicPeakPicker.get_min_distance_indices(
                min_distance=mock_min_distance, min_distance_unit=mock_min_distance_unit, mz_arr=self.mock_mz_arr
            ),
        )


if __name__ == "__main__":
    unittest.main()
