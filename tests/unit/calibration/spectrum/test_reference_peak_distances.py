import unittest

import numpy as np

from ionmapper.calibration.spectrum.reference_peak_distances import ReferencePeakDistances


class TestReferencePeakDistances(unittest.TestCase):
    def test_get_distances_max_peak_in_window_when_mz(self) -> None:
        peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0])
        peak_int_arr = np.array([5.0, 5, 2, 2, 5, 5.1, 5])
        ref_mz_arr = np.array([90, 100.0, 101.5, 102.4])
        distances = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
            ref_mz_arr=ref_mz_arr,
            max_distance=0.9,
            max_distance_unit="mz",
        )
        expected = np.array([np.nan, 0.0, 0.5, 0.1])
        np.testing.assert_array_almost_equal(expected, distances, decimal=8)

    def test_get_distances_max_peak_in_window_when_ppm(self) -> None:
        peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0])
        peak_int_arr = np.array([5.0, 5, 2, 2, 5, 5.1, 5])
        ref_mz_arr = np.array([90, 100.0, 101.5, 102.4])
        distances = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
            ref_mz_arr=ref_mz_arr,
            max_distance=9e3,
            max_distance_unit="ppm",
        )
        expected = np.array([np.nan, 0.0, 0.5, 0.1])
        np.testing.assert_array_almost_equal(expected, distances, decimal=8)

    def test_get_distances_max_peak_in_window_when_invalid(self) -> None:
        unit = "badunit"
        mock_peak_mz_arr = np.array([10.0, 20, 30])
        mock_peak_int_arr = np.array([1, 2, 3])
        mock_ref_arr = np.array([10.0])
        with self.assertRaises(ValueError) as error:
            ReferencePeakDistances.get_distances_max_peak_in_window(
                peak_mz_arr=mock_peak_mz_arr,
                peak_int_arr=mock_peak_int_arr,
                ref_mz_arr=mock_ref_arr,
                max_distance=0.3,
                max_distance_unit=unit,
            )
        self.assertIn("badunit", str(error.exception))

    def test_get_distances_nearest_when_mz(self) -> None:
        peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0, 106.0])
        ref_mz_arr = np.array([100.6, 102.0, 103.9, 105.5])
        distances = ReferencePeakDistances.get_distances_nearest(
            peak_mz_arr=peak_mz_arr,
            ref_mz_arr=ref_mz_arr,
            max_distance=0.3,
            max_distance_unit="mz",
        )
        expected = np.array([-0.1, 0, 0.1, np.nan])
        np.testing.assert_array_almost_equal(expected, distances, decimal=8)

    def test_get_distances_nearest_when_ppm(self) -> None:
        peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0, 106.0])
        ref_mz_arr = np.array([100.6, 102.0, 103.9, 105.5])
        distances = ReferencePeakDistances.get_distances_nearest(
            peak_mz_arr=peak_mz_arr,
            ref_mz_arr=ref_mz_arr,
            max_distance=3e3,
            max_distance_unit="ppm",
        )
        expected = np.array([-0.1, 0, 0.1, np.nan])
        np.testing.assert_array_almost_equal(expected, distances, decimal=8)

    def test_get_distances_nearest_when_invalid(self) -> None:
        unit = "badunit"
        mock_peak_mz_arr = np.array([10.0, 20, 30])
        mock_ref_arr = np.array([10.0])
        with self.assertRaises(ValueError) as error:
            ReferencePeakDistances.get_distances_nearest(
                peak_mz_arr=mock_peak_mz_arr,
                ref_mz_arr=mock_ref_arr,
                max_distance=0.3,
                max_distance_unit=unit,
            )
        self.assertIn("badunit", str(error.exception))


if __name__ == "__main__":
    unittest.main()
