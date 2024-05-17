import unittest
from functools import cached_property
from unittest.mock import MagicMock, patch

import numpy as np

from ionmapper.peak_picking.basic_peak_picker import BasicPeakPicker
from ionmapper.calibration.deprecated.calibrate_spectrum import CalibrateSpectrum
from ionmapper.calibration.models.linear_model import LinearModel
from ionmapper.calibration.models.polynomial_model import PolynomialModel
from ionmapper.calibration.deprecated.reference_distance_estimator import (
    ReferenceDistanceEstimator,
)
from typing import NoReturn


class TestCalibrateSpectrum(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_reference_mz = np.array([100, 200, 300])
        self.mock_detect_smooth_sigma = MagicMock(name="mock_detect_smooth_sigma")
        self.mock_detect_prominence = MagicMock(name="mock_detect_prominence")
        self.mock_n_candidates = MagicMock(name="mock_n_candidates")
        self.mock_model_type = "linear"
        self.mock_distance_limit = 2.0
        self.mock_basic_peak_picker = MagicMock(name="mock_basic_peak_picker", spec=BasicPeakPicker)
        self.mock_reference_distance_estimator = MagicMock(
            name="mock_reference_distance_estimator",
            spec=ReferenceDistanceEstimator,
        )
        self.mock_mz_arr = MagicMock(name="mock_mz_arr")
        self.mock_int_arr = MagicMock(name="mock_int_arr")
        self.mock_problem_mz = MagicMock(name="mock_problem_mz")
        self.mock_problem_dist = MagicMock(name="mock_problem_dist")

    @cached_property
    @patch("ionmapper.calibration.deprecated.calibrate_spectrum.ReferenceDistanceEstimator")
    @patch("ionmapper.calibration.deprecated.calibrate_spectrum.BasicPeakPicker")
    def mock_compute(self, mock_basic_peak_picker, mock_reference_distance_estimator) -> CalibrateSpectrum:
        mock_basic_peak_picker.return_value = self.mock_basic_peak_picker
        mock_reference_distance_estimator.return_value = self.mock_reference_distance_estimator
        return CalibrateSpectrum(
            reference_mz=self.mock_reference_mz,
            peak_picker=self.mock_basic_peak_picker,
            n_candidates=self.mock_n_candidates,
            model_type=self.mock_model_type,
            distance_limit=self.mock_distance_limit,
        )

    @patch.object(LinearModel, "zero")
    @patch.object(CalibrateSpectrum, "_get_mz_distance_pairs")
    def test_calibrate_spectrum_when_zero_points(self, mock_get_mz_distance_pairs, mock_zero) -> None:
        self.mock_problem_mz.__len__.return_value = 0
        mock_get_mz_distance_pairs.return_value = (
            self.mock_problem_mz,
            self.mock_problem_dist,
        )
        model = self.mock_compute.calibrate_spectrum(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_get_mz_distance_pairs.assert_called_once_with(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_zero.assert_called_once_with()
        self.assertEqual(mock_zero.return_value, model)

    @patch.object(LinearModel, "zero")
    @patch.object(CalibrateSpectrum, "_get_mz_distance_pairs")
    def test_calibrate_spectrum_when_too_few_points(self, mock_get_mz_distance_pairs, mock_zero) -> None:
        self.mock_problem_mz.__len__.return_value = 2
        mock_get_mz_distance_pairs.return_value = (
            self.mock_problem_mz,
            self.mock_problem_dist,
        )
        model = self.mock_compute.calibrate_spectrum(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_get_mz_distance_pairs.assert_called_once_with(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_zero.assert_called_once_with()
        self.assertEqual(mock_zero.return_value, model)

    @patch.object(LinearModel, "fit_lsq")
    @patch.object(CalibrateSpectrum, "_get_mz_distance_pairs")
    def test_calibrate_spectrum_when_linear(self, mock_get_mz_distance_pairs, mock_fit_linear) -> None:
        self.mock_problem_mz.__len__.return_value = 3
        mock_get_mz_distance_pairs.return_value = (
            self.mock_problem_mz,
            self.mock_problem_dist,
        )
        model = self.mock_compute.calibrate_spectrum(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_get_mz_distance_pairs.assert_called_once_with(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_fit_linear.assert_called_once_with(x_arr=self.mock_problem_mz, y_arr=self.mock_problem_dist)
        self.assertEqual(mock_fit_linear.return_value, model)

    @patch.object(LinearModel, "fit_siegelslopes")
    @patch.object(CalibrateSpectrum, "_get_mz_distance_pairs")
    def test_calibrate_spectrum_when_linear_siegelslopes(
        self, mock_get_mz_distance_pairs, mock_fit_linear_siegelslopes
    ) -> None:
        self.mock_problem_mz.__len__.return_value = 3
        self.mock_compute._model_type = "linear_siegelslopes"
        mock_get_mz_distance_pairs.return_value = (
            self.mock_problem_mz,
            self.mock_problem_dist,
        )
        model = self.mock_compute.calibrate_spectrum(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_get_mz_distance_pairs.assert_called_once_with(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_fit_linear_siegelslopes.assert_called_once_with(x_arr=self.mock_problem_mz, y_arr=self.mock_problem_dist)
        self.assertEqual(mock_fit_linear_siegelslopes.return_value, model)

    @patch.object(PolynomialModel, "fit_lsq")
    @patch.object(CalibrateSpectrum, "_get_mz_distance_pairs")
    def test_calibrate_spectrum_when_polynomial(self, mock_get_mz_distance_pairs, mock_fit_polynomial) -> None:
        self.mock_problem_mz.__len__.return_value = 3
        self.mock_compute._model_type = "poly_2"
        mock_get_mz_distance_pairs.return_value = (
            self.mock_problem_mz,
            self.mock_problem_dist,
        )
        model = self.mock_compute.calibrate_spectrum(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_get_mz_distance_pairs.assert_called_once_with(mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr)
        mock_fit_polynomial.assert_called_once_with(x_arr=self.mock_problem_mz, y_arr=self.mock_problem_dist, degree=2)
        self.assertEqual(mock_fit_polynomial.return_value, model)

    @unittest.skip
    def test_get_mz_distance_pairs(self) -> NoReturn:
        raise NotImplementedError()

    def test_get_mz_distance_pairs_when_no_peaks(self) -> None:
        self.mock_basic_peak_picker.pick_peaks_mz.return_value = np.array([])
        problem_mz, problem_dist = self.mock_compute._get_mz_distance_pairs(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr
        )
        self.mock_basic_peak_picker.pick_peaks_mz.assert_called_once_with(
            mz_arr=self.mock_mz_arr, int_arr=self.mock_int_arr
        )
        np.testing.assert_array_equal(np.array([]), problem_mz)
        np.testing.assert_array_equal(np.array([]), problem_dist)


if __name__ == "__main__":
    unittest.main()
