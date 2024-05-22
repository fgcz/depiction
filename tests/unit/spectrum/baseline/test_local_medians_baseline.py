from functools import cached_property

from depiction.spectrum.baseline import LocalMediansBaseline
from depiction.misc.integration_test_utils import IntegrationTestUtils
from unittest.mock import MagicMock, patch
import numpy as np
import os
import unittest


@patch.dict(os.environ, {"NUMBA_DEBUGINFO": "1"})
class TestLocalMediansBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_window_unit = "index"
        self.mock_window_size = 5
        IntegrationTestUtils.treat_warnings_as_error(self)

    @cached_property
    def mock_baseline(self) -> LocalMediansBaseline:
        return LocalMediansBaseline(window_size=self.mock_window_size, window_unit=self.mock_window_unit)

    def test_evaluate_baseline_when_unit_index(self) -> None:
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        mock_int_arr = np.array([0, 0, 10, 10, 10, 10, 0, 0, 0, 10, 10])
        int_baseline = self.mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        np.testing.assert_array_equal([10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0], int_baseline)

        # the operation should be symmetric
        mock_int_arr_rev = np.flip(mock_int_arr)
        int_baseline_rev = self.mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr_rev)
        np.testing.assert_array_equal([0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10], int_baseline_rev)

    def test_evaluate_baseline_when_unit_ppm(self) -> None:
        self.mock_window_unit = "ppm"
        mock_mz_arr = np.linspace(10, 100, 20)
        mock_int_arr = np.ones(20)
        int_baseline = self.mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        np.testing.assert_array_equal(np.ones(20), int_baseline)

    def test_evaluate_baseline_when_unit_ppm_correct_left(self) -> None:
        self.mock_window_unit = "ppm"
        self.mock_window_size = 500
        # to keep it simple, construct to have an almost constant ppm error and then count
        n_values = 20
        mock_mz_arr = np.zeros(n_values)
        mock_mz_arr[0] = 200
        mock_int_arr = np.ones(n_values)
        mock_int_arr[:2] = 0
        ppm_distance = 150  # i.e. not symmetric but that's not the point here
        for i in range(1, n_values):
            mz_error = ppm_distance / 1e6 * mock_mz_arr[i - 1]
            mock_mz_arr[i] = mock_mz_arr[i - 1] + mz_error

        int_baseline = self.mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        expected_arr = np.ones(n_values)
        expected_arr[:3] = 0
        np.testing.assert_array_equal(expected_arr, int_baseline)

    def test_evaluate_baseline_when_unit_ppm_correct_right(self) -> None:
        self.mock_window_unit = "ppm"
        self.mock_window_size = 500
        # to keep it simple, construct to have an almost constant ppm error and then count
        n_values = 20
        mock_mz_arr = np.zeros(n_values)
        mock_mz_arr[0] = 200
        mock_int_arr = np.ones(n_values)
        mock_int_arr[-2:] = 0
        ppm_distance = 200  # i.e. not symmetric but that's not the point here
        for i in range(1, n_values):
            mz_error = ppm_distance / 1e6 * mock_mz_arr[i - 1]
            mock_mz_arr[i] = mock_mz_arr[i - 1] + mz_error

        int_baseline = self.mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        expected_arr = np.ones(n_values)
        expected_arr[-3:] = 0
        np.testing.assert_array_equal(expected_arr, int_baseline)

    @patch.object(LocalMediansBaseline, "evaluate_baseline")
    def test_subtract_baseline(self, method_evaluate_baseline) -> None:
        method_evaluate_baseline.return_value = np.array([20, 20, 30, 30, 30])
        mock_int_arr = np.array([50, 10, 10, 10, 50])
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        int_arr = self.mock_baseline.subtract_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        np.testing.assert_array_equal([30, 0, 0, 0, 20], int_arr)


if __name__ == "__main__":
    unittest.main()
