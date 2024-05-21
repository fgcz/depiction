from functools import cached_property
from ionplotter.evaluate_tophat_baseline import EvaluateTophatBaseline
from ionplotter.misc.integration_test_utils import IntegrationTestUtils
from unittest.mock import patch, MagicMock
import ionplotter.evaluate_tophat_baseline as test_module
import numpy as np
import os
import unittest


@patch.dict(os.environ, {"NUMBA_DEBUGINFO": "1"})
class TestEvaluateTophatBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_window_size = 5
        self.mock_window_unit = "index"
        IntegrationTestUtils.treat_warnings_as_error(self)

    @cached_property
    def mock_evaluate(self) -> EvaluateTophatBaseline:
        # noinspection PyTypeChecker
        return EvaluateTophatBaseline(window_size=self.mock_window_size, window_unit=self.mock_window_unit)

    def test_compute_erosion(self) -> None:
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        element_size = 5
        eroded = test_module._compute_erosion(x, element_size)
        np.testing.assert_array_equal([10, 10, 10, 20, 30, 40, 50, 60, 70, 80], eroded)

    def test_compute_dilation(self) -> None:
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        element_size = 5
        dilation = test_module._compute_dilation(x, element_size)
        np.testing.assert_array_equal([30, 40, 50, 60, 70, 80, 90, 100, 100, 100], dilation)

    def test_compute_opening(self) -> None:
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        element_size = 5
        opening = test_module._compute_opening(x, element_size)
        np.testing.assert_array_equal([10, 20, 30, 40, 50, 60, 70, 80, 80, 80], opening)

    def test_optimize_structuring_element_size(self) -> None:
        x = np.array([10, 20, 30, 10, 10, 10, 20, 10, 10, 10, 10, 10, 10, 10, 50, 10, 10], dtype=float)
        element_size = test_module._optimize_structuring_element_size(x, tolerance=1e-6)
        self.assertEqual(3, element_size)

        # sanity check
        np.testing.assert_array_equal(
            test_module._compute_opening(x, element_size),
            test_module._compute_opening(x, element_size + 2),
        )
        self.assertFalse(
            np.array_equal(
                test_module._compute_opening(x, element_size),
                test_module._compute_opening(x, element_size - 2),
            )
        )

    def test_evaluate_baseline(self) -> None:
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        x = np.array([10, 20, 30, 10, 10, 10, 20, 10, 10, 10, 10, 10, 10, 10, 50, 10, 10])
        baseline = self.mock_evaluate.evaluate_baseline(mock_mz_arr, x)
        np.testing.assert_array_equal(np.full_like(x, 10.0), baseline)

    @patch.object(EvaluateTophatBaseline, "evaluate_baseline")
    def test_subtract_baseline(self, method_evaluate_baseline) -> None:
        method_evaluate_baseline.return_value = np.array([20, 20, 30, 30, 30])
        mock_int_arr = np.array([50, 10, 10, 10, 50])
        mock_mz_arr = MagicMock(name="mock_mz_arr")
        int_arr = self.mock_evaluate.subtract_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
        np.testing.assert_array_equal([30, 0, 0, 0, 20], int_arr)

    def test_get_element_size_when_index(self) -> None:
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        self.assertEqual(5, self.mock_evaluate.get_element_size(mz_arr=mock_mz_arr))

    def test_get_element_size_when_ppm(self) -> None:
        self.mock_window_unit = "ppm"
        self.mock_window_size = 500
        mock_mz_arr = np.arange(995, 1005, 0.05)
        self.assertEqual(10, self.mock_evaluate.get_element_size(mz_arr=mock_mz_arr))

    def test_get_element_size_when_invalid(self) -> None:
        self.mock_window_unit = "mz"
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        with self.assertRaises(ValueError):
            self.mock_evaluate.get_element_size(mz_arr=mock_mz_arr)


if __name__ == "__main__":
    unittest.main()
