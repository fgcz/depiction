import unittest
from functools import cached_property

import numpy as np

from ionplotter.misc.experimental.resample_mass_axis import ResampleMassAxis


class TestResampleMassAxis(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_target_mz_arr = np.array([100.0, 101, 102, 103, 104, 105])

    @cached_property
    def mock_resample(self):
        return ResampleMassAxis(target_mz_arr=self.mock_target_mz_arr)

    def test_evaluate_spectrum_when_exact(self) -> None:
        mz_arr = self.mock_target_mz_arr.copy()
        int_arr = np.array([1.0, 2, 3, 4, 5, 6])
        result = self.mock_resample.evaluate_spectrum(mz_arr, int_arr)
        np.testing.assert_array_equal(int_arr, result)

    def test_evaluate_spectrum_when_interpolated(self) -> None:
        mz_arr = np.array([100, 102.5, 105])
        int_arr = np.array([1, 3.5, 6])
        result = self.mock_resample.evaluate_spectrum(mz_arr, int_arr)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_almost_equal(expected, result, decimal=6)

    def test_evaluate_spectrum_when_extrapolating_expect_zeros(self) -> None:
        mz_arr = np.array([102.0, 103, 104])
        int_arr = np.array([1.0, 2, 3])
        result = self.mock_resample.evaluate_spectrum(mz_arr, int_arr)
        expected = np.array([0.0, 0, 1, 2, 3, 0])
        np.testing.assert_array_almost_equal(expected, result, decimal=6)


if __name__ == "__main__":
    unittest.main()
