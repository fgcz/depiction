import unittest
from functools import cached_property

import numpy as np

from ionmapper.tools.limit_mz_range import LimitMzRange


class TestLimitMzRange(unittest.TestCase):
    def setUp(self):
        self.mock_mz_range = (100.0, 110.0)

    @cached_property
    def mock_target(self):
        return LimitMzRange(mz_range=self.mock_mz_range)

    def test_evaluate_spectrum(self):
        mz_arr = np.array([99, 100.0, 105, 110.0, 111.0])
        int_arr = np.arange(5)
        result_mz, result_int = self.mock_target.evaluate_spectrum(mz_arr, int_arr)
        np.testing.assert_array_equal(np.array([100.0, 105.0, 110.0]), result_mz)
        np.testing.assert_array_equal(np.array([1, 2, 3]), result_int)

    def test_evaluate_spectrum_when_left_out_of_bounds(self):
        mz_arr = np.array([99, 100.0, 105, 110.0, 111.0])
        int_arr = np.arange(5)
        self.mock_mz_range = (80.0, 110.0)
        result_mz, result_int = self.mock_target.evaluate_spectrum(mz_arr, int_arr)
        np.testing.assert_array_equal(np.array([99.0, 100.0, 105.0, 110.0]), result_mz)
        np.testing.assert_array_equal(np.array([0, 1, 2, 3]), result_int)

    def test_evaluate_spectrum_when_right_out_of_bounds(self):
        mz_arr = np.array([99, 100.0, 105, 110.0, 111.0])
        int_arr = np.arange(5)
        self.mock_mz_range = (100.0, 120.0)
        result_mz, result_int = self.mock_target.evaluate_spectrum(mz_arr, int_arr)
        np.testing.assert_array_equal(np.array([100.0, 105.0, 110.0, 111.0]), result_mz)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result_int)


if __name__ == "__main__":
    unittest.main()
