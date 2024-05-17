import unittest

import numpy as np

from ionmapper.evaluate_baseline_correction import EvaluateMWMVBaselineCorrection


class TestEvaluateMWMVBaselineCorrection(unittest.TestCase):
    def setUp(self):
        self.eval_baseline = EvaluateMWMVBaselineCorrection()

    def test_subtract_baseline_basic(self):
        mz_arr = np.arange(20)
        int_arr = np.full(20, 0.5)
        int_arr[10] += 2
        result = self.eval_baseline.subtract_baseline(mz_arr, int_arr)
        expected = np.zeros(20)
        expected[10] = 2
        np.testing.assert_array_equal(expected, result)

    def test_subtract_baseline_exact(self):
        int_arr = 0.1 * np.arange(11)
        result = self.eval_baseline.subtract_baseline(mz_arr=None, int_arr=int_arr)
        expected = np.array(
            [
                0.0,
                0.08888889,
                0.16666667,
                0.23333333,
                0.28888889,
                0.33333333,
                0.36666667,
                0.38888889,
                0.4,
                0.4,
                0.4,
            ]
        )
        np.testing.assert_array_almost_equal(expected, result)

    def test_subtract_baseline_when_no_mz_values(self):
        mz_arr = np.arange(20) * 0.1
        int_arr = np.arange(20)
        result_with_mz = self.eval_baseline.subtract_baseline(mz_arr=mz_arr, int_arr=int_arr)
        result_without_mz = self.eval_baseline.subtract_baseline(mz_arr=None, int_arr=int_arr)
        np.testing.assert_array_equal(result_with_mz, result_without_mz)

    def test_subtract_baseline_small_spec(self):
        mz_values = np.array([1, 2, 3])
        int_values = np.array([1, 2, 1])
        result = self.eval_baseline.subtract_baseline(mz_values, int_values)
        np.testing.assert_array_equal(int_values, result)


if __name__ == "__main__":
    unittest.main()
