import unittest

import numpy as np

from ionmapper.evaluate_signal_proc import EvaluateGaussianSmoothing
from ionmapper.misc.integration_test_utils import IntegrationTestUtils

class TestEvaluateGaussianSmoothing(unittest.TestCase):
    def setUp(self):
        IntegrationTestUtils.treat_warnings_as_error(self)

    def test_evaluate(self):
        mz_values = None
        window = 3
        sd = 1
        original_values = np.array([1, 1, 1, 1, 1, 5, 5, 5, 1])
        expected_values = np.array([1, 1, 1, 1, 2.096274, 3.903726, 5, 3.903726, 1])
        eval_gauss = EvaluateGaussianSmoothing(window=window, sd=sd)
        result = eval_gauss.evaluate(mz_values, original_values)
        np.testing.assert_array_almost_equal(expected_values, result)

    def test_evaluate_when_signal_shorter_than_filter(self):
        # current behavior is to not filter at all in these cases
        mz_values = None
        int_values = np.array([1, 2, 3])
        eval_gauss = EvaluateGaussianSmoothing(window=5)
        result = eval_gauss.evaluate(mz_values, int_values)
        self.assertIs(int_values, result)


if __name__ == "__main__":
    unittest.main()
