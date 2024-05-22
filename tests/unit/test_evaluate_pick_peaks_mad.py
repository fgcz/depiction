import unittest

import numpy as np

from depiction.evaluate_pick_peaks import EvaluatePickPeaksMAD


class TestEvaluatePickPeaksMAD(unittest.TestCase):
    def test_evaluate_sample(self) -> None:
        # note: this unit test is not really ideal right now
        n_points = 20
        # deterministic noise
        int_array = np.sin(np.array([0.1] * n_points)) * 0.2
        int_array[5] = 2.5
        int_array[11] = 3.5
        # evaluate
        result = EvaluatePickPeaksMAD().evaluate(int_array=int_array)
        np.testing.assert_array_equal(np.array([5, 11]), result)


if __name__ == "__main__":
    unittest.main()
