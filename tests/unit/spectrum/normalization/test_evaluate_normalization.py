import unittest
from unittest.mock import MagicMock

import numpy as np

from depiction.spectrum.normalization.evaluate_normalization import (
    EvaluateMedianNormalization,
    EvaluateTICNormalization,
)


class TestEvaluateTICNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_mz_values = MagicMock(name="mock_mz_values", spec=[])
        self.mock_int_values = np.array([10.0, 20, 10, 10])

    def test_evaluate_when_default(self) -> None:
        eval = EvaluateTICNormalization()
        result = eval.evaluate(mz_values=self.mock_mz_values, int_values=self.mock_int_values)
        np.testing.assert_array_equal(np.array([0.2, 0.4, 0.2, 0.2]), result)

    def test_evaluate_when_custom_target(self) -> None:
        eval = EvaluateTICNormalization(target_value=2.5)
        result = eval.evaluate(mz_values=self.mock_mz_values, int_values=self.mock_int_values)
        np.testing.assert_array_equal(np.array([0.5, 1.0, 0.5, 0.5]), result)


class TestEvaluateMedianNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_mz_values = MagicMock(name="mock_mz_values", spec=[])
        self.mock_int_values = np.array([10.0, 20, 10, 10])

    def test_evaluate_when_default(self) -> None:
        eval = EvaluateMedianNormalization()
        result = eval.evaluate(mz_values=self.mock_mz_values, int_values=self.mock_int_values)
        np.testing.assert_array_equal(np.array([1.0, 2.0, 1, 1]), result)

    def test_evaluate_when_custom_target(self) -> None:
        eval = EvaluateMedianNormalization(target_value=2.5)
        result = eval.evaluate(mz_values=self.mock_mz_values, int_values=self.mock_int_values)
        np.testing.assert_array_equal(np.array([2.5, 5, 2.5, 2.5]), result)


if __name__ == "__main__":
    unittest.main()
