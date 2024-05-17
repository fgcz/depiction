import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from ionmapper.calibration.deprecated.model_smoothing import ModelSmoothing
from typing import NoReturn


class TestModelSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_sigma = 1.0

    @cached_property
    def mock_smoothing(self) -> ModelSmoothing:
        return ModelSmoothing(sigma=self.mock_sigma)

    @unittest.skip
    def test_smooth_spatial(self) -> NoReturn:
        raise NotImplementedError()

    @unittest.skip
    def test_smooth_sequential(self) -> NoReturn:
        raise NotImplementedError()

    def test_get_model_type_and_values_when_valid(self) -> None:
        class MockModel:
            def __init__(self, coef) -> None:
                self.coef = coef

        mock_models = [MockModel(coef=[0, 1]), MockModel(coef=[0, 2])]
        model_type, values = self.mock_smoothing.get_model_type_and_values(models=mock_models)
        self.assertEqual(model_type, MockModel)
        np.testing.assert_array_equal([[0, 1], [0, 2]], values)

    def test_get_model_type_and_values_when_invalid(self) -> None:
        mock_models = [MagicMock(), MagicMock()]
        with self.assertRaises(ValueError) as error:
            self.mock_smoothing.get_model_type_and_values(models=mock_models)
        self.assertIn("all models must have the same type", error.exception.args[0])


if __name__ == "__main__":
    unittest.main()
