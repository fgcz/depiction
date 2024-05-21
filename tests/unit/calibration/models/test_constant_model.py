import unittest

import numpy as np

from ionplotter.calibration.models.constant_model import ConstantModel


class TestConstantModel(unittest.TestCase):
    def test_coef(self) -> None:
        np.testing.assert_array_equal(np.array([1234]), ConstantModel(value=1234).coef)

    def test_value(self) -> None:
        self.assertEqual(1234, ConstantModel(value=1234).value)

    def test_is_zero(self) -> None:
        self.assertFalse(ConstantModel(value=1234).is_zero)
        self.assertTrue(ConstantModel(value=0).is_zero)
        self.assertFalse(ConstantModel(value=1e-12).is_zero)

    def test_predict(self) -> None:
        np.testing.assert_array_equal(
            np.array([1234, 1234]),
            ConstantModel(value=1234).predict([1, 2]),
        )

    def test_identity(self) -> None:
        with self.assertRaises(ValueError):
            ConstantModel.identity()

    def test_zero(self) -> None:
        np.testing.assert_array_equal(
            np.array([0]),
            ConstantModel.zero().coef,
        )

    def test_fit_mean(self) -> None:
        np.testing.assert_array_equal(
            50,
            ConstantModel.fit_mean(np.array([1, 2]), np.array([10, 20, 120])).value,
        )

    def test_fit_median(self) -> None:
        np.testing.assert_array_equal(
            20,
            ConstantModel.fit_median(np.array([1, 2]), np.array([10, 20, 120])).value,
        )


if __name__ == "__main__":
    unittest.main()