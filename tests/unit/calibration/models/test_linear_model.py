import unittest
from functools import cached_property

import numpy as np

from ionmapper.calibration.models.linear_model import LinearModel


class TestLinearModel(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_coef = [1, 2]

    @cached_property
    def mock_model(self):
        return LinearModel(coef=self.mock_coef)

    def test_coef(self) -> None:
        np.testing.assert_array_equal(np.array([1, 2]), self.mock_model.coef, strict=True)

    def test_raise_error_when_invalid_coef(self) -> None:
        with self.assertRaises(ValueError):
            LinearModel(coef=[1, 2, 3])

    def test_is_zero_when_false(self) -> None:
        self.assertFalse(self.mock_model.is_zero)

    def test_is_zero_when_true(self) -> None:
        self.assertTrue(LinearModel.zero().is_zero)
        self.mock_coef = [0.0, 0]
        self.assertTrue(self.mock_model.is_zero)

    def test_predict(self) -> None:
        np.testing.assert_array_equal(
            np.array([3.0, 5]),
            self.mock_model.predict([1.0, 2]),
        )

    def test_identity(self) -> None:
        np.testing.assert_array_equal(
            np.array([0, 1]),
            LinearModel.identity().coef,
        )

    def test_zero(self) -> None:
        np.testing.assert_array_equal(
            np.array([0, 0]),
            LinearModel.zero().coef,
        )

    def test_fit_lsq(self) -> None:
        model = LinearModel.fit_lsq(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.assertAlmostEqual(3, model.intercept, places=7)
        self.assertAlmostEqual(1, model.slope, places=7)

    def test_fit_linear_siegelslopes(self) -> None:
        mock_x = np.array([100, 200, 300])
        mock_y = np.array([1, 2, 3])
        model = LinearModel.fit_siegelslopes(mock_x, mock_y)
        np.testing.assert_array_almost_equal(np.array([0, 0.01]), model.coef, decimal=7)


if __name__ == "__main__":
    unittest.main()
