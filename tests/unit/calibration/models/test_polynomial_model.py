import unittest
from functools import cached_property
from unittest.mock import patch, ANY

import numpy as np

from ionplotter.calibration.models.polynomial_model import PolynomialModel


class TestPolynomialModel(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_coef = [1, 2, 3]

    @cached_property
    def mock_model(self):
        return PolynomialModel(coef=self.mock_coef)

    def test_coef(self) -> None:
        np.testing.assert_array_equal(
            np.array([1, 2, 3]),
            self.mock_model.coef,
        )

    def test_is_zero_when_false(self) -> None:
        self.assertFalse(self.mock_model.is_zero)

    def test_is_zero_when_true(self) -> None:
        self.assertTrue(PolynomialModel.zero().is_zero)
        self.mock_coef = [0.0, 0, 0]
        self.assertTrue(self.mock_model.is_zero)

    def test_degree(self) -> None:
        self.assertEqual(2, self.mock_model.degree)

    def test_predict(self) -> None:
        np.testing.assert_array_equal(
            np.array([6.0, 11]),
            self.mock_model.predict([1.0, 2]),
        )

    def test_identity(self) -> None:
        np.testing.assert_array_equal(
            np.array([0, 1]),
            PolynomialModel.identity().coef,
        )

    def test_zero(self) -> None:
        np.testing.assert_array_equal(
            np.array([0, 0]),
            PolynomialModel.zero().coef,
        )

    @patch("numpy.polyfit")
    def test_fit_lsq(self, mock_polyfit) -> None:
        self.mock_model_type = "poly_5"
        mock_x = np.array([100, 200, 300])
        mock_y = np.array([1, 2, 3])
        mock_polyfit.return_value = np.array([5, 7])

        model = PolynomialModel.fit_lsq(mock_x, mock_y, degree=5)

        np.testing.assert_array_equal(np.array([5, 7]), model.coef)
        mock_polyfit.assert_called_once_with(ANY, ANY, deg=5)
        np.testing.assert_array_equal(mock_x, mock_polyfit.call_args[0][0])
        np.testing.assert_array_equal(mock_y, mock_polyfit.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
