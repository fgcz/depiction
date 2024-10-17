from unittest.mock import ANY

import numpy as np
import pytest

from depiction.calibration.models.polynomial_model import PolynomialModel


@pytest.fixture()
def mock_coef():
    return [1, 2, 3]


@pytest.fixture()
def mock_model(mock_coef):
    return PolynomialModel(coef=mock_coef)


def test_coef(mock_model) -> None:
    np.testing.assert_array_equal(mock_model.coef, np.array([1, 2, 3]))


def test_is_zero_when_false(mock_model) -> None:
    assert not mock_model.is_zero


def test_is_zero_when_true() -> None:
    assert PolynomialModel.zero().is_zero
    assert PolynomialModel([0.0, 0, 0]).is_zero


def test_degree(mock_model) -> None:
    assert mock_model.degree == 2


def test_predict(mock_model) -> None:
    np.testing.assert_array_equal(mock_model.predict([1.0, 2]), np.array([6.0, 11]))


def test_identity() -> None:
    np.testing.assert_array_equal(np.array([0, 1]), PolynomialModel.identity().coef)


def test_zero() -> None:
    np.testing.assert_array_equal(np.array([0, 0]), PolynomialModel.zero().coef)


def test_fit_lsq(mocker) -> None:
    mock_polyfit = mocker.patch("numpy.polyfit")
    mock_x = np.array([100, 200, 300])
    mock_y = np.array([1, 2, 3])
    mock_polyfit.return_value = np.array([5, 7])

    model = PolynomialModel.fit_lsq(mock_x, mock_y, degree=5)

    np.testing.assert_array_equal(np.array([5, 7]), model.coef)
    mock_polyfit.assert_called_once_with(ANY, ANY, deg=5)
    np.testing.assert_array_equal(mock_x, mock_polyfit.call_args[0][0])
    np.testing.assert_array_equal(mock_y, mock_polyfit.call_args[0][1])
