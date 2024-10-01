import numpy as np
import pytest

from depiction.calibration.models import LinearModel, PolynomialModel
from depiction.calibration.models.fit_model import fit_model


@pytest.fixture
def sample_data():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    return x, y


def test_linear_model(sample_data):
    x, y = sample_data
    model = fit_model(x, y, "linear")
    assert isinstance(model, LinearModel)
    assert np.isclose(model.slope, 2.0, atol=1e-6)
    assert np.isclose(model.intercept, 0.0, atol=1e-6)


def test_polynomial_model(sample_data):
    x, y = sample_data
    model = fit_model(x, y, "poly_2")
    assert isinstance(model, PolynomialModel)
    assert model.degree == 2
    assert np.isclose(model.coef, [0.0, 2.0, 0.0], atol=1e-6).all()


def test_linear_siegelslopes(sample_data):
    x, y = sample_data
    model = fit_model(x, y, "linear_siegelslopes")
    assert isinstance(model, LinearModel)
    np.testing.assert_array_almost_equal(model.coef, [0.0, 2.0], decimal=3)


def test_not_enough_points():
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    linear_model = fit_model(x, y, "linear")
    poly_model = fit_model(x, y, "poly_2")

    assert isinstance(linear_model, LinearModel)
    assert isinstance(poly_model, PolynomialModel)
    assert np.isclose(linear_model.slope, 0.0)
    assert np.isclose(linear_model.intercept, 0.0)
    assert all(np.isclose(poly_model.coef, 0.0))


def test_unknown_model_type():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="Unknown model_type='unknown'"):
        fit_model(x, y, "unknown")


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomial_degrees(sample_data, degree):
    x, y = sample_data
    model = fit_model(x, y, f"poly_{degree}")
    assert isinstance(model, PolynomialModel)
    assert model.degree == degree
    assert len(model.coef) == degree + 1


def test_input_types():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    model = fit_model(np.array(x), np.array(y), "linear")
    assert isinstance(model, LinearModel)


def test_empty_input():
    x = np.array([])
    y = np.array([])
    model = fit_model(x, y, "linear")
    assert isinstance(model, LinearModel)
    assert np.isclose(model.slope, 0.0)
    assert np.isclose(model.intercept, 0.0)


def test_single_point():
    x = np.array([1.0])
    y = np.array([2.0])
    model = fit_model(x, y, "poly_2")
    assert isinstance(model, PolynomialModel)
    assert all(np.isclose(model.coef, 0.0))
