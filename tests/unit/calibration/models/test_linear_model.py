import numpy as np
import pytest

from depiction.calibration.models.linear_model import LinearModel


@pytest.fixture
def mock_coef() -> list[float]:
    return [1, 2]


@pytest.fixture
def mock_model(mock_coef: list[float]) -> LinearModel:
    return LinearModel(coef=mock_coef)


def test_coef(mock_model: LinearModel) -> None:
    np.testing.assert_array_equal(np.array([1, 2]), mock_model.coef, strict=True)


def test_raise_error_when_invalid_coef() -> None:
    with pytest.raises(ValueError):
        LinearModel(coef=[1, 2, 3])


def test_is_zero_when_false(mock_model: LinearModel) -> None:
    assert not mock_model.is_zero


def test_is_zero_when_true(mock_model: LinearModel) -> None:
    mock_coef = [0.0, 0]
    assert LinearModel(coef=mock_coef).is_zero


def test_predict(mock_model: LinearModel) -> None:
    np.testing.assert_array_equal(
        np.array([3.0, 5]),
        mock_model.predict([1.0, 2]),
    )


def test_identity() -> None:
    np.testing.assert_array_equal(
        np.array([0, 1]),
        LinearModel.identity().coef,
    )


def test_zero() -> None:
    np.testing.assert_array_equal(
        np.array([0, 0]),
        LinearModel.zero().coef,
    )


def test_fit_lsq() -> None:
    model = LinearModel.fit_lsq(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert model.intercept == pytest.approx(3, abs=1e-7)
    assert model.slope == pytest.approx(1, abs=1e-7)


def test_fit_linear_siegelslopes() -> None:
    mock_x = np.array([100, 200, 300])
    mock_y = np.array([1, 2, 3])
    model = LinearModel.fit_siegelslopes(mock_x, mock_y)
    np.testing.assert_array_almost_equal(np.array([0, 0.01]), model.coef, decimal=7)


if __name__ == "__main__":
    pytest.main()
