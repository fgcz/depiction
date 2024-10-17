import numpy as np
import pytest

from depiction.calibration.models.constant_model import ConstantModel


def test_coef() -> None:
    np.testing.assert_array_equal(np.array([1234]), ConstantModel(value=1234).coef)


def test_value() -> None:
    assert 1234 == ConstantModel(value=1234).value


def test_is_zero() -> None:
    assert not ConstantModel(value=1234).is_zero
    assert ConstantModel(value=0).is_zero
    assert not ConstantModel(value=1e-12).is_zero


def test_predict() -> None:
    np.testing.assert_array_equal(
        np.array([1234, 1234]),
        ConstantModel(value=1234).predict([1, 2]),
    )


def test_identity() -> None:
    with pytest.raises(ValueError):
        ConstantModel.identity()


def test_zero() -> None:
    np.testing.assert_array_equal(
        np.array([0]),
        ConstantModel.zero().coef,
    )


def test_fit_mean() -> None:
    np.testing.assert_array_equal(
        50,
        ConstantModel.fit_mean(np.array([1, 2]), np.array([10, 20, 120])).value,
    )


def test_fit_median() -> None:
    np.testing.assert_array_equal(
        20,
        ConstantModel.fit_median(np.array([1, 2]), np.array([10, 20, 120])).value,
    )
