import numpy as np
import pytest

from depiction.spectrum.smoothing.evaluate_gaussian_smoothing import EvaluateGaussianSmoothing


@pytest.fixture(autouse=True)
def _setup_test(treat_warnings_as_error):
    pass


def test_evaluate() -> None:
    mz_values = None
    window = 3
    sd = 1
    original_values = np.array([1, 1, 1, 1, 1, 5, 5, 5, 1])
    expected_values = np.array([1, 1, 1, 1, 2.096274, 3.903726, 5, 3.903726, 1])
    eval_gauss = EvaluateGaussianSmoothing(window=window, sd=sd)
    result = eval_gauss.evaluate(mz_values, original_values)
    np.testing.assert_array_almost_equal(expected_values, result)


def test_evaluate_when_signal_shorter_than_filter() -> None:
    # current behavior is to not filter at all in these cases
    mz_values = None
    int_values = np.array([1, 2, 3])
    eval_gauss = EvaluateGaussianSmoothing(window=5)
    result = eval_gauss.evaluate(mz_values, int_values)
    assert int_values is result
