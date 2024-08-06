from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from depiction.clustering.metrics import cross_correlation


@pytest.fixture
def sample_2d_arrays():
    return {
        "A": np.array([[1, 2, 3], [4, 5, 6]]),
        "B": np.array([[1, 2, 3], [4, 5, 6]]),
        "B_perfect": np.array([[2, 4, 6], [8, 10, 12]]),
        "B_no_corr": np.array([[1, 1, 1], [1, 1, 1]]),
        "B_negative": np.array([[-1, -2, -3], [-4, -5, -6]]),
        "B_different_shape": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }


@pytest.fixture
def sample_lists():
    return {
        "A": [[1, 2, 3], [4, 5, 6]],
        "B": [[1, 2, 3], [4, 5, 6]],
    }


@pytest.fixture
def invalid_inputs():
    return {
        "1d": (np.array([1, 2, 3]), np.array([4, 5, 6])),
        "mismatched": (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [3, 4]])),
    }


@pytest.mark.parametrize(
    "key_a, key_b, expected",
    [
        ("A", "B_perfect", np.array([[1, 1], [1, 1]])),
        ("A", "B_no_corr", np.array([[0, 0], [0, 0]])),
        ("A", "B_negative", np.array([[-1, -1], [-1, -1]])),
    ],
)
def test_correlations(sample_2d_arrays, key_a, key_b, expected):
    result = cross_correlation(sample_2d_arrays[key_a], sample_2d_arrays[key_b])
    assert_array_almost_equal(result, expected)


def test_different_shapes(sample_2d_arrays):
    result = cross_correlation(sample_2d_arrays["A"], sample_2d_arrays["B_different_shape"])
    assert result.shape == (2, 3)


def test_list_input(sample_lists):
    result = cross_correlation(sample_lists["A"], sample_lists["B"])
    expected = np.array([[1, 1], [1, 1]])
    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "key, error_msg",
    [
        ("1d", "Input arrays must be 2-dimensional"),
        ("mismatched", "Arrays must have the same second dimension"),
    ],
)
def test_invalid_inputs(invalid_inputs, key, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        cross_correlation(*invalid_inputs[key])
