import copy

import numpy as np
import pytest

from depiction.tools.simulate.generate_synthetic_panel import sample_random_mz


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


def test_sample_random_mz_output_shape(rng):
    result = sample_random_mz(100, 1000, 10, 50, rng)
    assert result.shape == (50,)


def test_sample_random_mz_range(rng):
    min_mz, max_mz = 100, 1000
    result = sample_random_mz(min_mz, max_mz, 1, 100, rng)
    assert np.min(result) >= min_mz
    assert np.max(result) <= max_mz


def test_sample_random_too_high_min_distance(rng):
    with pytest.raises(ValueError) as error:
        sample_random_mz(100, 1000, 10, 100, rng)
    assert "min_distance_mz" in str(error.value)


def test_sample_random_mz_min_distance(rng):
    min_distance_mz = 5
    result = sample_random_mz(100, 1000, min_distance_mz, 50, rng)
    differences = np.diff(np.sort(result))
    assert np.all(differences >= min_distance_mz)


def test_sample_random_mz_reproducibility(rng):
    result1 = sample_random_mz(100, 1000, 10, 50, copy.deepcopy(rng))
    result2 = sample_random_mz(100, 1000, 10, 50, copy.deepcopy(rng))
    assert np.allclose(result1, result2)


@pytest.mark.parametrize(
    "min_mz,max_mz,min_distance_mz,n",
    [
        (100, 1000, 10, 50),
        (0, 100, 1, 100),
        (1000, 2000, 5, 200),
    ],
)
def test_sample_random_mz_various_inputs(min_mz, max_mz, min_distance_mz, n, rng):
    result = sample_random_mz(min_mz, max_mz, min_distance_mz, n, rng)
    assert result.shape == (n,)
    assert np.min(result) >= min_mz
    assert np.max(result) <= max_mz
    differences = np.diff(np.sort(result))
    assert np.all(differences >= min_distance_mz)


def test_sample_random_mz_invalid_inputs():
    rng = np.random.default_rng(seed=42)
    with pytest.raises(ValueError):
        sample_random_mz(1000, 100, 10, 50, rng)
    with pytest.raises(ValueError):
        sample_random_mz(100, 1000, 1000, 50, rng)
    with pytest.raises(ZeroDivisionError):
        sample_random_mz(100, 1000, 10, 0, rng)


if __name__ == "__main__":
    pytest.main()
