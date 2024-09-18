from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from depiction.clustering.maxmin_sampling import maxmin_sampling


@pytest.fixture
def sample_vectors():
    return np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])


@pytest.fixture
def rng():
    return np.random.default_rng(42)  # Fixed seed for reproducibility


def test_basic_sampling(sample_vectors, rng):
    k = 3
    result = maxmin_sampling(sample_vectors, k, rng)
    assert len(result) == k
    assert len(np.unique(result)) == k  # All selected indices should be unique


def test_full_sampling(sample_vectors, rng):
    k = len(sample_vectors)
    result = maxmin_sampling(sample_vectors, k, rng)
    assert_array_equal(np.sort(result), np.arange(k))


def test_single_sample(sample_vectors, rng):
    result = maxmin_sampling(sample_vectors, 1, rng)
    assert len(result) == 1
    assert 0 <= result[0] < len(sample_vectors)


def test_diverse_sampling(rng):
    vectors = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    k = 3
    result = maxmin_sampling(vectors, k, rng)
    selected_vectors = vectors[result]

    # Calculate pairwise distances between selected vectors
    distances = np.sum((selected_vectors[:, np.newaxis] - selected_vectors) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances

    # Check that the minimum distance is relatively large
    assert np.min(distances) > 0.5


def test_invalid_k_too_large(sample_vectors, rng):
    with pytest.raises(ValueError, match="k .* cannot be greater than the number of vectors"):
        maxmin_sampling(sample_vectors, len(sample_vectors) + 1, rng)


def test_invalid_k_too_small(sample_vectors, rng):
    with pytest.raises(ValueError, match="k .* must be at least 1"):
        maxmin_sampling(sample_vectors, 0, rng)


def test_reproducibility(sample_vectors):
    k = 3
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = maxmin_sampling(sample_vectors, k, rng1)
    result2 = maxmin_sampling(sample_vectors, k, rng2)

    assert_array_equal(result1, result2)


def test_high_dimensional_data(rng):
    vectors = np.random.rand(100, 50)  # 100 vectors of 50 dimensions
    k = 10
    result = maxmin_sampling(vectors, k, rng)
    assert len(result) == k
    assert len(np.unique(result)) == k


def test_edge_case_identical_vectors(rng):
    vectors = np.ones((10, 3))  # 10 identical 3D vectors
    k = 5
    result = maxmin_sampling(vectors, k, rng)
    assert len(result) == k
    assert len(np.unique(result)) == 1
    assert all(result == result[0])


if __name__ == "__main__":
    pytest.main()
