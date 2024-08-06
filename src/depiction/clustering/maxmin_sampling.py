from __future__ import annotations

import numpy as np
from numpy.random import Generator


def maxmin_sampling(vectors: np.ndarray, k: int, rng: Generator) -> np.ndarray:
    """
    Sample k diverse vectors from the given set of vectors using the MaxMin algorithm.

    The algorithm works as follows:
    1. Start by randomly selecting one vector.
    2. For each subsequent selection, choose the vector that is farthest from all previously selected vectors.
    3. Repeat until k vectors are selected.

    Parameters:
    vectors (np.ndarray): Array of shape (n, d) where n is the number of vectors and d is the dimension
    k (int): Number of vectors to sample
    rng (Generator): NumPy random number generator

    Returns:
    np.ndarray: Indices of the selected vectors

    Raises:
    ValueError: If k is greater than the number of vectors or less than 1
    """
    n, d = vectors.shape

    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than the number of vectors ({n})")
    if k < 1:
        raise ValueError(f"k ({k}) must be at least 1")

    # Initialize with a random point
    selected = [rng.integers(n)]

    # Compute distances to the selected point
    distances = np.sum((vectors - vectors[selected[0]]) ** 2, axis=1)

    for _ in range(1, k):
        # Select the point with maximum distance to the already selected points
        new_point = np.argmax(distances)
        selected.append(new_point)

        # Update distances
        new_distances = np.sum((vectors - vectors[new_point]) ** 2, axis=1)
        distances = np.minimum(distances, new_distances)

    return np.array(selected)
