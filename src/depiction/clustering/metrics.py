from __future__ import annotations

import numpy as np


def cross_correlation(A: np.ndarray | list, B: np.ndarray | list) -> np.ndarray:
    """
    Compute cross-correlation coefficients between two 2D arrays.

    Parameters:
    A (np.ndarray | list): Shape (m, n)
    B (np.ndarray | list): Shape (k, n)

    Returns:
    np.ndarray: Cross-correlation coefficients, shape (m, k)

    Raises:
    ValueError: If input arrays are not 2D or don't have matching second dimensions
    """
    # Convert inputs to numpy arrays if they're lists
    A = np.array(A) if isinstance(A, list) else A
    B = np.array(B) if isinstance(B, list) else B

    # Check if inputs are 2D
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")

    # Ensure the second dimension matches
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Arrays must have the same second dimension. A has shape {A.shape}, B has shape {B.shape}")

    # Subtract mean
    A_centered = A - np.mean(A, axis=1, keepdims=True)
    B_centered = B - np.mean(B, axis=1, keepdims=True)

    # Compute numerator
    numerator = np.dot(A_centered, B_centered.T)

    # Compute denominator
    denominator = np.sqrt(np.sum(A_centered**2, axis=1, keepdims=True) * np.sum(B_centered**2, axis=1))

    # Avoid division by zero
    denominator = np.maximum(denominator, np.finfo(float).eps)

    # Compute correlation
    correlation = numerator / denominator

    return correlation
