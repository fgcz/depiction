import numpy as np
from numpy.typing import NDArray
from numba import njit


class NumpyUtil:
    """Provides some missing numpy functionality."""

    @staticmethod
    def search_sorted_closest(full_array: NDArray[float], values: NDArray[float]) -> NDArray[int]:
        """Returns an array of indices with one entry for every value in values, indicating the index of the closest
        value in full_array. `full_array` must be sorted."""
        # determine the insert indices
        insert_idx = np.searchsorted(full_array, values, side="right")

        # compare insert_idx and insert_idx - 1 for each position to identify which is closer
        pad_full_array = np.pad(full_array.astype(float), (0, 1), mode="constant", constant_values=np.inf)

        distance_to_left = np.abs(values - pad_full_array[insert_idx - 1])
        distance_to_right = np.abs(values - pad_full_array[insert_idx])

        # if the value is closer to the right, we need to increment the index
        is_closer_to_left = distance_to_left < distance_to_right
        closest_indices = insert_idx - is_closer_to_left

        return closest_indices

    @staticmethod
    def get_sorted_indices_within_distance(array: NDArray[float], value: float, max_distance: float) -> NDArray[int]:
        """Returns the indices of values withn at most (inclusive) max_distance from value. This is equivalent to
        np.where(np.abs(array-value) <= max_distance), but this method is faster by making use of the sorted constraint
        on the input array."""
        left_idx = np.searchsorted(array, value - max_distance, side="left")
        right_idx = np.searchsorted(array, value + max_distance, side="right")
        return np.arange(left_idx, right_idx)


@njit(["int64(float64[:], float64)", "int64(int64[:], int64)", "int64(boolean[:], boolean)"])
def get_first_index(array: NDArray, value) -> int:
    for i in range(len(array)):
        if array[i] == value:
            return i
    return len(array)
