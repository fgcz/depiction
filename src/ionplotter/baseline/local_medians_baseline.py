from typing import Literal

import numpy as np
import numba
from numba import njit
from numpy.typing import NDArray
from dataclasses import dataclass

from ionplotter.misc import numpy_util


@dataclass(frozen=True)
class LocalMediansBaseline:
    """Implementation of local medians [1] baseline removal, as described in [2] with their choice for handling edges.
    [1]: https://doi.org/10.1007/BF00208805
    [2]: https://doi.org/10.1366/000370210792434350

    The implementation in ppm space is a further extension of the same concept. (Might have been described before.)
    """

    window_size: int | float
    window_unit: Literal["index", "ppm"] = "index"

    def evaluate_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        mz_arr = np.asarray(mz_arr, dtype=float)
        int_arr = np.asarray(int_arr, dtype=float)

        if self.window_unit == "index":
            return _eval_fast_unit_index(int_arr=int_arr, window_size=int(self.window_size))
        elif self.window_unit == "ppm":
            return _eval_fast_unit_ppm(mz_arr=mz_arr, int_arr=int_arr, window_size=float(self.window_size))
        else:
            raise ValueError(f"Unsupported window_unit: {self.window_unit}")

    def subtract_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        baseline = self.evaluate_baseline(mz_arr=mz_arr, int_arr=int_arr)
        return np.maximum(int_arr - baseline, 0)


@njit("float64[:](float64[:], int64)")
def _eval_fast_unit_index(int_arr: NDArray[float], window_size: int) -> NDArray[float]:
    int_baseline = np.zeros_like(int_arr)
    n_values = len(int_baseline)
    w_half = window_size // 2

    # the start and end are handled separately as described in [2]
    begin_value = np.median(int_arr[:window_size])
    end_value = np.median(int_arr[-window_size:])

    for i_boundary in range(0, w_half):
        int_baseline[i_boundary] = begin_value
        int_baseline[-(i_boundary + 1)] = end_value

    # for the rest we compute as expected
    for i_center in range(w_half, n_values - w_half):
        int_baseline[i_center] = np.median(int_arr[i_center - w_half : i_center + w_half + 1])

    return int_baseline


@njit(
    (
        numba.types.Array(numba.types.float64, 1, "C", readonly=True),
        numba.types.Array(numba.types.float64, 1, "C", readonly=True),
        numba.types.float64,
    ),
)
def _eval_fast_unit_ppm(
    mz_arr: NDArray[float],
    int_arr: NDArray[float],
    window_size: float,
) -> NDArray[float]:
    int_baseline = np.zeros_like(int_arr)
    n_values = len(int_baseline)

    # the basic idea, since the window is specified in terms of ppm tolerance, we can compute for every m/z value
    # the tolerance in terms of m/z and then use this array of tolerances to compute the medians in the correct
    # windows
    mz_tol_arr = (window_size / 1e6) * mz_arr
    mz_tol_arr_half = mz_tol_arr / 2.0

    # As before, we handle as a special case the positions at the boundaries which would not be sufficiently covered
    # with a half size window.
    # left: all elements that fall within the full tolerance of the first element
    n_boundary_left = numpy_util.get_first_index(mz_arr - mz_arr[0] > mz_tol_arr[0], True) - 1
    if n_boundary_left:
        int_baseline[: n_boundary_left + 1] = np.median(int_arr[:n_boundary_left])

    # right: this side is a bit more complex, since the error will increase, assuming relatively small window sizes
    #        for now we just approximate it by taking the right-most value analogously to above
    n_boundary_right = n_values - numpy_util.get_first_index(mz_arr[-1] - mz_arr > mz_tol_arr[-1], False)
    if n_boundary_right:
        int_baseline[-n_boundary_right:] = np.median(int_arr[-n_boundary_right:])

    for i in range(n_boundary_left, n_values - n_boundary_right):
        left_idx = np.searchsorted(mz_arr, mz_arr[i] - mz_tol_arr_half[i], side="left")
        right_idx = np.searchsorted(mz_arr, mz_arr[i] + mz_tol_arr_half[i], side="right")

        int_baseline[i] = np.median(int_arr[left_idx:right_idx])

    return int_baseline
