from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ionplotter.baseline.baseline import Baseline
from ionplotter.persistence import ImzmlReadFile


@dataclass(frozen=True)
class TophatBaseline(Baseline):
    """
    Implements the approach described in [1] for baseline removal.
    Optimization is not built in but has to be done, with the optimize_window_size method, if needed.

    [1] Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
        Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.
    """

    window_size: int | float
    window_unit: Literal["ppm", "index"]

    def evaluate_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        mz_arr = np.asarray(mz_arr, dtype=float)
        int_arr = np.asarray(int_arr, dtype=float)

        # element_size = optimize_structuring_element_size(int_arr)
        element_size = self.get_element_size(mz_arr=mz_arr)

        # from the paper
        int_arr_opened = _compute_opening(int_arr, element_size)
        int_arr_approx = (_compute_dilation(int_arr, element_size) + _compute_erosion(int_arr, element_size)) / 2
        return np.minimum(int_arr_approx, int_arr_opened)

    def get_element_size(self, mz_arr: NDArray[float]) -> int:
        if self.window_unit == "ppm":
            # TODO only roughly makes sense for continuous/profile data (not centroided!)
            mean_ppm = np.mean(np.diff(mz_arr) / mz_arr[:-1]) * 1e6
            return round(self.window_size / mean_ppm)
        elif self.window_unit == "index":
            return int(self.window_size)
        else:
            raise ValueError(f"Invalid {self.window_unit=}")

    def optimize_window_size(self, read_file: ImzmlReadFile, n_spectra: int, rng_seed: int = 0) -> int:
        """Optimizes the window size for the provided file, by considering some random spectra. It's possible to set the
        value even to 1, if only one spectrum should be considered.
        :param read_file: The file to optimize the window size for.
        :param n_spectra: The number of random spectra to consider.
        :param rng_seed: The seed for the random number generator.
        """
        rng = np.random.default_rng(rng_seed)
        if n_spectra > read_file.n_spectra:
            raise ValueError(f"Invalid number of spectra to consider: {n_spectra} > {read_file.n_spectra}")
        spectra_ids = rng.choice(read_file.n_spectra, size=n_spectra, replace=False)
        window_sizes = []
        with read_file.reader() as reader:
            for spectrum_id in spectra_ids:
                int_arr = reader.get_spectrum_int(spectrum_id).astype(float)
                window_sizes.append(_optimize_structuring_element_size(int_arr=int_arr, tolerance=1e-6))
        return max(window_sizes)


@njit("float64[:](float64[:], int64)")
def _compute_erosion(x: NDArray[float], element_size: int) -> NDArray[float]:
    eroded = np.zeros_like(x)
    n = len(x)
    hs = element_size // 2
    for i in range(n):
        i_min = max(0, i - hs)
        i_max = min(i + hs + 1, n)
        eroded[i] = np.min(x[i_min:i_max])
    return eroded


@njit("float64[:](float64[:], int64)")
def _compute_dilation(x: NDArray[float], element_size: int) -> NDArray[float]:
    dilation = np.zeros_like(x)
    n = len(x)
    hs = element_size // 2
    for i in range(n):
        i_min = max(0, i - hs)
        i_max = min(i + hs + 1, n)
        dilation[i] = np.max(x[i_min:i_max])
    return dilation


@njit("float64[:](float64[:], int64)")
def _compute_opening(int_arr: NDArray[float], element_size: int) -> NDArray[float]:
    eroded = _compute_erosion(int_arr, element_size)
    return _compute_dilation(eroded, element_size)


@njit(["int64(float64[:], float64)"])
def _optimize_structuring_element_size(int_arr: NDArray[float], tolerance: float) -> int:
    # TODO this is quite slow, taking about 2s for an array with 40k elements
    openings = []

    def eq_relation(opening_1, opening_2):
        rel_diff = np.abs(opening_1 - opening_2) / np.maximum(opening_1, opening_2)
        return np.all(rel_diff < tolerance)

    # TODO correct upper limit (and error handling)
    upper_limit = (len(int_arr) + 1) // 2
    # TODO optimize the search (for large inputs infeasible)
    for element_size in range(3, upper_limit, 2):
        opening = _compute_opening(int_arr=int_arr, element_size=element_size)
        openings.append((element_size, opening))
        if len(openings) >= 3:
            opening_0, opening_1, opening_2 = (
                openings[-1][1],
                openings[-2][1],
                openings[-3][1],
            )
            if eq_relation(opening_0, opening_1) and eq_relation(opening_1, opening_2):
                return openings[-3][0]

    # TODO error handling
    print("Warning: No convergence in optimize_structuring_element_size")
    return upper_limit
