from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterByIntensity:
    """Filters peaks by their intensity.
    :param min_intensity: The minimum intensity, below which peaks are discarded.
    :param normalization: optional normalization (for thresholding only), either `None`, `"tic"`, `"median"`.
    """

    min_intensity: float
    normalization: Optional[str] = None

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        """Returns the subset of the peak indices which have an intensity above the threshold.
        :param spectrum_int_arr: The intensity array.
        :param peak_idx_arr: The indices of the peaks.
        """
        if len(peak_idx_arr) == 0:
            return np.array([], dtype=int)
        int_arr_norm = self._normalize_intensities(
            spectrum_int_arr=spectrum_int_arr, peak_int_arr=spectrum_int_arr[peak_idx_arr]
        )
        int_arr_idx = int_arr_norm >= self.min_intensity
        return peak_idx_arr[int_arr_idx]

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        if len(peak_mz_arr) == 0:
            return peak_mz_arr, peak_int_arr
        int_arr_norm = self._normalize_intensities(spectrum_int_arr=spectrum_int_arr, peak_int_arr=peak_int_arr)
        int_arr_idx = int_arr_norm >= self.min_intensity
        return peak_mz_arr[int_arr_idx], peak_int_arr[int_arr_idx]

    def _normalize_intensities(self, spectrum_int_arr: NDArray[float], peak_int_arr: NDArray[float]) -> NDArray[float]:
        if self.normalization == "tic":
            norm = np.sum(spectrum_int_arr)
        elif self.normalization == "median":
            norm = np.median(peak_int_arr)
        elif self.normalization == "vec_norm":
            norm = np.linalg.norm(peak_int_arr)
        elif self.normalization is None:
            norm = None
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        if norm is None:
            return peak_int_arr
        elif abs(norm) < np.finfo(float).eps:
            return np.zeros_like(peak_int_arr)
        else:
            return peak_int_arr / norm

    def debug_diagnose_threshold_correspondence(
        self,
        spectrum_int_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ):
        input_threshold = self.min_intensity
        input_threshold_unit = self.normalization
        min_val = np.finfo(float).eps

        norm_tic = max(np.sum(spectrum_int_arr), min_val)
        norm_median = max(np.median(peak_int_arr), min_val)
        norm_vec_norm = max(np.linalg.norm(peak_int_arr), min_val)

        if input_threshold_unit == "tic":
            thresholds = {
                "tic": input_threshold,
                "median": input_threshold / norm_tic * norm_median,
                "vec_norm": input_threshold / norm_tic * norm_vec_norm,
            }
        elif input_threshold_unit == "median":
            thresholds = {
                "tic": input_threshold / norm_median * norm_tic,
                "median": input_threshold,
                "vec_norm": input_threshold / norm_median * norm_vec_norm,
            }
        elif input_threshold_unit == "vec_norm":
            thresholds = {
                "tic": input_threshold / norm_vec_norm * norm_tic,
                "median": input_threshold / norm_vec_norm * norm_median,
                "vec_norm": input_threshold,
            }
        else:
            raise ValueError(f"Unknown unit: {input_threshold_unit}")

        print("Threshold value conversion:", thresholds)
