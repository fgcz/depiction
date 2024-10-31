from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterNHighestIntensity:
    max_count: int

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_idx_arr: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Returns up to `max_count` many peaks by picking the ones with the highest intensity.
        If possible `max_count` many peaks will be returned, but if there are not enough peaks in the data,
        all peaks will be returned."""
        n_peaks = len(peak_idx_arr)
        if n_peaks <= self.max_count:
            return peak_idx_arr
        else:
            # sort by intensity and return the indices of the max_count highest peaks
            int_arr = spectrum_int_arr[peak_idx_arr]
            sorted_idx = np.argsort(int_arr)
            return peak_idx_arr[sorted_idx[-self.max_count :]]

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_mz_arr: NDArray[np.float64],
        peak_int_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns up to `max_count` many peaks by picking sequentially the one with the next highest intensity.
        If possible `max_count` many peaks will be returned, but if there are not enough peaks in the data,
        all peaks will be returned.
        :param spectrum_mz_arr: The mz array of the full spectrum.
        :param spectrum_int_arr: The intensity array of the full spectrum.
        :param peak_mz_arr: The mz array of the peaks.
        :param peak_int_arr: The intensity array of the peaks.
        """
        n_peaks = len(peak_mz_arr)
        if n_peaks <= self.max_count:
            return peak_mz_arr, peak_int_arr
        else:
            # sort by intensity and return the indices of the max_count highest peaks
            sorted_idx = np.sort(np.argsort(peak_int_arr)[-self.max_count :])
            return peak_mz_arr[sorted_idx], peak_int_arr[sorted_idx]
