from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterByReferencePeakDistance:
    max_distance: float
    reference_mz: NDArray[np.float64]

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_idx_arr: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Returns the subset of the peak indices which are within the distance threshold of a reference peak.
        :param spectrum_mz_arr: The mz array.
        :param spectrum_int_arr: The intensity array.
        :param peak_idx_arr: The indices of the peaks.
        :param reference_mz: The mz values of the reference peaks.
        :param distance_threshold: The maximum distance between a reference peak and a sample peak to be considered.
        """
        if len(peak_idx_arr) == 0:
            return np.array([], dtype=int)

        # get the data
        # TODO avoid potentially redundant sorting
        peak_mz = np.sort(spectrum_mz_arr[peak_idx_arr])
        reference_mz = np.sort(self.reference_mz)
        n_peaks = len(peak_mz)

        # for every peak, find the distance to its closest reference
        min_distances = np.full(n_peaks, np.inf)
        for i_peak, mz in enumerate(peak_mz):
            min_distances[i_peak] = np.min(np.abs(reference_mz - mz))

        # determine the peaks which are within the distance threshold
        array_idx_within_threshold = min_distances <= self.max_distance
        if not np.any(array_idx_within_threshold):
            return np.array([])
        else:
            return peak_idx_arr[array_idx_within_threshold]
