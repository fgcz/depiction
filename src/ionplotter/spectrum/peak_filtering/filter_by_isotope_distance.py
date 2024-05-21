from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FilterByIsotopeDistance:
    min_distance: float
    max_distance: float
    far_distance: float

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        """Returns the subset of the peak indices for which the next peak is within the expected distance.
        The full range of permitted distances is [min_dist, max_dist] U [far_dist, inf) which accounts for peaks
        from separate clusters.
        :param spectrum_mz_arr: The mz array.
        :param peak_idx_arr: The indices of the peaks.
        :param min_dist: The minimum distance between peaks.
        :param max_dist: The maximum distance between peaks.
        :param far_dist: The minimum distance between peaks from separate clusters.
        """
        n_peaks = len(peak_idx_arr)
        is_valid = np.zeros(n_peaks, dtype=bool)

        # TODO probably it's faster with np.diff and concatenating for the first/last value, refactor after test
        for i_peak in range(n_peaks):
            # get the distances to the previous and next peak (and set to 1 if there is no previous/next peak)
            mz_peak = spectrum_mz_arr[peak_idx_arr[i_peak]]
            dist_before, dist_after = 1, 1
            if i_peak > 0:
                dist_before = mz_peak - spectrum_mz_arr[peak_idx_arr[i_peak - 1]]
            if i_peak < n_peaks - 1:
                dist_after = spectrum_mz_arr[peak_idx_arr[i_peak + 1]] - mz_peak

            # check if the distances are within the permitted ranges
            valid_before = (self.min_distance <= dist_before <= self.max_distance) or (dist_before >= self.far_distance)
            valid_after = (self.min_distance <= dist_after <= self.max_distance) or (dist_after >= self.far_distance)
            is_valid[i_peak] = valid_before and valid_after

        # return the indices of the valid peaks
        return peak_idx_arr[is_valid]
