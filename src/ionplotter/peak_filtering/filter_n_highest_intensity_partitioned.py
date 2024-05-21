from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ionplotter.peak_filtering.filter_n_highest_intensity import (
    FilterNHighestIntensity,
)


@dataclass
class FilterNHighestIntensityPartitioned:
    """Returns up to `max_count` many peaks, by picking up to `max_count // n_partitions` many peaks from each
    equal-length segment of the spectrum (in terms of mz values).
    :param max_count: The maximum number of peaks to return.
    :param n_partitions: The number of partitions to divide the spectrum into.
    """

    max_count: int
    n_partitions: int

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        """Returns up to `max_count` many peaks, by picking up to `max_count // n_partitions` many peaks from each
        equal-length segment of the spectrum (in terms of mz values).
        :param spectrum_mz_arr: The mz array of the full spectrum.
        :param spectrum_int_arr: The intensity array of the full spectrum.
        :param peak_idx_arr: The indices of the peaks.
        """
        if len(peak_idx_arr) == 0:
            return np.array([], dtype=int)

        # Determine the mz limits of the partitions.
        mz_partitions = self._get_mz_partitions(spectrum_mz_arr)

        # Setup filter function for each partition.
        filter_fn = FilterNHighestIntensity(max_count=self.max_count // self.n_partitions)

        # Determine the indices of the peaks in each partition.
        peak_idx_result = []
        for i_partition in range(self.n_partitions):
            idx_partition_left = np.searchsorted(spectrum_mz_arr[peak_idx_arr], mz_partitions[i_partition], side="left")
            idx_partition_right = np.searchsorted(
                spectrum_mz_arr[peak_idx_arr],
                mz_partitions[i_partition + 1],
                side="right",
            )
            peak_idx_partition = peak_idx_arr[idx_partition_left:idx_partition_right]
            peak_idx_result.extend(
                filter_fn.filter_index_peaks(
                    spectrum_mz_arr=spectrum_mz_arr,
                    spectrum_int_arr=spectrum_int_arr,
                    peak_idx_arr=peak_idx_partition,
                )
            )
        return np.asarray(peak_idx_result)

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Returns up to `max_count` many peaks, by picking up to `max_count // n_partitions` many peaks from each
        equal-length segment of the spectrum (in terms of mz values).
        :param spectrum_mz_arr: The mz array of the full spectrum.
        :param spectrum_int_arr: The intensity array of the full spectrum.
        :param peak_mz_arr: The mz values of the peaks.
        :param peak_int_arr: The intensity values of the peaks.
        """
        if len(peak_mz_arr) == 0:
            return peak_mz_arr, peak_int_arr

        mz_partitions = self._get_mz_partitions(spectrum_mz_arr)
        filter_fn = FilterNHighestIntensity(max_count=self.max_count // self.n_partitions)

        # apply filtering to each partition
        result_mz = []
        result_int = []
        for i_partition in range(self.n_partitions):
            idx_partition_left = np.searchsorted(peak_mz_arr, mz_partitions[i_partition], side="left")
            idx_partition_right = np.searchsorted(peak_mz_arr, mz_partitions[i_partition + 1], side="right")
            peak_mz_partition = peak_mz_arr[idx_partition_left:idx_partition_right]
            peak_int_partition = peak_int_arr[idx_partition_left:idx_partition_right]
            mz_filtered, int_filtered = filter_fn.filter_peaks(
                spectrum_mz_arr=spectrum_mz_arr,
                spectrum_int_arr=spectrum_int_arr,
                peak_mz_arr=peak_mz_partition,
                peak_int_arr=peak_int_partition,
            )
            result_mz.extend(mz_filtered)
            result_int.extend(int_filtered)

        return np.asarray(result_mz), np.asarray(result_int)

    def _get_mz_partitions(self, spectrum_mz_arr: NDArray[float]) -> NDArray[float]:
        """Returns the mz limits of the partitions."""
        mz_min = np.min(spectrum_mz_arr)
        mz_max = np.max(spectrum_mz_arr)
        return np.linspace(mz_min, mz_max, self.n_partitions + 1)
