from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
import scipy.signal

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ionmapper.peak_filtering import PeakFilteringType


@dataclass
class BasicPeakPicker:
    """A basic peak picker, for use in the calibration process.
    It combines
    - a simple gaussian smoothing of the intensity array, and
    - a peak picking algorithm based on local maxima detection with a prominence threshold.
    This is a "Basic" Peak Picker not applying advanced logic, and the smooth_sigma will be scaled by the average
    distance between peaks in the spectrum. It might be worth revisiting this in the future.
    :param smooth_sigma: sigma for the gaussian smoothing applied before peak detection, in m/z units
    :param min_prominence: prominence threshold for the peak detection
    :param min_distance: minimal distance between peaks
    :param min_distance_unit: units for the minimal distance between peaks, either "index" or "mz"
    """

    smooth_sigma: float | None
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: str | None = None
    peak_filtering: PeakFilteringType | None = None

    def get_smoothed_intensities(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        """Returns the smoothed intensities of the provided spectrum, as it will be used for pick picking."""
        if self.smooth_sigma is None:
            return int_arr
        else:
            mz_mean_diff = np.mean(np.diff(mz_arr))
            sigma = self.smooth_sigma / mz_mean_diff
            return scipy.ndimage.gaussian_filter1d(int_arr, sigma=sigma)

    def pick_peaks_index(
        self,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
    ) -> NDArray[int]:
        """Picks the peaks in an intensity array and returns their indices."""
        # TODO consider removing this method in the future, since code that expects it is incompatible with
        #   interpolated peaks. the only reason to keep it would be if there is a place where we want specifically that
        int_arr_smooth = self.get_smoothed_intensities(mz_arr=mz_arr, int_arr=int_arr)
        idx_peaks, _ = scipy.signal.find_peaks(
            int_arr_smooth,
            prominence=self.min_prominence,
            distance=self.get_min_distance_indices(
                min_distance=self.min_distance, min_distance_unit=self.min_distance_unit, mz_arr=mz_arr
            ),
        )
        if self.peak_filtering is not None:
            idx_peaks = self.peak_filtering.filter_index_peaks(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, peak_idx_arr=idx_peaks
            )
        return idx_peaks

    def pick_peaks_mz(
        self,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
    ) -> NDArray[float]:
        """Picks the peaks in an intensity array and returns their m/z values."""
        idx_peaks = self.pick_peaks_index(mz_arr=mz_arr, int_arr=int_arr)
        return mz_arr[idx_peaks]

    def pick_peaks(
        self,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Picks the peaks in an intensity array and returns their m/z values and intensities."""
        idx_peaks = self.pick_peaks_index(mz_arr=mz_arr, int_arr=int_arr)
        return mz_arr[idx_peaks], int_arr[idx_peaks]

    def clone(self) -> BasicPeakPicker:
        """Creates a (deep) copy of this object."""
        return copy.deepcopy(self)

    @staticmethod
    def get_min_distance_indices(
        min_distance: float | None, min_distance_unit: str | None, mz_arr: NDArray[float]
    ) -> int | None:
        """
        Returns the minimal distance in terms of indices to use for peak picking,
        based on the configuration and the provided m/z array (as for some units it has an influence).
        :param min_distance: the minimal distance between peaks
        :param min_distance_unit: units for the minimal distance between peaks, either "index" or "mz"
        :param mz_arr: the m/z array to use for the minimal distance computation
        """
        if min_distance is None:
            return None
        elif min_distance_unit == "index":
            return min_distance
        elif min_distance_unit == "mz":
            # convert the distance into indices
            med_distance = np.median(np.diff(mz_arr))
            return max(round(min_distance / med_distance), 1)
        else:
            raise ValueError(f"Unknown min_distance_unit: {min_distance_unit}")
