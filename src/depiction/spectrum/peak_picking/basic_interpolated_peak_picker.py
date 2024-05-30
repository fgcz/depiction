from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import loguru
import numpy as np
import scipy

from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from depiction.spectrum.peak_filtering import PeakFilteringType


@dataclass
class BasicInterpolatedPeakPicker:
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: str | None = None
    peak_filtering: PeakFilteringType | None = None

    # TODO note unlike the BasicPeakPicker, this one does not support pick_peaks_index and is something that might have
    #      to be considered in the rest of the code before it can be used
    def pick_peaks(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
        # find local maxima
        local_maxima_indices = self._find_local_maxima_indices(mz_arr=mz_arr, int_arr=int_arr)

        # remove 0 and -1 if they are present, since they are currently unhandled
        local_maxima_indices = np.setdiff1d(local_maxima_indices, [0, len(mz_arr) - 1])

        # for each of the local maxima, find the interpolated maximum
        peaks_interp = [
            self._interpolate_max_mz_and_intensity(local_max_index, mz_arr, int_arr)
            for local_max_index in local_maxima_indices
        ]
        peaks_interp = [peak for peak in peaks_interp if peak[0] is not None and peak[1] is not None]
        peak_mz, peak_int = zip(*peaks_interp)
        peak_mz = np.asarray(peak_mz)
        peak_int = np.asarray(peak_int)

        if self.peak_filtering is not None:
            peak_mz, peak_int = self.peak_filtering.filter_peaks(
                spectrum_mz_arr=mz_arr,
                spectrum_int_arr=int_arr,
                peak_mz_arr=peak_mz,
                peak_int_arr=peak_int,
            )

        return peak_mz, peak_int

    def _interpolate_max_mz_and_intensity(
        self, local_max_index: int, mz_arr: NDArray[float], int_arr: NDArray[float]
    ) -> tuple[float, float] | tuple[None, None]:
        """Returns the interpolated m/z and intensity of a local maximum index.
        :param local_max_index: index of the local maximum in the mz_arr and int_arr arrays
        :param mz_arr: m/z values of the spectrum
        :param int_arr: intensity values of the spectrum
        :return: interpolated m/z and intensity of the local maximum
        """
        # Select points for interpolation problem
        interp_indices = [local_max_index - 1, local_max_index, local_max_index + 1]
        # Note: using 64 bit precision is essential, as there can be numerical issues with 32 bit floats
        interp_mz = mz_arr[interp_indices].astype(float)
        interp_int = int_arr[interp_indices].astype(float)
        # Fit a cubic spline
        spline = scipy.interpolate.CubicSpline(interp_mz, interp_int)
        roots = spline.derivative().roots()
        if len(roots) == 0:
            loguru.logger.warning(
                f"Error: {len(roots)} roots found for local maximum at index {local_max_index}; "
                f"{interp_mz=}, {interp_int=}, {roots=}"
            )
            return None, None
        else:
            # evaluate the value of the spline at the root
            mz_max = min(roots, key=lambda r: abs(r - mz_arr[local_max_index]))
            int_max = spline(mz_max)
            return mz_max, int_max

    def _find_local_maxima_indices(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[int]:
        """Returns the indices of local maxima in the intensity array, respecting the `min_distance` parameter."""
        local_maxima_indices, _ = scipy.signal.find_peaks(
            int_arr,
            prominence=self.min_prominence,
            distance=BasicPeakPicker.get_min_distance_indices(
                min_distance=self.min_distance, min_distance_unit=self.min_distance_unit, mz_arr=mz_arr
            ),
        )
        return local_maxima_indices
