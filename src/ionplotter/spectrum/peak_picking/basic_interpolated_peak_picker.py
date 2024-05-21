from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy

from ionplotter.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ionplotter.spectrum.peak_filtering import PeakFilteringType


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
        local_maxima_indices, _ = scipy.signal.find_peaks(
            int_arr,
            prominence=self.min_prominence,
            distance=BasicPeakPicker.get_min_distance_indices(
                min_distance=self.min_distance, min_distance_unit=self.min_distance_unit, mz_arr=mz_arr
            ),
        )

        # setup result
        # TODO preallocate in the future
        peak_mz = []
        peak_int = []

        # for each of the local maxima find the interpolated maximum
        for local_max_index in local_maxima_indices:
            if local_max_index == 0 or local_max_index == len(mz_arr) - 1:
                # TODO should we handle these, and if yes: how?
                continue

            # Select points for interpolation problem
            interp_indices = [local_max_index - 1, local_max_index, local_max_index + 1]
            # Note: using 64 bit precision is essential, as there can be numerical issues with 32 bit floats
            interp_mz = mz_arr[interp_indices].astype(float)
            interp_int = int_arr[interp_indices].astype(float)

            # Fit a cubic spline
            spline = scipy.interpolate.CubicSpline(interp_mz, interp_int)
            roots = spline.derivative().roots()

            if len(roots) == 0:
                # TODO error handling (also in the clause below)
                print(f"Error: {len(roots)} roots found for local maximum at index {local_max_index}")
                print(f"{interp_mz=}, {interp_int=}, {roots=}")
            else:
                # evaluate the value of the spline at the root
                mz_max = min(roots, key=lambda r: abs(r - mz_arr[local_max_index]))
                int_max = spline(mz_max)

                # store the result
                peak_mz.append(mz_max)
                peak_int.append(int_max)

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
