# TODO better name

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ms_peak_picker
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from depiction.spectrum.peak_filtering import PeakFilteringType


@dataclass
class MSPeakPicker:
    """Interfaces to the ms-peak-picker Python package.
    TODO proper description and better name
    """

    fit_type: str = "quadratic"
    peak_filtering: PeakFilteringType | None = None

    def pick_peaks(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> tuple[NDArray[float], NDArray[float]]:
        peak_list = ms_peak_picker.pick_peaks(mz_arr, int_arr, fit_type=self.fit_type)
        peak_mz = np.array([peak.mz for peak in peak_list])
        peak_int = np.array([peak.intensity for peak in peak_list])

        if self.peak_filtering is not None and len(peak_mz) > 0:
            peak_mz, peak_int = self.peak_filtering.filter_peaks(
                spectrum_mz_arr=mz_arr,
                spectrum_int_arr=int_arr,
                peak_mz_arr=peak_mz,
                peak_int_arr=peak_int,
            )

        return peak_mz, peak_int
