from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from depiction.spectrum.peak_filtering import PeakFilteringType


@dataclass
class ChainFilters:
    """Evaluates a sequential chain of several peak filters."""

    filters: list[PeakFilteringType]

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_idx_arr: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        for filter_fn in self.filters:
            peak_idx_arr = filter_fn.filter_index_peaks(spectrum_mz_arr, spectrum_int_arr, peak_idx_arr)
        return peak_idx_arr

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_mz_arr: NDArray[np.float64],
        peak_int_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        for filter_fn in self.filters:
            peak_mz_arr, peak_int_arr = filter_fn.filter_peaks(
                spectrum_mz_arr, spectrum_int_arr, peak_mz_arr, peak_int_arr
            )
        return peak_mz_arr, peak_int_arr
