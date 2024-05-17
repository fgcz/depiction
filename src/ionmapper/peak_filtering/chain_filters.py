from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    from ionmapper.peak_filtering import PeakFilteringType


@dataclass
class ChainFilters:
    """Evaluates a sequential chain of several peak filters."""

    filters: list[PeakFilteringType]

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        for filter_fn in self.filters:
            peak_idx_arr = filter_fn.filter_index_peaks(spectrum_mz_arr, spectrum_int_arr, peak_idx_arr)
        return peak_idx_arr

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        for filter_fn in self.filters:
            peak_mz_arr, peak_int_arr = filter_fn.filter_peaks(
                spectrum_mz_arr, spectrum_int_arr, peak_mz_arr, peak_int_arr
            )
        return peak_mz_arr, peak_int_arr
