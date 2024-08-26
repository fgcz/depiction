# TODO extremely experimental, not part of pyproject.toml dependencies etc,
#   usage not yet recommended!
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import findmfpy

from depiction.spectrum.peak_filtering import PeakFilteringType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class FindMFPeakPicker:
    resolution: float = 10000.0
    width: float = 2.0
    int_width: float = 2.0
    int_threshold: float = 10.0
    area: bool = True
    max_peaks: int = 0
    peak_filtering: PeakFilteringType | None = None

    def pick_peaks(
        self, mz_arr: NDArray[np.float64], int_arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        peak_mz_arr, peak_int_arr = findmfpy.pick_peaks(
            mz_arr,
            int_arr,
            resolution=self.resolution,
            width=self.width,
            int_width=self.int_width,
            int_threshold=self.int_threshold,
            area=self.area,
            max_peaks=self.max_peaks,
        )
        if self.peak_filtering is not None:
            return self.peak_filtering.filter_peaks(
                spectrum_mz_arr=mz_arr,
                spectrum_int_arr=int_arr,
                peak_mz_arr=peak_mz_arr,
                peak_int_arr=peak_int_arr,
            )
        else:
            return peak_mz_arr, peak_int_arr
