# TODO extremely experimental, not part of pyproject.toml dependencies etc,
#   usage not yet recommended!
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import findmfpy
from numpy.typing import NDArray


@dataclass
class FindMFPeakpicker:
    resolution: float = 10000.0
    width: float = 2.0
    int_width: float = 2.0
    int_threshold: float = 10.0
    area: bool = True
    max_peaks: int = 0

    def pick_peaks(
        self, mz_arr: NDArray[np.float64], int_arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return findmfpy.pick_peaks(
            mz_arr,
            int_arr,
            resolution=self.resolution,
            width=self.width,
            int_width=self.int_width,
            int_threshold=self.int_threshold,
            area=self.area,
            max_peaks=self.max_peaks,
        )
