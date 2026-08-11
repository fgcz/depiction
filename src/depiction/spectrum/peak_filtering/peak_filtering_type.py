from typing import Protocol
from numpy.typing import NDArray

import numpy as np


class PeakFilteringType(Protocol):
    """Selects a subset of already-picked peaks.

    Both methods return the surviving peaks in ascending m/z order -- the order they were passed
    in. Callers write the result straight to imzML, which requires it.
    """

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_idx_arr: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        raise NotImplementedError

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[np.float64],
        spectrum_int_arr: NDArray[np.float64],
        peak_mz_arr: NDArray[np.float64],
        peak_int_arr: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError
