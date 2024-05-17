import numpy as np
from typing import Optional


class ImzmlAlignmentTracker:
    """
    Helper class that tracks whether the spectra added to a writer are aligned.
    The logic has been extracted to this dedicated class since it's fairly slow, and should be refactored
    once a native implementaiton of the imzml parser is available.
    """

    def __init__(self, first_mz_arr: Optional[np.ndarray] = None, is_aligned: bool = False):
        self._first_mz_arr = first_mz_arr
        self._is_aligned = is_aligned

    @property
    def is_aligned(self) -> bool:
        """Returns True if the spectra are aligned."""
        return self._is_aligned

    def track_mz_array(self, mz_arr: np.ndarray):
        if self._first_mz_arr is None:
            self._first_mz_arr = mz_arr
            self._is_aligned = True
        elif self._is_aligned:
            self._is_aligned = np.array_equal(self._first_mz_arr, mz_arr)
