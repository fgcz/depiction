from dataclasses import dataclass

import numpy as np
import scipy
import scipy.signal
from numpy.typing import NDArray

from depiction.spectrum.peak_filtering import PeakFilteringType


@dataclass
class FilterBySnrThreshold(PeakFilteringType):
    """Implements SNR threshold based on a median absolute deviation (MAD) estimate of the noise level."""
    snr_threshold: float
    window_size: int = 10

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        noise_level = self._estimate_noise_level(signal=spectrum_int_arr)
        snr = spectrum_int_arr / noise_level
        selection = snr > self.snr_threshold
        return peak_mz_arr[selection], peak_int_arr[selection]

    def _estimate_noise_level(self, signal: NDArray[float]) -> NDArray[float]:
        """Estimates the noise level in the signal using median absolute deviation (MAD)."""
        # TODO window size again should be adjustable in different units -> centralize this common functionality
        filtered_signal = scipy.signal.medfilt(signal, kernel_size=self.window_size)
        return np.abs(signal - filtered_signal)
