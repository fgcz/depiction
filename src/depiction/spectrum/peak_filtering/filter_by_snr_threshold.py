from dataclasses import dataclass

import numpy as np
import scipy
import scipy.signal
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.spectrum.unit_conversion import WindowSize
from numpy.typing import NDArray
from pydantic import BaseModel


class FilterBySnrThresholdConfig(BaseModel):
    snr_threshold: float
    window_size: WindowSize


@dataclass
class FilterBySnrThreshold(PeakFilteringType):
    """Implements SNR threshold based on a median absolute deviation (MAD) estimate of the noise level."""

    config: FilterBySnrThresholdConfig

    def filter_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> tuple[NDArray[float], NDArray[float]]:
        noise_level = self._estimate_noise_level(
            signal=spectrum_int_arr, kernel_size=self.config.window_size.convert_to_index_scalar(mz_arr=spectrum_mz_arr)
        )
        snr = spectrum_int_arr / noise_level
        selection = snr > self.config.snr_threshold
        return peak_mz_arr[selection], peak_int_arr[selection]

    @staticmethod
    def _estimate_noise_level(signal: NDArray[float], kernel_size: int) -> NDArray[float]:
        """Estimates the noise level in the signal using median absolute deviation (MAD)."""
        filtered_signal = scipy.signal.medfilt(signal, kernel_size=kernel_size)
        return np.abs(signal - filtered_signal)
