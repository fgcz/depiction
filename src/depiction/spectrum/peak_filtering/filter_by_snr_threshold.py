from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy
import scipy.signal
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.spectrum.unit_conversion import WindowSize
from numpy.typing import NDArray
from pydantic import BaseModel


class FilterBySnrThresholdConfig(BaseModel):
    method: Literal["FilterBySnrThreshold"] = "FilterBySnrThreshold"
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
        selection = self._select_peaks(
            spectrum_mz_arr=spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )
        return peak_mz_arr[selection], peak_int_arr[selection]

    def filter_index_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
    ) -> NDArray[int]:
        selection = self._select_peaks(
            spectrum_mz_arr=spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_mz_arr=spectrum_mz_arr[peak_idx_arr],
            peak_int_arr=spectrum_int_arr[peak_idx_arr],
        )
        return peak_idx_arr[selection]

    def _select_peaks(
        self,
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_mz_arr: NDArray[float],
        peak_int_arr: NDArray[float],
    ) -> NDArray[bool]:
        noise_level = self._estimate_noise_level(
            signal=spectrum_int_arr, kernel_size=self.config.window_size.convert_to_index_scalar(mz_arr=spectrum_mz_arr)
        )
        peak_noise_level = np.interp(peak_mz_arr, spectrum_mz_arr, noise_level)
        eps = 1e-30
        snr = (peak_int_arr + eps) / (peak_noise_level + eps)
        return snr > self.config.snr_threshold

    @staticmethod
    def _estimate_noise_level(signal: NDArray[float], kernel_size: int) -> NDArray[float]:
        """Estimates the noise level in the signal using median absolute deviation (MAD)."""
        # Ensure kernel size is odd
        kernel_size += 1 - (kernel_size % 2)
        filtered_signal = scipy.signal.medfilt(signal, kernel_size=kernel_size)
        return np.abs(signal - filtered_signal)
