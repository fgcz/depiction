import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from pydantic import BaseModel

from depiction.spectrum.peak_filtering import PeakFilteringType


# class FilterBySnrThresholdConfig(BaseModel):
#    method: Literal["FilterBySnrThreshold"] = "FilterBySnrThreshold"
#    snr_threshold: float
#    window_size: WindowSize


class FilterBySnrThresholdNewConfig(BaseModel):
    snr_threshold: float = 3.0
    n_noise_est_cutoff: int = 200


@dataclass
class FilterBySnrThresholdNew(PeakFilteringType):
    """Implements SNR threshold based on a median absolute deviation (MAD) estimate of the noise level."""

    config: FilterBySnrThresholdNewConfig

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
        snr = self.estimate_snr(spectrum_mz_arr, spectrum_int_arr)
        return snr > self.config.snr_threshold

    def estimate_snr(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        # TODO this sort of illustrates the issues that the raw spectra and the picked ones are not distinguished but
        #      also in some cases we only have access to picked spectra (and might still want to perform some picking
        #      to remove some noise)

        n_cutoff = self.config.n_noise_est_cutoff
        noise_std = np.std(int_arr[-n_cutoff:])
        noise_norm = np.sqrt(np.mean(np.abs(int_arr[-n_cutoff:])))
        scaled_noise = noise_std * np.sqrt(np.abs(int_arr)) / noise_norm
        return int_arr / scaled_noise
