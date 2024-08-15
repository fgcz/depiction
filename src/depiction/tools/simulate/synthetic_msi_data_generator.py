# TODO use generate_synthetic_imzml.py instead

from collections.abc import Sequence

import numpy as np
import scipy
import xarray
from numpy.typing import NDArray

from depiction.estimate_ppm_error import EstimatePPMError
from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence import ImzmlWriteFile, ImzmlWriter


# TODO configure shifts


class SyntheticMSIDataGenerator:
    """Helper that creates synthetic MSI data."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_imzml_for_labels(
        self,
        write_file: ImzmlWriteFile,
        label_image: MultiChannelImage,
        label_masses: NDArray[float],
        n_isotopes: int,
        mz_arr: NDArray[float],
        baseline_max_intensity: float = 2.0,
        background_noise_strength: float = 0.05,
    ) -> None:
        with write_file.writer() as writer:
            xarray.apply_ufunc(
                self._generate_and_write_single_spectrum,
                label_image.data_flat,
                label_image.data_flat.x.values,
                label_image.data_flat.y.values,
                kwargs={
                    "label_masses": label_masses,
                    "mz_arr": mz_arr,
                    "n_isotopes": n_isotopes,
                    "baseline_max_intensity": baseline_max_intensity,
                    "background_noise_strength": background_noise_strength,
                    "writer": writer,
                },
                input_core_dims=[["c"], [], []],
                output_core_dims=[["c"]],
                vectorize=True,
            )

    def _generate_and_write_single_spectrum(
        self,
        labels: NDArray[int],
        x: int,
        y: int,
        label_masses: NDArray[float],
        mz_arr: NDArray[float],
        n_isotopes: int,
        baseline_max_intensity: float,
        background_noise_strength: float,
        writer: ImzmlWriter,
    ) -> None:
        masses = label_masses[labels > 0]
        intensities = labels[labels > 0]
        int_arr = self.generate_single_spectrum(
            peak_masses=masses,
            peak_intensities=intensities,
            mz_arr=mz_arr,
            n_isotopes=n_isotopes,
            baseline_max_intensity=baseline_max_intensity,
            background_noise_strength=background_noise_strength,
        )
        writer.add_spectrum(mz_arr, int_arr, (x, y, 1))
        # TODO not sure how to avoid returning something here
        return np.zeros_like(labels)

    def generate_single_spectrum(
        self,
        peak_masses: Sequence[float],
        peak_intensities: Sequence[float],
        mz_arr: NDArray[float],
        n_isotopes: int = 1,
        baseline_max_intensity: float = 2.0,
        background_noise_strength: float = 0.05,
    ) -> NDArray[float]:
        int_arr = np.zeros_like(mz_arr)

        # add bg noise (TODO this is not really that maldi like yet)
        int_arr += scipy.ndimage.gaussian_filter1d(self.rng.uniform(0, 1, len(mz_arr)), 2) * background_noise_strength

        # add baseline
        int_arr += self.get_baseline(n_points=len(mz_arr), max_intensity=baseline_max_intensity)

        def next_peak(peak_mz: float) -> float:
            # corresponds to a gaussian with a FWHM of 0.1
            return np.exp(-(((mz_arr - peak_mz) / 0.1) ** 2)) * self.rng.uniform(0.7, 1)

        # add the actual masses
        for peak_mass, peak_intensity in zip(peak_masses, peak_intensities):
            for i_isotope in range(n_isotopes):
                scale = 1 / (i_isotope + 1)
                int_arr += next_peak(peak_mass + i_isotope * 1.00235) * scale * peak_intensity

        return int_arr

    @staticmethod
    def get_mz_arr(min_mass: float, max_mass: float, bin_width_ppm: float) -> NDArray[float]:
        return EstimatePPMError.ppm_to_mz_values(bin_width_ppm, mz_min=min_mass, mz_max=max_mass)

    @staticmethod
    def get_baseline(n_points: int, max_intensity: float = 2.0) -> NDArray[float]:
        # TODO the current implementation is stupid as it does not take into account the actual mz values
        return np.exp(np.linspace(3, 0, n_points)) / np.exp(3) * max_intensity
