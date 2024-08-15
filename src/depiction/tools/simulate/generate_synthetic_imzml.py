from pathlib import Path

import cyclopts
import numpy as np
import polars as pl
from numpy.typing import NDArray
from perlin_noise import PerlinNoise

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence import ImzmlWriteFile, ImzmlModeEnum
from depiction.persistence.types import GenericWriteFile


class GenerateSyntheticImzml:
    def __init__(self, height: int, width: int, rng: np.random.Generator, mz_min: float, mz_max: float) -> None:
        self._height = height
        self._width = width
        self._rng = rng
        self._mz_min = mz_min
        self._mz_max = mz_max

    def generate_shift_map(self, mean: float = 0.3, std: float = 0.1) -> NDArray[float]:
        """Generates a mass shift map"""
        seed = round(self._rng.uniform() * 1e12)
        noise_gen = PerlinNoise(octaves=6, seed=seed)
        noise_2d = np.asarray(
            [[noise_gen([i / self._width, j / self._height]) for j in range(self._width)] for i in range(self._height)]
        )
        # adjust the values to the desired mean and std
        return (noise_2d - noise_2d.mean()) / noise_2d.std() * std + mean

    def get_shifted_target_masses(
        self, mass_list: NDArray[float], shift: float, std_noise: float = 0.001
    ) -> NDArray[float]:
        """Given a mean shift returns a shifted mass list for a particular pixel, with some additional normal noise on
        the masses.
        """
        return mass_list + shift + self._rng.normal(scale=std_noise, size=mass_list.shape)

    def generate_centroided_spectrum(
        self, target_masses: NDArray[float], target_intensities: NDArray[float], snr: float = 3.0, n_isotopes: int = 3
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Generates a centroided spectrum with the given target masses and intensities,
        scaled relative to the maximal value in target_intensities.
        """
        target_mz, target_int = self._generate_isotopic_peaks(
            mz_arr=target_masses, int_arr=target_intensities, n_isotopes=n_isotopes
        )
        n_noise_peaks = round((self._mz_max - self._mz_min))
        noise_mz, noise_int = self._generate_noise_peaks(
            n=n_noise_peaks, target_mz=target_mz, target_min_distance_mz=0.5
        )
        mz_arr = np.concatenate([target_mz, noise_mz])
        int_arr = np.concatenate([target_int * snr, noise_int])
        idx = np.argsort(mz_arr)
        return mz_arr[idx], int_arr[idx]

    def write_file(self, labels: MultiChannelImage, target_masses: NDArray[float], output: GenericWriteFile):
        # input validation
        if labels.n_channels != len(target_masses):
            msg = f"Number of channels in labels ({labels.n_channels}) does not match the number of target masses ({len(target_masses)})"
            raise ValueError(msg)
        if labels.dimensions != (self._width, self._height):
            msg = f"Dimensions of labels ({labels.dimensions}) do not match the dimensions of the synthetic image ({self._width}, {self._height})"
            raise ValueError(msg)

        # collect the relevant input information
        shift_map = self.generate_shift_map()
        label_map = labels.data_spatial.values

        # actually create the file
        with output.writer() as writer:
            for i_spectrum, coordinates in enumerate(np.ndindex(self._height, self._width)):
                shift_value = shift_map[coordinates]
                target_mz_i = self.get_shifted_target_masses(target_masses, shift=shift_value)
                mz_arr, int_arr = self.generate_centroided_spectrum(target_mz_i, label_map[coordinates])
                writer.add_spectrum(mz_arr, int_arr, coordinates)

    def _generate_isotopic_peaks(
        self, mz_arr: NDArray[float], int_arr: NDArray[float], n_isotopes: int
    ) -> tuple[NDArray[float], NDArray[float]]:
        n_peaks = len(mz_arr)
        result_mz = np.zeros(n_isotopes * n_peaks)
        result_int = np.zeros(n_isotopes * n_peaks)
        int_max = int_arr.max()

        for i_peak in range(n_peaks):
            for i_isotope in range(n_isotopes):
                i_result = i_peak * n_isotopes + i_isotope
                result_mz[i_result] = mz_arr[i_peak] + i_isotope * 1.00235
                result_int[i_result] = int_arr[i_peak] / int_max * (1 / (i_isotope + 1))

        return result_mz, result_int

    def _generate_noise_peaks(
        self, n: int, target_mz: NDArray[float], target_min_distance_mz: float
    ) -> tuple[NDArray[float], NDArray[float]]:
        mz_arr = np.linspace(self._mz_min, self._mz_max, n)
        int_arr = self._rng.uniform(0, 1, n)
        baseline = np.exp(np.linspace(3, 0, len(mz_arr))) / np.exp(3)
        noise_mz, noise_int = mz_arr, (int_arr + baseline) * 0.5

        # clear the ones which are too close to a target
        sel_idx = np.ones(n, dtype=bool)
        for target in target_mz:
            sel_idx &= np.abs(noise_mz - target) > target_min_distance_mz
        return noise_mz[sel_idx], noise_int[sel_idx]


app = cyclopts.App()


@app.default
def generate_imzml(
    label_image_file: Path,
    mass_list_file: Path,
    output_file: Path,
):
    label_img = MultiChannelImage.read_hdf5(label_image_file)
    panel_df = pl.read_csv(mass_list_file)

    gen = GenerateSyntheticImzml(
        height=label_img.dimensions[1],
        width=label_img.dimensions[0],
        rng=np.random.default_rng(),
        mz_min=panel_df["mass"].min(),
        mz_max=panel_df["mass"].max(),
    )
    write_file = ImzmlWriteFile(output_file, imzml_mode=ImzmlModeEnum.PROCESSED)
    gen.write_file(labels=label_img, target_masses=panel_df["mass"].to_numpy(), output=write_file)


if __name__ == "__main__":
    app()
