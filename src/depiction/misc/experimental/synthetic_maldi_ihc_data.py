from collections.abc import Sequence

import numpy as np
import scipy
from numpy.typing import NDArray
from tqdm import tqdm
from xarray import DataArray

from depiction.estimate_ppm_error import EstimatePPMError
from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence import ImzmlWriteFile


class SyntheticMaldiIhcData:
    """Helper that creates synthetic MALDI IHC data."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_label_image_circles(
        self,
        n_labels: int,
        image_height: int,
        image_width: int,
        radius_mean: float = 15,
        radius_std: float = 5,
    ) -> MultiChannelImage:
        """Generates a label image with a circle for each specified label.
        Will generate a full rectangular image.
        :param n_labels: The number of labels to generate.
        :param image_height: The height of the image.
        :param image_width: The width of the image.
        :param radius_mean: The mean radius of the circles (drawn from a normal distribution).
        :param radius_std: The standard deviation of the radius of the circles (drawn from a normal distribution).
        """
        label_image = np.zeros((image_height, image_width, n_labels))

        for i_label in range(n_labels):
            center_h = self.rng.uniform(0, image_height)
            center_w = self.rng.uniform(0, image_width)
            radius = self.rng.normal(radius_mean, radius_std)

            for h in range(image_height):
                for w in range(image_width):
                    distance = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
                    if distance < radius:
                        label_image[h, w, i_label] = 1

        data = DataArray(label_image, dims=("y", "x", "c"), coords={"c": [f"synthetic_{i}" for i in range(n_labels)]})
        data["bg_value"] = 0.
        return MultiChannelImage(data)

    def generate_imzml_for_labels(
        self,
        write_file: ImzmlWriteFile,
        label_image: MultiChannelImage,
        label_masses: NDArray[float],
        n_isotopes: int,
        bin_width_ppm: float = 50,
        baseline_strength: float = 2.0,
        background_noise_strength: float = 0.05,
    ) -> None:
        mz_arr = self.get_mz_arr(round(min(label_masses) - 10), round(max(label_masses) + 10), bin_width_ppm)
        with write_file.writer() as writer:
            # TODO reimplement this by iterating over the xarray directly if possible
            # sparse_values = label_image.sparse_values
            # sparse_coordinates = label_image.sparse_coordinates
            flat_data_array = label_image.data_flat
            sparse_coordinates = flat_data_array.coords["c"].values

            for i, (x, y) in tqdm(enumerate(sparse_coordinates), total=len(sparse_coordinates)):
                labels = flat_data_array.sel(i=i).values
                masses = label_masses[labels > 0]
                intensities = labels[labels > 0]
                int_arr = self.generate_single_spectrum(
                    peak_masses=masses,
                    peak_intensities=intensities,
                    mz_arr=mz_arr,
                    n_isotopes=n_isotopes,
                    baseline_strength=baseline_strength,
                    background_noise_strength=background_noise_strength,
                )
                writer.add_spectrum(mz_arr, int_arr, (x, y, 1))

    def generate_single_spectrum(
        self,
        peak_masses: Sequence[float],
        peak_intensities: Sequence[float],
        mz_arr: NDArray[float],
        n_isotopes: int = 1,
        baseline_strength: float = 2.0,
        background_noise_strength: float = 0.05,
    ) -> NDArray[float]:
        int_arr = np.zeros_like(mz_arr)

        # add bg noise (TODO this is not really that maldi like yet)
        int_arr += scipy.ndimage.gaussian_filter1d(self.rng.uniform(0, 1, len(mz_arr)), 2) * background_noise_strength

        # add baseline
        int_arr += mz_arr ** (-0.3) * baseline_strength

        def next_peak(peak_mz: float) -> float:
            # corresponds to a gaussian with a FWHM of 0.1
            return np.exp(-(((mz_arr - peak_mz) / 0.1) ** 2)) * self.rng.uniform(0.7, 1)

        # add the actual masses
        for peak_mass, peak_intensity in zip(peak_masses, peak_intensities):
            for i_isotope in range(n_isotopes):
                scale = 1 / (i_isotope + 1)
                int_arr += next_peak(peak_mass + i_isotope * 1.00235) * scale * peak_intensity

        return int_arr

    def get_mz_arr(self, min_mass: float, max_mass: float, bin_width_ppm: float) -> NDArray[float]:
        return EstimatePPMError.ppm_to_mz_values(bin_width_ppm, mz_min=min_mass, mz_max=max_mass)

    @staticmethod
    def generate_diagonal_stripe_pattern(
        image_height: int, image_width: int, bandwidth: float, rotation: float = 45.0, phase: float = 0.0
    ) -> NDArray[float]:
        """Generates a diagonal stripe pattern.
        Values are in the range [0, 1].
        :param image_height: The height of the image.
        :param image_width: The width of the image.
        :param bandwidth: The bandwidth of the sine wave, i.e. after this many pixels (unrotated) the pattern repeats.
        :param rotation: The rotation of the pattern in degrees (by default 45 degrees).
        :param phase: The phase of the sine wave, can be used to shift the pattern (periodicity of 360 degrees).
        """

        def f(x, y):
            return np.sin(y / bandwidth * 2 * np.pi + np.radians(phase))

        data = np.zeros((image_height, image_width))
        phi = np.radians(rotation)
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        for i in range(image_height):
            for j in range(image_width):
                i_rot, j_rot = np.dot(rot, [i, j])
                data[i, j] = (f(i_rot, j_rot) + 1) / 2

        return data
