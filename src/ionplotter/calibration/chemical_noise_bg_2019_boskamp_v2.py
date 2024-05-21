import math
from typing import Any, Optional, Literal

import matplotlib
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn
from numpy.typing import NDArray

from ionplotter.parallel_ops import ParallelConfig, WriteSpectraParallel
from ionplotter.persistence import ImzmlWriteFile, ImzmlReadFile
import scipy.stats

# TODO experimental/non-prod


class ChemicalNoiseCalibration:
    """An implementation/code of Boskamp et al. 2019 paper.

    In a quick evaluation it seems to perform similarly to the existing targeted mass calibration in terms of aligning
    the spectra among each other, however it suffers from the problem that errors >0.5 Da are not necessarily correctly
    accounted for.
    A potential compromise is to first evaluate the median error and only thereafter perform this calibration, however
    to do so requires again at least a part of the targeted calibration.
    """

    _lambda_averagine = 1.0 + 4.95e-4

    def __init__(
        self,
        n_mass_intervals: int,
        interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"],
        parallel_config: ParallelConfig,
        use_ppm_space: bool,
    ) -> None:
        self._n_mass_intervals = n_mass_intervals
        self._interpolation_mode = interpolation_mode
        self._parallel_config = parallel_config
        self._use_ppm_space = use_ppm_space

    @property
    def n_mass_intervals(self) -> int:
        return self._n_mass_intervals

    def get_peptide_mass_from_nominal(self, mass_nominal: float) -> float:
        return self._lambda_averagine * mass_nominal

    def get_kendrick_shift(self, exact_mass: float) -> float:
        nominal_mass = exact_mass / self._lambda_averagine
        return nominal_mass - math.floor(nominal_mass + 0.5)

    def plot_kendrick_shift(
        self,
        peak_mz_arr: NDArray[float],
        unit: Literal["m/z", "ppm"] = "m/z",
        ax: Optional[matplotlib.axes.Axes] = None,
        scatter_kwargs: Optional[dict[str, Any]] = None,
        robust_regression: bool = True,
    ) -> None:
        scatter_kwargs = {"s": 1} | (scatter_kwargs or {})
        kendrick_shifts = np.array([self.get_kendrick_shift(mz) for mz in peak_mz_arr])

        data = pd.DataFrame({"m/z": peak_mz_arr, "Kendrick shift (m/z)": kendrick_shifts})
        data["Kendrick shift (ppm)"] = kendrick_shifts * 1e6 / peak_mz_arr

        plot_y = "Kendrick shift (m/z)" if unit == "m/z" else "Kendrick shift (ppm)"
        seaborn.regplot(
            data=data,
            x="m/z",
            y=plot_y,
            ax=ax,
            scatter_kws=scatter_kwargs,
            robust=robust_regression,
        )

    def _get_mz_partitions(self, mz_arr: NDArray[float]) -> tuple[list[NDArray[int]], NDArray[float]]:
        """Returns the indices of the partitions and the center of the partitions."""
        # TODO if this class gets used in production, this method should be refactored in a way that it doesn't have to
        #      be called so often - for testing this is now fine
        indices_list = np.array_split(np.arange(len(mz_arr)), self._n_mass_intervals)
        mz_center = np.array([(mz_arr[indices[-1]] + mz_arr[indices[0]]) / 2 for indices in indices_list], dtype=float)
        return indices_list, mz_center

    def get_moments_approximation(
        self, mz_arr: NDArray[float], int_arr: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float], NDArray[complex]]:
        """Returns the shifts, dispersion and moments approximation for the given spectrum.
        The moments are a complex quantity from which the former two are calculated for convenience.
        """
        # formula (4) in the paper
        moments_arr = np.zeros(self._n_mass_intervals, dtype=complex)
        partition_indices_list, _ = self._get_mz_partitions(mz_arr=mz_arr)
        for k, partition_indices in enumerate(partition_indices_list):
            normalization_approx = np.sum(int_arr[partition_indices])
            integral_approx = np.sum(
                int_arr[partition_indices] * np.exp(2 * np.pi * 1j * mz_arr[partition_indices] / self._lambda_averagine)
            )
            moments_arr[k] = integral_approx / normalization_approx

        # formula (5) in the paper
        shifts_arr = np.angle(moments_arr) / (2 * np.pi)
        dispersion_arr = np.abs(moments_arr)
        return shifts_arr, dispersion_arr, moments_arr

    def interpolate_shifts(self, mz_arr: NDArray[float], shifts_arr: NDArray[float]) -> NDArray[float]:
        _, partition_center_mz = self._get_mz_partitions(mz_arr=mz_arr)
        if self._interpolation_mode == "linear":
            interpolate_shift = np.interp(mz_arr, partition_center_mz, shifts_arr)
        elif self._interpolation_mode == "cubic_spline":
            tck = scipy.interpolate.splrep(partition_center_mz, shifts_arr, s=0, k=3)
            interpolate_shift = scipy.interpolate.splev(mz_arr, tck, der=0)
        elif self._interpolation_mode == "refit_linear":
            mz_center = self._get_mz_partitions(mz_arr)[1]
            fit = scipy.stats.siegelslopes(shifts_arr, mz_center)
            interpolate_shift = fit.slope * mz_arr + fit.intercept
        else:
            raise ValueError(f"Unknown interpolation mode: {self._interpolation_mode}")
        return interpolate_shift

    def align_masses(
        self,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
    ) -> NDArray[float]:
        """Aligns the masses of the given spectrum, by using interpolated shift estimates as in the paper.
        :param mz_arr: m/z values of the spectrum
        :param int_arr: intensity values of the spectrum
        """
        # the approach from the paper is to partition the shifts (partition_center_mz, shifts_arr)
        # to the full range and then use these to align the masses
        # we can use the partition_indices_mz_centers to align the masses
        shifts_arr, _, _ = self.get_moments_approximation(mz_arr=mz_arr, int_arr=int_arr)
        if self._use_ppm_space:
            mz_center = self._get_mz_partitions(mz_arr=mz_arr)[1]
            shifts_arr = shifts_arr / mz_center * 1e6
        interpolate_shift = self.interpolate_shifts(mz_arr=mz_arr, shifts_arr=shifts_arr)
        if self._use_ppm_space:
            interpolate_shift = interpolate_shift / 1e6 * mz_arr
        return mz_arr - interpolate_shift

    def align_masses_all(
        self,
        read_file: ImzmlReadFile,
        write_file: ImzmlWriteFile,
    ) -> None:
        """Applies `align_masses` to all spectra in the given file and writes the results to the output file."""
        parallelize = WriteSpectraParallel.from_config(self._parallel_config)

        def chunk_operation(reader, spectra_indices, writer) -> None:
            for spectrum_id in spectra_indices:
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                mz_arr = self.align_masses(mz_arr, int_arr)
                writer.add_spectrum(mz_arr, int_arr, reader.coordinates[spectrum_id])

        parallelize.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=chunk_operation,
        )
