from typing import Optional
from collections.abc import Sequence

import numpy as np

from ionplotter.parallel_ops import ParallelConfig
from ionplotter.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from ionplotter.persistence import ImzmlReader, ImzmlReadFile


class EstimatePPMError:
    """Estimates PPM error for a given imzML file, which can be used as a reference for aligning imzML files."""

    def __init__(self, parallel_config: Optional[ParallelConfig] = None) -> None:
        if parallel_config is None:
            parallel_config = ParallelConfig.no_parallelism()
        self._parallel_config = parallel_config

    def estimate(self, read_file: ImzmlReadFile) -> dict[str, float]:
        """
        Estimates the PPM error for the given imzML file.
        Returns a dictionary containing the median and std of the PPM error medians (for each spectrum).
        Additionally, mz_min and mz_max are returned which is useful when performing spectra alignment in the next step.
        """
        parallelize = ReadSpectraParallel.from_config(config=self._parallel_config)
        results = parallelize.map_chunked(read_file=read_file, operation=self._get_ppm_values)
        ppm_values = np.concatenate([r[0] for r in results])
        return {
            "ppm_median": np.nanmedian(ppm_values),
            "ppm_std": np.nanstd(ppm_values),
            "mz_min": min(r[1] for r in results),
            "mz_max": max(r[2] for r in results),
        }

    @staticmethod
    def _get_ppm_values(
        reader: ImzmlReader,
        spectra_ids: Sequence[int],
    ) -> tuple[list[float], float, float]:
        result_ppm = []
        result_min = np.inf
        result_max = -np.inf

        for i_spectrum in spectra_ids:
            mz_arr = reader.get_spectrum_mz(i_spectrum)
            if len(mz_arr) < 2:
                result_ppm.append(np.nan)
            else:
                # the median ppm for this spectrum's m/z values
                ppm_value = np.median(np.diff(mz_arr) / mz_arr[:-1]) * 1e6
                result_ppm.append(ppm_value)
                result_min = min(result_min, mz_arr[0])
                result_max = max(result_max, mz_arr[-1])

        return result_ppm, result_min, result_max

    @staticmethod
    def ppm_to_mz_values(ppm_error: float, mz_min: float, mz_max: float) -> np.ndarray:
        """
        Returns an array of m/z values, which can be used for binning the spectra.
        :param ppm_error: The estimated PPM error.
        :param mz_min: The minimum m/z value of the spectra.
        :param mz_max: The maximum m/z value of the spectra.
        """
        # Calculate the number of points in the array
        n_points = int(np.log(mz_max / mz_min) / np.log(1 + ppm_error / 1e6)) + 1

        # Generate the m/z array
        mz_array = mz_min * (1 + ppm_error / 1e6) ** np.arange(n_points)

        # Rescale the array
        actual_width = mz_array[-1] - mz_array[0]
        target_width = mz_max - mz_min
        mz_array = (mz_array - mz_array[0]) / actual_width * target_width + mz_min

        return mz_array
