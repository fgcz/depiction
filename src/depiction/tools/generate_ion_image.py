import argparse
from collections.abc import Sequence
from pprint import pprint
from typing import Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlReader
from depiction.tools.image_channel_statistics import ImageChannelStatistics


class GenerateIonImage:
    """Generates ion images from an imzML file.

    The generate_*_for_file methods use an `ImzmlReadFile` as input, and take a list of selections of what to plot.
    If multiple selections should be plotted, the class is designed to avoid reading the data multiple times, and thus
    it can be expected to be faster than calling the generate_*_for_file methods multiple times.
    """

    def __init__(self, parallel_config: ParallelConfig) -> None:
        # TODO for peak picked data, it could be worth considering an option to only select the closest peak in case
        #      multiple peaks would fall within the window of interest
        self._parallel_config = parallel_config

    def generate_ion_images_for_file(
        self,
        input_file: ImzmlReadFile,
        mz_values: Sequence[float],
        tol: float | Sequence[float],
        channel_names: Optional[list[str]] = None,
    ) -> MultiChannelImage:
        """
        Generates an ion image for each of the provided mz values, and returns a multi-channel `SparseImage2d`.
        Multiple peaks will be summed to obtain the intensity.
        :param input_file: the input file
        :param mz_values: the mz values
        :param tol: the tolerance, for the m/z readout
        :param channel_names: the names of the channels, if None, the channels will be numbered
        """
        channel_values = self._generate_channel_values(input_file=input_file, mz_values=mz_values, tol=tol)
        data = (
            channel_values.assign_coords(
                c=channel_names,
                x=("i", input_file.coordinates_2d[:, 0]),
                y=("i", input_file.coordinates_2d[:, 1]),
            )
            .set_xindex(["y", "x"])
            .unstack("i")
        )
        return MultiChannelImage(data)

    def _generate_channel_values(
        self, input_file: ImzmlReadFile, mz_values: Sequence[float], tol: float | Sequence[float]
    ) -> DataArray:
        if np.isscalar(tol):
            tol = [tol] * len(mz_values)
        parallelize = ReadSpectraParallel.from_config(self._parallel_config)
        array = parallelize.map_chunked(
            read_file=input_file,
            operation=self._compute_channels_chunk,
            bind_args=dict(mz_values=mz_values, tol_values=tol),
            reduce_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )
        return DataArray(array, dims=("i", "c"), attrs={"bg_value": np.nan})

    def generate_range_images_for_file(
        self,
        input_file: ImzmlReadFile,
        mz_ranges: list[tuple[float, float]],
        channel_names: Optional[list[str]] = None,
    ) -> MultiChannelImage:
        """Generates an image for each of the provided mz ranges, and returns a multi-channel `SparseImage2d` with
        the summed intensities.
        :param input_file: the input file
        :param mz_ranges: the mz ranges
        :param channel_names: the names of the channels, if None, the channels will be numbered
        """
        parallelize = ReadSpectraParallel.from_config(self._parallel_config)
        channel_values = parallelize.map_chunked(
            read_file=input_file,
            operation=self._compute_for_mz_ranges,
            bind_args=dict(mz_ranges=mz_ranges),
            reduce_fn=lambda chunks: np.concatenate(chunks, axis=0),
        )
        return MultiChannelImage.from_numpy_sparse(
            values=channel_values,
            coordinates=input_file.coordinates_2d,
            channel_names=channel_names,
            # TODO clarfiy (see above)
            bg_value=np.nan,
        )

    @classmethod
    def _compute_channels_chunk(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        mz_values: Sequence[float],
        tol_values: Sequence[float],
    ) -> NDArray[float]:
        return cls._compute_for_mz_ranges(
            reader=reader,
            spectra_ids=spectra_ids,
            mz_ranges=[(mz - tol, mz + tol) for mz, tol in zip(mz_values, tol_values)],
        )

    @staticmethod
    def _compute_for_mz_ranges(
        reader: ImzmlReader,
        spectra_ids: list[int],
        mz_ranges: list[tuple[float, float]],
    ) -> NDArray[float]:
        """Generates summed intensities for the provided mz ranges.
        :param reader: the reader
        :param spectra_ids: the spectra ids to process, i.e. for parallelization
        :param mz_ranges: the mz ranges to search (inclusive TODO check this)
        :return: a 2d array with the summed intensities, with shape (n_spectra, n_ranges)
        """
        n_spectra = len(spectra_ids)
        n_ranges = len(mz_ranges)
        result = np.zeros((n_spectra, n_ranges), dtype=float)
        for i_spectrum, spectrum_id in enumerate(spectra_ids):
            for i_range, (mz_min, mz_max) in enumerate(mz_ranges):
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                mz_begin = np.searchsorted(mz_arr, mz_min, side="left")
                mz_end = np.searchsorted(mz_arr, mz_max, side="right")
                result[i_spectrum, i_range] = np.sum(int_arr[mz_begin:mz_end])
        return result


def main_generate_ion_image(
    input_imzml_path: str, mass_list_path: str, output_hdf5: str, n_jobs: int, output_stats_path: Optional[str]
) -> None:
    parallel_config = ParallelConfig(n_jobs=n_jobs, task_size=None)
    gen_image = GenerateIonImage(parallel_config=parallel_config)
    mass_list_df = pl.read_csv(mass_list_path)

    image = gen_image.generate_ion_images_for_file(
        input_file=ImzmlReadFile(input_imzml_path),
        mz_values=mass_list_df["mass"],
        tol=mass_list_df["tol"],
        channel_names=mass_list_df["label"],
    )
    xarray = image.to_dense_xarray(bg_value=np.nan)
    xarray.to_netcdf(output_hdf5)

    if output_stats_path:
        stats = ImageChannelStatistics.compute_xarray(xarray)
        stats.write_csv(output_stats_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-imzml", dest="input_imzml_path")
    parser.add_argument("--mass-list", dest="mass_list_path")
    parser.add_argument("--output-hdf5")
    parser.add_argument("--n-jobs", default=20, type=int)
    parser.add_argument("--output-stats", required=False, default=None, dest="output_stats_path")
    args = vars(parser.parse_args())
    pprint(args)
    main_generate_ion_image(**args)


if __name__ == "__main__":
    main()
