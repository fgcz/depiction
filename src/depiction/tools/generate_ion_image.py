from collections.abc import Sequence
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlReader


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
        return MultiChannelImage.from_spatial(data, bg_value=np.nan)

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
        return DataArray(array, dims=("i", "c"))

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
        return MultiChannelImage.from_flat(
            values=channel_values,
            coordinates=input_file.coordinates_array_2d,
            channel_names=channel_names,
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
