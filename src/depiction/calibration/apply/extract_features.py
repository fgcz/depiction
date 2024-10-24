import xarray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel
from depiction.persistence.types import GenericReadFile, GenericReader


class ExtractFeatures:
    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def get_image(self, read_peaks: GenericReadFile) -> MultiChannelImage:
        all_features = self.get_all_features(read_peaks=read_peaks)
        return MultiChannelImage.from_flat(
            values=all_features,
            coordinates=read_peaks.coordinates_array_2d,
            channel_names="c" not in all_features.coords,
        )

    def get_all_features(self, read_peaks: GenericReadFile) -> DataArray:
        read_parallel = ReadSpectraParallel.from_config(self._parallel_config)
        return read_parallel.map_chunked(
            read_file=read_peaks,
            operation=self.get_chunk_features,
            bind_args=dict(calibration=self._calibration),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
        )

    @staticmethod
    def get_chunk_features(
        reader: GenericReader, spectra_indices: list[int], calibration: CalibrationMethod
    ) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            features = calibration.extract_spectrum_features(peak_mz_arr=mz_arr, peak_int_arr=int_arr)
            collect.append(features)
        combined = xarray.concat(collect, dim="i")
        combined.coords["i"] = spectra_indices
        return combined
