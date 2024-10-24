# TODO very experimental first version
# TODO needs better name (but like this for easier distinguishing)
import numpy as np
import xarray
from loguru import logger
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel, WriteSpectraParallel
from depiction.parallel_ops.parallel_map import ParallelMap
from depiction.persistence.types import GenericReadFile, GenericWriteFile, GenericReader, GenericWriter


class CalibrateImage:
    def __init__(
        self,
        calibration: CalibrationMethod,
        parallel_config: ParallelConfig,
        coefficient_output_file: Path | None = None,
    ) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config
        self._coefficient_output_file = coefficient_output_file

    def calibrate_image(
        self, read_peaks: GenericReadFile, write_file: GenericWriteFile, read_full: Optional[GenericReadFile] = None
    ) -> None:
        read_full = read_full or read_peaks

        logger.info("Extracting all features...")
        all_features = ExtractFeatures(self._calibration, self._parallel_config).get_image(read_peaks)
        self._write_data_array(all_features, group="features_raw")

        logger.info("Preprocessing features...")
        all_features = self._calibration.preprocess_image_features(all_features=all_features)
        self._write_data_array(all_features, group="features_processed")

        logger.info("Fitting models...")
        model_coefs = FitModels(self._calibration, self._parallel_config).get_image(all_features)
        self._write_data_array(model_coefs, group="model_coefs")

        logger.info("Applying models...")
        ApplyModels(self._calibration, self._parallel_config).write_to_file(
            read_file=read_full, write_file=write_file, all_model_coefs=model_coefs
        )

    def _write_data_array(self, image: MultiChannelImage, group: str) -> None:
        """Exports the given image into a HDF5 group of the coefficient output file (if specified)."""
        if not self._coefficient_output_file:
            return
        image.write_hdf5(path=self._coefficient_output_file, mode="a", group=group)


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


class FitModels:
    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def get_image(self, all_features: MultiChannelImage) -> MultiChannelImage:
        result = self.get_all_features(all_features)
        return MultiChannelImage.from_flat(
            result, coordinates=all_features.coordinates_flat, channel_names="c" not in result.coords
        )

    def get_all_features(self, all_features):
        parallel_map = ParallelMap.from_config(self._parallel_config)
        # TODO to be refactored
        all_features_flat = all_features.data_flat
        result = parallel_map(
            operation=self.get_chunk_features,
            tasks=np.array_split(all_features_flat.coords["i"], self._parallel_config.n_jobs),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
            bind_kwargs={"all_features": all_features_flat},
        )
        return result

    def get_chunk_features(self, spectra_indices: NDArray[int], all_features: DataArray) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            features = all_features.sel(i=spectrum_id)
            model_coef = self._calibration.fit_spectrum_model(features=features)
            collect.append(model_coef)
        return xarray.concat(collect, dim="i")


class ApplyModels:
    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def write_to_file(
        self, read_file: GenericReadFile, write_file: GenericWriteFile, all_model_coefs: MultiChannelImage
    ) -> None:
        write_parallel = WriteSpectraParallel.from_config(self._parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self.calibrate_spectra,
            bind_args=dict(
                calibration=self._calibration,
                all_model_coefs=all_model_coefs,
            ),
        )

    @staticmethod
    def calibrate_spectra(
        reader: GenericReader,
        spectra_indices: list[int],
        writer: GenericWriter,
        calibration: CalibrationMethod,
        all_model_coefs: MultiChannelImage,
    ) -> None:
        for spectrum_id in spectra_indices:
            # TODO sanity check the usage of i as spectrum_id (i.e. check the coords!)
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            features = all_model_coefs.data_flat.isel(i=spectrum_id)
            calib_mz_arr, calib_int_arr = calibration.apply_spectrum_model(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, model_coef=features
            )
            writer.add_spectrum(calib_mz_arr, calib_int_arr, coords)
