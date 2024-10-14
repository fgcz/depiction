# TODO very experimental first version
# TODO needs better name (but like this for easier distinguishing)
from pathlib import Path
from typing import Optional

import numpy as np
import xarray
from loguru import logger
from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel, WriteSpectraParallel
from depiction.parallel_ops.parallel_map import ParallelMap
from depiction.persistence.types import GenericReadFile, GenericWriteFile, GenericReader, GenericWriter


class PerformCalibration:
    def __init__(
        self,
        calibration: CalibrationMethod,
        parallel_config: ParallelConfig,
        coefficient_output_file: Path | None = None,
    ) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config
        self._coefficient_output_file = coefficient_output_file

    def _validate_per_spectra_array(self, array: DataArray, coordinates_2d: NDArray[float]) -> None:
        """Checks the DataArray has the correct shapes and dimensions. Used for debugging."""
        # TODO make it configurable in the future, whether this check is executed, during development it definitely
        #      should be here since it can safe a ton of time
        expected_coords = {"i", "x", "y"}
        if set(array.coords) != expected_coords:
            raise ValueError(f"Expected coords={expected_coords}, got={set(array.coords)}")
        expected_dims = {"i", "c"}

        errors = []
        if set(array.dims) != expected_dims:
            logger.error(f"Expected dims={expected_dims}, got={set(array.dims)}")
            errors.append("Mismatch in dimensions")
        if not np.array_equal(array.x.values, coordinates_2d[:, 0]):
            logger.error(f"Expected x: values={coordinates_2d[:, 0]} shape={coordinates_2d[:, 0].shape}")
            logger.error(f"Actual   x: values={array.x.values} shape={array.x.values.shape}")
            logger.info(f"(Expected x values without offset: {coordinates_2d[:, 0] - coordinates_2d[:, 0].min()})")
            errors.append("Mismatch in x values")
        if not np.array_equal(array.y.values, coordinates_2d[:, 1]):
            logger.error(f"Expected y: values={coordinates_2d[:, 1]} shape={coordinates_2d[:, 1].shape}")
            logger.error(f"Actual   y: values={array.y.values} shape={array.y.values.shape}")
            logger.info(f"(Expected y values without offset: {coordinates_2d[:, 1] - coordinates_2d[:, 1].min()})")
            errors.append("Mismatch in y values")
        if not np.array_equal(array.i.values, np.arange(len(array.i))):
            errors.append("Mismatch in i values")
            logger.error(f"Expected i: values={np.arange(len(array.i))} shape={np.arange(len(array.i)).shape}")
            logger.error(f"Actual   i: values={array.i.values} shape={array.i.values.shape}")
        if errors:
            raise ValueError(errors)

    def calibrate_image(
        self, read_peaks: GenericReadFile, write_file: GenericWriteFile, read_full: Optional[GenericReadFile] = None
    ) -> None:
        read_full = read_full or read_peaks

        logger.info("Extracting all features...")
        all_features = self._extract_all_features(read_peaks)
        self._write_data_array(all_features, group="features_raw")

        logger.info("Preprocessing features...")
        all_features = self._calibration.preprocess_image_features(all_features=all_features)
        self._write_data_array(all_features, group="features_processed")

        logger.info("Fitting models...")
        model_coefs = self._fit_all_models(all_features=all_features)
        self._write_data_array(model_coefs, group="model_coefs")

        logger.info("Applying models...")
        self._apply_all_models(read_file=read_full, write_file=write_file, all_model_coefs=model_coefs)

    def _extract_all_features(self, read_peaks: GenericReadFile) -> MultiChannelImage:
        read_parallel = ReadSpectraParallel.from_config(self._parallel_config)
        all_features = read_parallel.map_chunked(
            read_file=read_peaks,
            operation=self._extract_chunk_features,
            bind_args=dict(
                calibration=self._calibration,
            ),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
        )
        return MultiChannelImage.from_flat(
            values=all_features,
            coordinates=DataArray(read_peaks.coordinates_2d, dims=["i", "d"], coords={"d": ["x", "y"]}),
            channel_names=None,
        )

    def _apply_all_models(
        self, read_file: GenericReadFile, write_file: GenericWriteFile, all_model_coefs: MultiChannelImage
    ) -> None:
        write_parallel = WriteSpectraParallel.from_config(self._parallel_config)
        write_parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._calibrate_spectra,
            bind_args=dict(
                calibration=self._calibration,
                all_model_coefs=all_model_coefs,
            ),
        )

    def _fit_all_models(self, all_features: MultiChannelImage) -> MultiChannelImage:
        parallel_map = ParallelMap.from_config(self._parallel_config)
        # TODO to be refactored
        all_features_flat = all_features.data_flat
        result = parallel_map(
            operation=self._fit_chunk_models,
            tasks=np.array_split(all_features_flat.coords["i"], self._parallel_config.n_jobs),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
            bind_kwargs={"all_features": all_features_flat},
        )
        return MultiChannelImage.from_flat(result, coordinates=all_features.coordinates_flat, channel_names=None)

    def _fit_chunk_models(self, spectra_indices: NDArray[int], all_features: DataArray) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            features = all_features.sel(i=spectrum_id)
            model_coef = self._calibration.fit_spectrum_model(features=features)
            collect.append(model_coef)
        return xarray.concat(collect, dim="i")

    def _write_data_array(self, image: MultiChannelImage, group: str) -> None:
        if not self._coefficient_output_file:
            return

        # TODO this uses deprecated functionality but needs to be handled somewhere more generally later
        array = image.data_flat.assign_coords({"i": np.arange(len(image.data_flat.i))})
        # TODO engine should not be necessary, but using it for debugging
        array.to_netcdf(path=self._coefficient_output_file, group=group, format="NETCDF4", engine="netcdf4", mode="a")

    @staticmethod
    def _extract_chunk_features(
        reader: GenericReader,
        spectra_indices: list[int],
        calibration: CalibrationMethod,
    ) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            features = calibration.extract_spectrum_features(peak_mz_arr=mz_arr, peak_int_arr=int_arr)
            collect.append(features)
        combined = xarray.concat(collect, dim="i")
        combined.coords["i"] = spectra_indices
        return combined

    @staticmethod
    def _calibrate_spectra(
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
