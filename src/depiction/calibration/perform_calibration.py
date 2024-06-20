# TODO very experimental first version
# TODO needs better name (but like this for easier distinguishing)
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import xarray
from loguru import logger
from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel, WriteSpectraParallel
from depiction.parallel_ops.parallel_map import ParallelMap
from depiction.persistence import ImzmlReadFile, ImzmlWriteFile, ImzmlReader, ImzmlWriter


class PerformCalibration:
    def __init__(
        self,
        calibration: CalibrationMethod,
        parallel_config: ParallelConfig,
        output_store: h5py.Group | None = None,
        coefficient_output_file: Path | None = None,
    ) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config
        self._output_store = output_store
        self._coefficient_output_file = coefficient_output_file

    # def _reshape(self, pattern: str, data: DataArray, coordinates) -> DataArray:
    #    if pattern == "i,c->y,x,c":
    #        data = data.copy()
    #        # TODO fix the deprecation here!
    #        data["i"] = pd.MultiIndex.from_arrays((coordinates[:, 1], coordinates[:, 0]), names=("y", "x"))
    #        data = data.unstack("i")
    #        return data.transpose("y", "x", "c")
    #    elif pattern == "y,x,c->i,c":
    #        data = data.transpose("y", "x", "c").copy()
    #        data = data.stack(i=("y", "x")).drop_vars(["i", "x", "y"])
    #        # convert to integers
    #        data["i"] = np.arange(len(data["i"]))
    #        return data.transpose("i", "c")
    #    else:
    #        raise ValueError(f"Unknown pattern={repr(pattern)}")

    def _validate_per_spectra_array(self, array: DataArray, coordinates_2d) -> None:
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
        self, read_peaks: ImzmlReadFile, write_file: ImzmlWriteFile, read_full: Optional[ImzmlReadFile] = None
    ) -> None:
        if read_full is None:
            read_full = read_peaks

        logger.info("Extracting all features...")
        all_features = self._extract_all_features(read_peaks).transpose("i", "c")
        self._validate_per_spectra_array(all_features, coordinates_2d=read_peaks.coordinates_2d)
        self._write_data_array(all_features, group="features_raw")

        logger.info("Preprocessing features...")
        all_features = self._calibration.preprocess_image_features(all_features=all_features).transpose("i", "c")
        self._validate_per_spectra_array(all_features, coordinates_2d=read_peaks.coordinates_2d)
        self._write_data_array(all_features, group="features_processed")

        logger.info("Fitting models...")
        model_coefs = self._fit_all_models(all_features=all_features).transpose("i", "c")
        self._validate_per_spectra_array(model_coefs, coordinates_2d=read_peaks.coordinates_2d)
        self._write_data_array(model_coefs, group="model_coefs")

        logger.info("Applying models...")
        self._apply_all_models(read_file=read_full, write_file=write_file, all_model_coefs=model_coefs)

    def _extract_all_features(self, read_peaks: ImzmlReadFile) -> DataArray:
        read_parallel = ReadSpectraParallel.from_config(self._parallel_config)
        all_features = read_parallel.map_chunked(
            read_file=read_peaks,
            operation=self._extract_chunk_features,
            bind_args=dict(
                calibration=self._calibration,
            ),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
        )
        return all_features.assign_coords(
            x=("i", read_peaks.coordinates_2d[:, 0]), y=("i", read_peaks.coordinates_2d[:, 1])
        )

    def _apply_all_models(
        self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile, all_model_coefs: DataArray
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

    def _fit_all_models(self, all_features: DataArray) -> DataArray:
        parallel_map = ParallelMap.from_config(self._parallel_config)
        result = parallel_map(
            operation=self._fit_chunk_models,
            tasks=np.array_split(all_features.coords["i"], self._parallel_config.n_jobs),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
            bind_kwargs={"all_features": all_features},
        )
        return result

    def _fit_chunk_models(self, spectra_indices: NDArray[int], all_features: DataArray) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            features = all_features.sel(i=spectrum_id)
            model_coef = self._calibration.fit_spectrum_model(features=features)
            collect.append(model_coef)
        combined = xarray.concat(collect, dim="i")
        combined.coords["i"] = spectra_indices
        return combined

    def _write_data_array(self, array: DataArray, group: str) -> None:
        if not self._coefficient_output_file:
            return
        # TODO engine should not be necessary, but using it for debugging
        array.to_netcdf(path=self._coefficient_output_file, group=group, format="NETCDF4", engine="netcdf4", mode="a")

    @staticmethod
    def _extract_chunk_features(
        reader: ImzmlReader,
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
        reader: ImzmlReader,
        spectra_indices: list[int],
        writer: ImzmlWriter,
        calibration: CalibrationMethod,
        all_model_coefs: DataArray,
    ) -> None:
        for spectrum_id in spectra_indices:
            # TODO sanity check the usage of i as spectrum_id (i.e. check the coords!)
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_id)
            features = all_model_coefs.sel(i=spectrum_id)
            calib_mz_arr, calib_int_arr = calibration.apply_spectrum_model(
                spectrum_mz_arr=mz_arr, spectrum_int_arr=int_arr, model_coef=features
            )
            writer.add_spectrum(calib_mz_arr, calib_int_arr, coords)
