from typing import Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
from depiction.calibration.deprecated.calibrate_spectrum import CalibrateSpectrum
from depiction.calibration.deprecated.model_smoothing import ModelSmoothing
from depiction.calibration.models.linear_model import LinearModel
from depiction.calibration.models.polynomial_model import PolynomialModel
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from depiction.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlReader,
    ImzmlWriter,
)


class CalibrateImage:
    def __init__(
        self,
        reference_mz: NDArray[float],
        model_type: str,
        parallel_config: ParallelConfig,
        output_store: Optional[h5py.Group] = None,
    ) -> None:
        self._reference_mz = reference_mz
        self._model_type = model_type
        self._parallel_config = parallel_config
        self._output_store = output_store

    def calibrate_image(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile) -> None:
        print("Computing models...")
        models_original = self._compute_models(read_file=read_file)
        print("Smoothing models...")
        smoothed_models = self._smooth_models(models_original)
        print("Applying models...")
        self._apply_models(read_file, smoothed_models, write_file)

    def _compute_models(self, read_file: ImzmlReadFile) -> list[LinearModel | PolynomialModel]:
        parallelize = ReadSpectraParallel.from_config(self._parallel_config)
        models_original = parallelize.map_chunked(
            read_file=read_file,
            operation=self._compute_models_operation,
            bind_args=dict(
                reference_mz=self._reference_mz,
                model_type=self._model_type,
            ),
            reduce_fn=parallelize.reduce_concat,
        )
        self._save_results(
            model_coefs=[model.coef for model in models_original],
            coordinates=read_file.coordinates,
            reference_mz=self._reference_mz,
        )
        return models_original

    @staticmethod
    def _compute_models_operation(
        reader: ImzmlReader,
        spectra_indices: NDArray[int],
        reference_mz: NDArray[float],
        model_type: str,
    ) -> list[LinearModel] | list[PolynomialModel]:
        # TODO hardcoded parameters
        peak_picker = BasicPeakPicker(smooth_sigma=0.1, min_prominence=0.05)
        calibrate = CalibrateSpectrum(reference_mz=reference_mz, peak_picker=peak_picker, model_type=model_type)
        results = []
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            model = calibrate.calibrate_spectrum(mz_arr=mz_arr, int_arr=int_arr)
            results.append(model)
        return results

    def _smooth_models(
        self, models_original: list[LinearModel | PolynomialModel]
    ) -> list[LinearModel | PolynomialModel]:
        smoother = ModelSmoothing(sigma=1.0)
        smoothed_models = smoother.smooth_sequential(models=models_original)
        self._save_results(smoothed_models=[model.coef for model in smoothed_models])
        return smoothed_models

    def _apply_models(
        self,
        read_file: ImzmlReadFile,
        smoothed_models: list[LinearModel | PolynomialModel],
        write_file: ImzmlWriteFile,
        mz_center: float = 0,
    ) -> None:
        # TODO full support for mz_center in this class
        parallelize = WriteSpectraParallel.from_config(self._parallel_config)
        parallelize.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._apply_models_operation,
            bind_args=dict(smoothed_models=smoothed_models, mz_center=mz_center),
        )

    @staticmethod
    def _apply_models_operation(
        reader: ImzmlReader,
        spectra_indices: list[int],
        writer: ImzmlWriter,
        smoothed_models: list[LinearModel | PolynomialModel],
        mz_center: float,
    ) -> None:
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            mz_arr = mz_arr - mz_center
            coordinates = reader.coordinates[spectrum_id]
            model = smoothed_models[spectrum_id]

            mz_error = model.predict(x=mz_arr)
            calibrated_mz = mz_arr - mz_error

            # Remove m/z values outside the range of the input spectrum
            tolerance = 1.0  # TODO!!! customizable
            is_valid = (np.min(mz_arr) - tolerance < calibrated_mz) & (calibrated_mz < np.max(mz_arr) + tolerance)
            calibrated_mz = calibrated_mz[is_valid]
            calibrated_int = int_arr[is_valid]

            if len(calibrated_mz) > 10:
                writer.add_spectrum(calibrated_mz, calibrated_int, coordinates)
            else:
                print(
                    "WARNING: too few data points after calibration, "
                    "returning the original mz array without calibration."
                )
                writer.add_spectrum(mz_arr, int_arr, coordinates)

    def _save_results(self, **kwargs) -> None:
        """Write datasets according to the given kwargs to the output HDF5 store."""
        if self._output_store is not None:
            for key, value in kwargs.items():
                self._output_store.create_dataset(key, data=value)
