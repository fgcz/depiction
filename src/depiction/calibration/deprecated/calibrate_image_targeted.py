from typing import Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
from depiction.calibration.deprecated.calibrate_spectrum import CalibrateSpectrum
from depiction.calibration.models.linear_model import LinearModel
from depiction.calibration.models.polynomial_model import PolynomialModel
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from depiction.spectrum.peak_filtering import FilterByIntensity
from depiction.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlReader,
)
from depiction.calibration.deprecated.adjust_median_shift import AdjustMedianShift


class CalibrateImageTargeted:
    """Calibrates an imzML file using a set of reference m/z values.

    The approach consists of the following steps:
    - Compute the median shift in ppm (per-spectrum)
    - Smooth the median shifts (per-image)
    - Fit a calibration model, after removing the median shift (per-spectrum)
    - Apply the calibration model to the m/z values (per-spectrum)
    """

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
        print("Compute median shift per spectrum...")
        median_shifts_ppm, smooth_median_shifts_ppm = self._compute_median_shifts(read_file=read_file)
        self._save_attrs(model_type=self._model_type)
        self._save_results(
            median_shifts_ppm=median_shifts_ppm,
            smooth_median_shifts_ppm=smooth_median_shifts_ppm,
        )

        print("Compute models...")
        models = self._compute_models(read_file=read_file, median_shifts_ppm=smooth_median_shifts_ppm)
        self._save_results(
            smoothed_models=[model.coef for model in models],
            coordinates=read_file.coordinates_2d,
            reference_mz=self._reference_mz,
        )

        # TODO smoothing etc

        print("Applying models...")
        self._apply_models(read_file, models, write_file)

    def _compute_median_shifts(self, read_file: ImzmlReadFile) -> tuple[NDArray[float], NDArray[float]]:
        peak_picker = BasicPeakPicker(smooth_sigma=1, min_prominence=0.002)
        adjust_median_shift = AdjustMedianShift(
            peak_picker=peak_picker,
            ref_mz_arr=self._reference_mz,
            parallel_config=self._parallel_config,
            max_distance=2.0,
            smooth_sigma=10.0,
        )
        median_shifts = adjust_median_shift.compute_median_shifts(read_file=read_file)
        smooth_median_shifts = adjust_median_shift.smooth_median_shifts(
            median_shifts, coordinates_2d=read_file.coordinates_2d
        )
        return median_shifts, smooth_median_shifts

    def _compute_models(
        self,
        read_file: ImzmlReadFile,
        median_shifts_ppm: NDArray[float],
    ) -> list[LinearModel | PolynomialModel]:
        parallelize = ReadSpectraParallel.from_config(self._parallel_config)
        models = parallelize.map_chunked(
            read_file=read_file,
            operation=self._compute_models_operation,
            bind_args=dict(
                reference_mz=self._reference_mz,
                model_type=self._model_type,
                median_shifts_ppm=median_shifts_ppm,
            ),
            reduce_fn=parallelize.reduce_concat,
        )
        return models

    @classmethod
    def _compute_models_operation(
        cls,
        reader: ImzmlReader,
        spectra_indices: NDArray[int],
        reference_mz: NDArray[float],
        model_type: str,
        median_shifts_ppm: NDArray[float],
    ) -> list[LinearModel] | list[PolynomialModel]:
        peak_filter = FilterByIntensity(min_intensity=0.0005, normalization="tic")
        peak_picker = BasicPeakPicker(smooth_sigma=1e-6, min_prominence=0.01)

        models = []
        for spectrum_id in spectra_indices:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            problem_mz, problem_dist_mz = CalibrateSpectrum.get_matches_from_config(
                peak_picker=peak_picker,
                peak_filtering=peak_filter,
                mz_arr=cls._apply_ppm_shift(mz_arr, -median_shifts_ppm[spectrum_id]),
                int_arr=int_arr,
                reference_mz_arr=reference_mz,
                distance_limit=2.0,
            )
            # the problem_dist returned by get_matches_from_config is in terms of m/z, not ppm
            # so we need to convert it to ppm (TODO methods should always indicate the unit)
            problem_dist_ppm = problem_dist_mz / problem_mz * 1e6 + median_shifts_ppm[spectrum_id]
            model, _ = CalibrateSpectrum.fit_calibration_model_for_problem(
                model_type=model_type,
                problem_mz=problem_mz,
                problem_distance=problem_dist_ppm,
                mz_arr=mz_arr,
                # TODO this prune_bad_limit can be problematic sometimes, while it makes sense as a safe guard
                #      there should be more functionality that can react appropriately to a bad value here (e.g.
                #      replace the model by its neighbours)
                prune_bad_limit=3000,
            )
            models.append(model)
        return models

    @staticmethod
    def _apply_ppm_shift(mz_arr: NDArray[float], ppm_shift: float) -> NDArray[float]:
        return mz_arr * (1 + ppm_shift / 1e6)

    def _apply_models(
        self,
        read_file: ImzmlReadFile,
        models: list[LinearModel | PolynomialModel],
        write_file: ImzmlWriteFile,
    ) -> None:
        parallelize = WriteSpectraParallel.from_config(self._parallel_config)

        def chunk_operation(reader, spectra_indices, writer) -> None:
            for spectrum_id in spectra_indices:
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                model = models[spectrum_id]
                mz_arr = np.array([mz * (1 - model.predict(mz) / 1e6) for mz in mz_arr])
                writer.add_spectrum(mz_arr, int_arr, reader.coordinates[spectrum_id])

        parallelize.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=chunk_operation,
        )

    def _save_attrs(self, **kwargs) -> None:
        if self._output_store is not None:
            for key, value in kwargs.items():
                self._output_store.attrs[key] = value

    def _save_results(self, **kwargs) -> None:
        """Write datasets according to the given kwargs to the output HDF5 store."""
        if self._output_store is not None:
            for key, value in kwargs.items():
                self._output_store.create_dataset(key, data=value)
