from typing import Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
from depiction.calibration.deprecated.calibrate_spectrum import CalibrateSpectrum
from depiction.calibration.models.linear_model import LinearModel
from depiction.calibration.models.polynomial_model import PolynomialModel
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.parallel_map import ParallelMap
from depiction.parallel_ops.read_spectra_parallel import ReadSpectraParallel
from depiction.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from depiction.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlReader,
    ImzmlWriter,
    ImzmlModeEnum,
)
from depiction.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware


class CalibrateImageTargeted:
    """Calibrates an imzML file to a set of reference m/z values.

    The approach consists of the following steps:
    - 1. For every spectrum:
    - 1.a. Pick the peaks.
    - 1.b. For every reference:
    - 1.b.a. Select a window of nearby peaks.
    - 1.b.b. Pick the strongest signal peak, or, give up if there is no suitable peak.
    - 1.b.c. Compute the median shift of these peaks.
    - 1.b.d. After removing the median shift from the observed masses,
             pick for every reference the nearest observed peak.
    - 1.c. This yields a vector of distances for every reference, with some values missing.
    - 2. Smooth these distance vectors spatially.
    - 3. For every spectrum:
    - 3.a. Fit a calibration model to the smoothed distances.
    - 3.b. Apply the calibration model to the m/z values.
    - 3.c. Save the results.
    """

    def __init__(
        self,
        reference_mz: NDArray[float],
        model_type: str,
        parallel_config: ParallelConfig,
        peak_picker: BasicPeakPicker,
        output_store: Optional[h5py.Group] = None,
        input_smoothing_activated: bool = True,
        input_smoothing_kernel_size: int = 27,
        input_smoothing_kernel_std: float = 10.0,
        distance_unit: str = "mz",
        max_distance: float = 2.0,
        accept_processed_data: bool = False,
    ) -> None:
        self._reference_mz = np.asarray(reference_mz)
        self._model_type = model_type
        self._parallel_config = parallel_config
        self._basic_peak_picker = peak_picker
        self._output_store = output_store
        self._input_smoothing_activated = input_smoothing_activated
        self._input_smoothing_kernel_size = input_smoothing_kernel_size
        self._input_smoothing_kernel_std = input_smoothing_kernel_std
        self._distance_unit = distance_unit
        self._max_distance = max_distance
        self._accept_processed_data = accept_processed_data

    def calibrate_image(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile) -> None:
        print("Computing distance vectors...")
        signed_distance_vectors, median_shifts = self._compute_reference_distance_vectors_for_file(read_file=read_file)
        self._save_results(signed_distance_vectors=signed_distance_vectors, median_shifts=median_shifts)

        if self._input_smoothing_activated:
            print("Smoothing distance vectors...")
            smoother = SpatialSmoothingSparseAware(
                kernel_size=self._input_smoothing_kernel_size,
                kernel_std=self._input_smoothing_kernel_std,
            )
            smoothed_distance_vectors = smoother.smooth_sparse_multi_channel(
                sparse_values=signed_distance_vectors,
                coordinates=read_file.coordinates_2d,
            )
            # TODO one might also consider interpolation here, but for now i won't add it
            self._save_results(smoothed_distance_vectors=smoothed_distance_vectors)
        else:
            print("Smoothing distance vectors skipped")
            smoothed_distance_vectors = signed_distance_vectors

        print("Compute models...")
        models = self._compute_models(
            distance_vectors=smoothed_distance_vectors,
            mz_refs=self._reference_mz,
            model_type=self._model_type,
        )
        self._save_results(
            models=[model.coef for model in models],
            coordinates=read_file.coordinates_2d,
            reference_mz=self._reference_mz,
        )
        self._save_attrs(model_type=self._model_type, distance_unit=self._distance_unit)

        print("Applying models...")
        self._apply_models(read_file, models, write_file)

    def _compute_reference_distance_vectors_for_file(
        self, read_file: ImzmlReadFile
    ) -> tuple[NDArray[float], NDArray[float]]:
        parallel = ReadSpectraParallel.from_config(config=self._parallel_config)

        return parallel.map_chunked(
            read_file=read_file,
            operation=self.process_chunk,
            reduce_fn=lambda chunks: (
                np.concatenate([c[0] for c in chunks], axis=0),
                np.concatenate([c[1] for c in chunks], axis=0),
            ),
            bind_args=dict(
                peak_picker=self._basic_peak_picker,
                distance_unit=self._distance_unit,
                max_distance=self._max_distance,
                reference_mz=self._reference_mz,
                pick_peaks=not self._accept_processed_data or read_file.imzml_mode == ImzmlModeEnum.CONTINUOUS,
            ),
        )

    @classmethod
    def process_chunk(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        peak_picker: BasicPeakPicker,
        distance_unit: str,
        max_distance: float,
        reference_mz: NDArray[float],
        pick_peaks: bool,
    ) -> tuple[NDArray[float], NDArray[float]]:
        distance_vectors = np.zeros((len(spectra_ids), len(reference_mz)))
        median_shifts = np.zeros(len(spectra_ids))
        for i_result, spectrum_id in enumerate(spectra_ids):
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            vector, med_shift = cls._compute_reference_distance_vector(
                mz_arr=mz_arr,
                int_arr=int_arr,
                unit=distance_unit,
                max_distance=max_distance,
                peak_picker=peak_picker,
                mz_ref_arr=reference_mz,
                pick_peaks=pick_peaks,
            )
            distance_vectors[i_result] = vector
            median_shifts[i_result] = med_shift
        return distance_vectors, median_shifts

    @classmethod
    def _compute_reference_distance_vector(
        cls,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
        unit: str,
        max_distance: float,
        peak_picker: BasicPeakPicker,
        mz_ref_arr: NDArray[float],
        pick_peaks: bool,
    ) -> tuple[NDArray[float], float]:
        # 1.a. Pick the peaks.
        if pick_peaks:
            mz_peak_arr, int_peak_arr = peak_picker.pick_peaks(mz_arr=mz_arr, int_arr=int_arr)
        else:
            mz_peak_arr, int_peak_arr = mz_arr, int_arr

        # 1.b. For every reference:
        # 1.b.a. Select a window of nearby peaks.
        # 1.b.b. Pick the strongest signal peak, or, give up if there is no suitable peak.
        strongest_peak_distances = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=mz_peak_arr,
            peak_int_arr=int_peak_arr,
            ref_mz_arr=mz_ref_arr,
            max_distance=max_distance,
            max_distance_unit=unit,
        )

        # 1.b.c. Compute the median shift of these peaks.
        if np.all(np.isnan(strongest_peak_distances)):
            # there are no peaks in the window around the reference, so there is also no model to fit
            return np.full(len(mz_ref_arr), np.nan), np.nan
        else:
            median_shift = np.nanmedian(strongest_peak_distances)

        # 1.b.d. After removing the median shift from the observed masses,
        # pick for every reference the nearest observed peak.
        # 1.c. This yields a vector of distances for every reference, with some values missing.
        signed_distances = ReferencePeakDistances.get_distances_nearest(
            peak_mz_arr=mz_peak_arr - median_shift,
            ref_mz_arr=mz_ref_arr,
            max_distance=max_distance,
            max_distance_unit=unit,
        )
        signed_distances += median_shift

        # Also return the median shift for traceability...
        return signed_distances, median_shift

    def _compute_models(
        self,
        distance_vectors: NDArray[float],
        mz_refs: NDArray[float],
        model_type: str,
    ) -> list[LinearModel | PolynomialModel]:
        parallel_map = ParallelMap(config=self._parallel_config)
        return parallel_map(
            operation=self._compute_models_for_chunk,
            tasks=[
                distance_vectors[spectra_indices, :]
                for spectra_indices in self._parallel_config.get_task_splits(n_items=distance_vectors.shape[0])
            ],
            bind_kwargs=dict(mz_refs=mz_refs, model_type=model_type),
            reduce_fn=ParallelMap.reduce_concat,
        )

    @classmethod
    def _compute_models_for_chunk(
        cls,
        distance_vectors: NDArray[float],
        mz_refs: NDArray[float],
        model_type: str,
    ) -> list[LinearModel] | list[PolynomialModel]:
        models = []
        for i in range(distance_vectors.shape[0]):
            x = mz_refs
            y = distance_vectors[i, :]

            # remove nan values (only checking the distance for nans)
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]

            models.append(CalibrateSpectrum.fit_model(x=x, y=y, model_type=model_type))
        return models

    def _apply_models(
        self,
        read_file: ImzmlReadFile,
        models: list[LinearModel | PolynomialModel],
        write_file: ImzmlWriteFile,
    ) -> None:
        parallelize = WriteSpectraParallel.from_config(self._parallel_config)
        unit = self._distance_unit

        def chunk_operation(reader: ImzmlReader, spectra_indices: list[int], writer: ImzmlWriter) -> None:
            for spectrum_id in spectra_indices:
                mz_arr, int_arr = reader.get_spectrum(spectrum_id)
                model = models[spectrum_id]
                if unit == "mz":
                    mz_arr = mz_arr - model.predict(mz_arr)
                elif unit == "ppm":
                    mz_arr = np.array([mz * (1 - model.predict(mz) / 1e6) for mz in mz_arr])
                else:
                    raise ValueError(f"Unknown unit={unit}")
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
