from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from depiction.calibration.deprecated.reference_distance_estimator import (
    ReferenceDistanceEstimator,
)
from depiction.parallel_ops import (
    ParallelConfig,
    ReadSpectraParallel,
    WriteSpectraParallel,
)
from depiction.spatial_smoothing import SpatialSmoothing

if TYPE_CHECKING:
    from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
    from numpy.typing import NDArray
    from depiction.persistence import (
        ImzmlReadFile,
        ImzmlReader,
        ImzmlWriteFile,
        ImzmlWriter,
    )


@dataclass(frozen=True)
class AdjustMedianShift:
    """This class computes the median shift for each spectrum in an image, smoothes the values, and then allows
    applying it to get a new spectrum to further calibrate. This is necessary to deal with acquisitions which have mass
    errors above roughly 0.5 Da.
    """

    peak_picker: BasicPeakPicker
    ref_mz_arr: NDArray[float]
    parallel_config: ParallelConfig
    max_distance: float = 2.0
    smooth_sigma: float = 10.0

    # @cached_property
    # def _ref_distance_estimator(self) -> ReferenceDistanceEstimator:
    #    return ReferenceDistanceEstimator(reference_mz=self.ref_mz_arr, n_candidates=3)

    def compute_shifts_ppm(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        # TODO having to create an instance for every spectrum is a design flaw, the problem is that cached_property
        #      creates a Rwlock which prevents the instance from being used in parallel (unpickleable)
        peak_idx = self.peak_picker.pick_peaks_index(mz_arr=mz_arr, int_arr=int_arr)
        if len(peak_idx) == 0:
            return np.array([])
        (
            indices,
            signed_distances_mz,
        ) = ReferenceDistanceEstimator(reference_mz=self.ref_mz_arr, n_candidates=3).compute_max_peak_within_distance(
            mz_peaks=mz_arr[peak_idx],
            int_peaks=int_arr[peak_idx],
            max_distance=self.max_distance,
            keep_missing=True,
        )
        signed_distances_ppm = signed_distances_mz / self.ref_mz_arr * 1e6
        signed_distances_ppm = signed_distances_ppm[indices != -1]
        return signed_distances_ppm

    def compute_median_shift_ppm(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> float:
        """Computes the median shift for the specified mass spectrum in ppm."""
        signed_distances_ppm = self.compute_shifts_ppm(mz_arr=mz_arr, int_arr=int_arr)
        return np.median(signed_distances_ppm)

    def compute_median_shifts(
        self, read_file: ImzmlReadFile, spectra_indices: list[int] | None = None
    ) -> NDArray[float]:
        parallelize = ReadSpectraParallel.from_config(self.parallel_config)
        values = parallelize.map_chunked(
            read_file=read_file,
            operation=self._compute_median_shifts_operation,
            bind_args={
                "self_copy": self,
            },
            reduce_fn=parallelize.reduce_concat,
            spectra_indices=spectra_indices,
        )
        return np.asarray(values)

    @classmethod
    def _compute_median_shifts_operation(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        self_copy: AdjustMedianShift,
    ) -> list[float]:
        values = []
        for spectrum_id in spectra_ids:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            values.append(self_copy.compute_median_shift_ppm(mz_arr=mz_arr, int_arr=int_arr))
        return values

    def smooth_median_shifts(
        self,
        median_shifts: NDArray[float],
        coordinates_2d: NDArray[int],
    ) -> NDArray[float]:
        return SpatialSmoothing(
            sigma=self.smooth_sigma,
            background_fill_mode="nearest",
            background_value=np.nan,
        ).smooth_sparse(sparse_values=median_shifts, coordinates=coordinates_2d)

    def apply_correction(
        self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile
    ) -> tuple[NDArray[float], NDArray[float]]:
        median_shifts = self.compute_median_shifts(read_file=read_file)
        smooth_median_shifts = self.smooth_median_shifts(
            median_shifts=median_shifts, coordinates_2d=read_file.coordinates_2d
        )
        parallelize = WriteSpectraParallel.from_config(self.parallel_config)
        parallelize.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._apply_correction_operation,
            bind_args={"median_shifts": smooth_median_shifts},
        )
        return median_shifts, smooth_median_shifts

    @classmethod
    def _apply_correction_operation(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        writer: ImzmlWriter,
        median_shifts: NDArray[float],
    ) -> None:
        for spectrum_id in spectra_ids:
            mz_arr, int_arr = reader.get_spectrum(spectrum_id)
            shift_ppm = median_shifts[spectrum_id]
            mz_arr_corrected = mz_arr - (mz_arr * shift_ppm / 1e6)
            writer.add_spectrum(mz_arr_corrected, int_arr, reader.coordinates[spectrum_id])
