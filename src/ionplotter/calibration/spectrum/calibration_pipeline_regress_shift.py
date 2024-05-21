import numpy as np
from numpy.typing import NDArray

from ionplotter.calibration.models.fit_model import fit_model
from ionplotter.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from ionplotter.spatial_smoothing_sparse_aware import SpatialSmoothingSparseAware


class CalibrationPipelineRegressShift:
    """
    Calibrates spectra in a targeted setting, by regression of a shift model mapping mass to shift,
    and then subtracting the predicted shift from the observed mass.0
    Only implemented for picked peaks.
    :param ref_mz_arr: Reference m/z values to which the spectra should be calibrated.
    :param max_distance: Maximum distance to consider for the calibration.
    :param max_distance_unit: Unit of the maximum distance. (mz or ppm)
    :param model_type: Type of the model to fit. (e.g. see fit_model)
    :param model_unit: Unit of the model. (mz or ppm)
    :param input_smoothing_activated: Whether to apply spatial smoothing to the input data.
    :param input_smoothing_kernel_size: Size of the kernel for spatial smoothing.
    :param input_smoothing_kernel_std: Standard deviation of the kernel for spatial smoothing.
    :param min_points: Minimum number of points required to fit a model.
    """

    def __init__(
        self,
        ref_mz_arr: NDArray[float],
        max_distance: float,
        max_distance_unit: str,
        model_type: str,
        model_unit: str,
        input_smoothing_activated: bool = True,
        input_smoothing_kernel_size: int = 27,
        input_smoothing_kernel_std: float = 10.0,
        min_points: int = 3,
    ) -> None:
        self._ref_mz_arr = ref_mz_arr
        self._max_distance = max_distance
        self._max_distance_unit = max_distance_unit
        self._min_points = min_points
        self._model_type = model_type
        self._model_unit = model_unit
        self._input_smoothing_activated = input_smoothing_activated
        self._input_smoothing_kernel_size = input_smoothing_kernel_size
        self._input_smoothing_kernel_std = input_smoothing_kernel_std

    def preprocess_spectrum(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> NDArray[float]:
        distances_mz = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
            ref_mz_arr=self._ref_mz_arr,
            max_distance=self._max_distance,
            max_distance_unit=self._max_distance_unit,
        )
        if np.sum(~np.isnan(distances_mz)) < self._min_points:
            # there are insufficient peaks in the window around the reference, so no model should be fitted
            return np.full_like(distances_mz, fill_value=np.nan)

        median_shift_mz = np.nanmedian(distances_mz)

        # After removing the median shift from the observed masses,
        # pick for every reference the nearest observed peak.
        # This yields a vector of distances for every reference, with some values missing.
        signed_distances = ReferencePeakDistances.get_distances_nearest(
            peak_mz_arr=peak_mz_arr - median_shift_mz,
            ref_mz_arr=self._ref_mz_arr,
            max_distance=self._max_distance,
            max_distance_unit=self._max_distance_unit,
        )
        signed_distances += median_shift_mz

        if self._model_unit == "mz":
            return signed_distances
        elif self._model_unit == "ppm":
            # convert the distances to ppm
            return signed_distances / self._ref_mz_arr * 1e6
        else:
            raise ValueError(f"Unknown unit={self._model_unit}")

    def process_coefficients(self, all_coefficients: NDArray[float], coordinates_2d: NDArray[int]) -> NDArray[float]:
        # apply spatial smoothing if activated
        if self._input_smoothing_activated:
            smoother = SpatialSmoothingSparseAware(
                kernel_size=self._input_smoothing_kernel_size,
                kernel_std=self._input_smoothing_kernel_std,
            )
            distance_vectors = smoother.smooth_sparse_multi_channel(
                sparse_values=all_coefficients,
                coordinates=coordinates_2d,
            )
        else:
            distance_vectors = all_coefficients

        return distance_vectors

    def calibrate_spectrum(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], coefficients: NDArray[float]
    ) -> NDArray[float]:
        x = self._ref_mz_arr[~np.isnan(coefficients)]
        y = coefficients[~np.isnan(coefficients)]

        # TODO correct import
        model = fit_model(x=x, y=y, model_type=self._model_type)
        shifts_pred = model.predict(spectrum_mz_arr)
        if self._model_unit == "mz":
            shifts_mz = shifts_pred
        elif self._model_unit == "ppm":
            shifts_mz = shifts_pred / 1e6 * spectrum_mz_arr
        else:
            raise ValueError(f"Unknown unit={self._model_unit}")
        return spectrum_mz_arr - shifts_mz
