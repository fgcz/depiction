import numpy as np
from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.calibration.models import LinearModel
from depiction.calibration.models.fit_model import fit_model
from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from depiction.image import MultiChannelImage
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.tools.calibrate.spatial_smoothing_config import SpatialSmoothingType


class CalibrationMethodRegressShift(CalibrationMethod):
    """
    Calibrates spectra in a targeted setting, by regression of a shift model mapping mass to shift,
    and then subtracting the predicted shift from the observed mass.0
    Only implemented for picked peaks.
    :param ref_mz_arr: Reference m/z values to which the spectra should be calibrated.
    :param max_distance: Maximum distance to consider for the calibration.
    :param max_distance_unit: Unit of the maximum distance. (mz or ppm)
    :param model_type: Type of the model to fit. (e.g. see fit_model)
    :param model_unit: Unit of the model. (mz or ppm)
    :param input_smoothing: Smoothing to apply to the input features (distance vectors).
    :param min_points: Minimum number of points required to fit a model.
    :param peak_filtering: Peak filtering to apply to the input peaks before extracting features at all.
    """

    def __init__(
        self,
        ref_mz_arr: NDArray[float],
        max_distance: float,
        max_distance_unit: str,
        model_type: str,
        model_unit: str,
        input_smoothing: SpatialSmoothingType | None,
        min_points: int = 3,
        peak_filtering: PeakFilteringType | None = None,
    ) -> None:
        self._ref_mz_arr = ref_mz_arr
        self._max_distance = max_distance
        self._max_distance_unit = max_distance_unit
        self._min_points = min_points
        self._model_type = model_type
        self._model_unit = model_unit
        self._input_smoothing = input_smoothing
        self._peak_filtering = peak_filtering

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        if self._peak_filtering:
            # TODO this is sort of problematic, to be reconsidered how we can avoid passing peak arrays as spectrum
            #      arrays (it should be part of the API and implemented consistently)
            peak_mz_arr, peak_int_arr = self._peak_filtering.filter_peaks(
                peak_mz_arr, peak_int_arr, peak_mz_arr, peak_int_arr
            )
        distances_mz = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
            ref_mz_arr=self._ref_mz_arr,
            max_distance=self._max_distance,
            max_distance_unit=self._max_distance_unit,
        )
        if np.sum(~np.isnan(distances_mz)) < self._min_points:
            # there are insufficient peaks in the window around the reference, so no model should be fitted
            return DataArray(np.full_like(distances_mz, fill_value=np.nan), dims=["c"])

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
            return DataArray(signed_distances, dims=["c"])
        elif self._model_unit == "ppm":
            # convert the distances to ppm
            return DataArray(signed_distances / self._ref_mz_arr * 1e6, dims=["c"])
        else:
            raise ValueError(f"Unknown unit={self._model_unit}")

    def preprocess_image_features(self, all_features: MultiChannelImage) -> MultiChannelImage:
        return self._input_smoothing.smooth_image(all_features) if self._input_smoothing else all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        coefficients = features.values  # TODO make use of xarray
        x = self._ref_mz_arr[~np.isnan(coefficients)]
        y = coefficients[~np.isnan(coefficients)]
        model = fit_model(x=x, y=y, model_type=self._model_type)
        return DataArray(model.coef, dims=["c"])

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        if not self._model_type.startswith("linear"):
            # TODO this shouldn't be hard to implement, basically instead of strings we could use enums and provide some
            #   helper methods. i think it would also make the code cleaner
            raise NotImplementedError("proper model support")
        model = LinearModel(model_coef.values)

        shifts_pred = model.predict(spectrum_mz_arr)
        if self._model_unit == "mz":
            shifts_mz = shifts_pred
        elif self._model_unit == "ppm":
            shifts_mz = shifts_pred / 1e6 * spectrum_mz_arr
        else:
            raise ValueError(f"Unknown unit={self._model_unit}")
        return spectrum_mz_arr - shifts_mz, spectrum_int_arr

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_distance={self._max_distance}, "
            f"max_distance_unit={self._max_distance_unit}, "
            f"model_type={self._model_type}, "
            f"model_unit={self._model_unit}, "
            f"input_smoothing={self._input_smoothing!r}, "
            f"peak_filtering={self._peak_filtering!r}, "
            f"min_points={self._min_points})"
        )
