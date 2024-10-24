import numpy as np
import xarray
from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from depiction.image import MultiChannelImage


class CalibrationMethodConstantGlobalShift(CalibrationMethod):
    """Computes a constant global shift across all spectra and applies it.

    This is a very naive method that is mainly used for a fair comparison of non-targeted calibration methods.
    """

    def __init__(self, ref_mz_arr: NDArray[float], max_distance: float = 2.0, max_distance_unit: str = "mz") -> None:
        self._ref_mz_arr = ref_mz_arr
        self._max_distance = max_distance
        self._max_distance_unit = max_distance_unit

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        distances_mz = ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
            ref_mz_arr=self._ref_mz_arr,
            max_distance=self._max_distance,
            max_distance_unit=self._max_distance_unit,
        )
        return DataArray(distances_mz, dims=["c"])

    def preprocess_image_features(self, all_features: MultiChannelImage) -> MultiChannelImage:
        # we compute the actual global distance here
        global_distance = np.nanmedian(all_features.data_flat.data.ravel())
        # return one value per spectrum
        return MultiChannelImage(
            data=xarray.full_like(all_features.data_spatial.isel(c=[0]), global_distance),
            is_foreground=all_features.fg_mask,
            is_foreground_label=all_features.is_foreground_label,
        )

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        return features

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        # subtract the global distance from the m/z values
        [global_distance] = model_coef.values
        return spectrum_mz_arr - global_distance, spectrum_int_arr

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_distance={self._max_distance}, max_distance_unit={self._max_distance_unit})"
        )
