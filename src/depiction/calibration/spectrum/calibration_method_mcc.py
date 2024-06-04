import numpy as np
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from numpy.typing import NDArray
from statsmodels.robust.norms import HuberT
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.calibration.spectrum.calibration_smoothing import smooth_image_features


class CalibrationMethodMCC(CalibrationMethod):
    def __init__(
        self,
        model_smoothing_activated: bool = True,
        model_smoothing_kernel_size: int = 27,
        model_smoothing_kernel_std: float = 10.0,
        max_pairwise_distance: float = 500,
    ) -> None:
        self._model_smoothing_activated = model_smoothing_activated
        self._model_smoothing_kernel_size = model_smoothing_kernel_size
        self._model_smoothing_kernel_std = model_smoothing_kernel_std
        self._max_pairwise_distance = max_pairwise_distance

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        l_none = 1.000482
        # c_0 = 0.029
        # c_1 = 4.98*10e-4

        # Compute all differences for elements in peak_mz_arr amd store in a DataArray
        delta = scipy.spatial.distance.pdist(np.expand_dims(peak_mz_arr, 1), metric="cityblock")

        # Get all distances smaller than the max pairwise distance
        delta = delta[delta < self._max_pairwise_distance]

        # Compute delta_lambda for each x
        delta_lambda = np.zeros_like(delta)
        for i, mi in enumerate(delta):
            term1 = mi % l_none
            if term1 < 0.5:
                delta_lambda[i] = term1
            else:
                delta_lambda[i] = -1 + term1

        sorted_indices = np.argsort(delta)
        delta = delta[sorted_indices]
        delta_lambda = delta_lambda[sorted_indices]

        # Add a constant term with the intercept set to zero
        X = delta.reshape(-1, 1)

        # Fit the model
        robust_model = sm.RLM(delta_lambda, X, M=HuberT())
        results = robust_model.fit()
        slope = results.params[0]
        peak_mz_corrected = peak_mz_arr * (1 - slope)

        delta_intercept = np.zeros_like(peak_mz_corrected)
        for i, mi in enumerate(peak_mz_corrected):
            term1 = mi % l_none
            if term1 < 0.5:
                delta_intercept[i] = term1
            else:
                delta_intercept[i] = -1 + term1

        intercept_coef = stats.trim_mean(delta_intercept, 0.3)
        return DataArray([intercept_coef, slope], dims=["c"])

    def preprocess_image_features(self, all_features: DataArray) -> DataArray:
        if self._model_smoothing_activated:
            return smooth_image_features(
                all_features=all_features,
                kernel_size=self._model_smoothing_kernel_size,
                kernel_std=self._model_smoothing_kernel_std,
            )
        else:
            return all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        return features

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        spectrum_corrected = spectrum_mz_arr * (1 - model_coef.values[1]) + model_coef.values[0]
        return spectrum_corrected
