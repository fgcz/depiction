from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage


class CalibrationMethodDummy(CalibrationMethod):
    """Returns the input data and creates some dummy coefficients to ensure compatibility."""

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        return DataArray([0], dims=["c"])

    def preprocess_image_features(self, all_features: MultiChannelImage) -> MultiChannelImage:
        return all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        return features

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        return spectrum_mz_arr, spectrum_int_arr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
