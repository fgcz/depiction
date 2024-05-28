from typing import Protocol

from numpy._typing import NDArray
from xarray import DataArray


class CalibrationType(Protocol):
    def extract_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        pass

    def preprocess_features(self, all_features: DataArray) -> DataArray:
        pass

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        pass

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        pass
