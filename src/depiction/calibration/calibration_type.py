from typing import Protocol

from numpy._typing import NDArray
from xarray import DataArray


class CalibrationType(Protocol):
    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        """Extracts a vector of features (dimension ["c"]) from a given, peak picked spectrum.
        For calibration methods which do not involve a feature extraction, an empty DataArray should be returned.
        :param peak_mz_arr: m/z values of the peaks in the spectrum
        :param peak_int_arr: intensity values of the peaks in the spectrum
        :return: a DataArray with the extracted features, with dimensions ["c"]
        """
        return DataArray([], dims=["c"])

    def preprocess_image_features(self, all_features: DataArray) -> DataArray:
        """Preprocesses the extracted features from all spectra in an image.
        For example, image-wide smoothing of the features could be applied here.
        If no preprocessing is necessary, the input DataArray should be returned.
        :param all_features: a DataArray with the extracted features, with dimensions ["i", "c"]
            and coordinates ["i", "x", "y"] for dimension "i"
        :return: a DataArray with the preprocessed features, with dimensions ["i", "c"]
            and coordinates ["i", "x", "y"] for dimension "i"
        """
        return all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        pass

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        pass
