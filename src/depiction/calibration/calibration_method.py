from typing import Protocol

from numpy.typing import NDArray
from xarray import DataArray


class CalibrationMethod(Protocol):
    """Defines the interface for a spectrum calibration method."""

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
        """Fits a model to the extracted features of a single spectrum and returns its coefficients.
        If this is not applicable, the implementation can also be performed in the `apply_spectrum_model` method,
        although it would be nicer to consider an adjustment of the interface.
        :param features: a DataArray with the extracted features, with dimensions ["c"]
        :return: a DataArray with the coefficients of the fitted model, with dimensions ["c"] (not necessarily the same)
        """
        ...

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        """Applies the fitted model to the spectrum and returns the calibrated spectrum.
        :param spectrum_mz_arr: m/z values of the spectrum
        :param spectrum_int_arr: intensity values of the spectrum
        :param model_coef: a DataArray with the coefficients of the fitted model, with dimensions ["c"]
        :return: a tuple with the calibrated m/z values and intensity values of the spectrum
        """
        ...
