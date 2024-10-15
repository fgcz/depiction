from typing import Literal

from loguru import logger
from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.calibration.chemical_noise_bg_2019_boskamp_v2 import ChemicalNoiseCalibration
from depiction.image import MultiChannelImage


class CalibrationMethodChemicalPeptideNoise(CalibrationMethod):
    _lambda_averagine = 1.0 + 4.95e-4

    def __init__(
        self,
        n_mass_intervals: int,
        interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"],
        use_ppm_space: bool,
    ) -> None:
        self._calibration = ChemicalNoiseCalibration(
            n_mass_intervals=n_mass_intervals,
            interpolation_mode=interpolation_mode,
            use_ppm_space=use_ppm_space,
        )

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        # shifts_arr, disp_arr, moments_arr = self._calibration.get_moments_approximation(mz_arr=peak_mz_arr,
        #                                                                                int_arr=peak_int_arr)
        ## TODO the only feature we actually need is shifts_arr, so for simplicity i'm just returning that
        # return DataArray(shifts_arr, dims=["c"])
        return DataArray([], dims=["c"])

    def preprocess_image_features(self, all_features: MultiChannelImage) -> MultiChannelImage:
        # TODO no smoothing applied for now, but could be added (just, avoid duplication with the RegressShift)
        return all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        # TODO in principle it could be more appropriate to perform the interpolation here, however, that would require
        #    us to provide the information about mz_arr, which also risks breaking the abstraction again
        return features

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        if len(spectrum_mz_arr) < self._calibration.n_mass_intervals:
            logger.warning(f"Spectrum too small to calculate moments approximation. (n={len(spectrum_mz_arr)})")
            return spectrum_mz_arr, spectrum_int_arr
        else:
            res_mz_arr = self._calibration.align_masses(mz_arr=spectrum_mz_arr, int_arr=spectrum_int_arr)
            return res_mz_arr, spectrum_int_arr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(calibration={self._calibration!r})"
