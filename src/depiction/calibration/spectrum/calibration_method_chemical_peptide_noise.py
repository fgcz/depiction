from typing import Literal

from numpy.typing import NDArray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.calibration.chemical_noise_bg_2019_boskamp_v2 import ChemicalNoiseCalibration
from depiction.parallel_ops import ParallelConfig


class CalibrationMethodChemicalPeptideNoise(CalibrationMethod):
    _lambda_averagine = 1.0 + 4.95e-4

    def __init__(
        self,
        n_mass_intervals: int,
        interpolation_mode: Literal["linear", "cubic_spline", "refit_linear"],
        parallel_config: ParallelConfig,
        use_ppm_space: bool,
    ) -> None:
        self._calibration = ChemicalNoiseCalibration(
            n_mass_intervals=n_mass_intervals,
            interpolation_mode=interpolation_mode,
            parallel_config=parallel_config,
            use_ppm_space=use_ppm_space,
        )

    def extract_spectrum_features(self, peak_mz_arr: NDArray[float], peak_int_arr: NDArray[float]) -> DataArray:
        # shifts_arr, disp_arr, moments_arr = self._calibration.get_moments_approximation(mz_arr=peak_mz_arr,
        #                                                                                int_arr=peak_int_arr)
        ## TODO the only feature we actually need is shifts_arr, so for simplicity i'm just returning that
        # return DataArray(shifts_arr, dims=["c"])
        return DataArray([], dims=["c"])

    def preprocess_image_features(self, all_features: DataArray) -> DataArray:
        # TODO no smoothing applied for now, but could be added (just, avoid duplication with the RegressShift)
        return all_features

    def fit_spectrum_model(self, features: DataArray) -> DataArray:
        # TODO in principle it could be more appropriate to perform the interpolation here, however, that would require
        #    us to provide the information about mz_arr, which also risks breaking the abstraction again
        return features

    def apply_spectrum_model(
        self, spectrum_mz_arr: NDArray[float], spectrum_int_arr: NDArray[float], model_coef: DataArray
    ) -> tuple[NDArray[float], NDArray[float]]:
        res_mz_arr = self._calibration.align_masses(mz_arr=spectrum_mz_arr, int_arr=spectrum_int_arr)
        return res_mz_arr, spectrum_int_arr