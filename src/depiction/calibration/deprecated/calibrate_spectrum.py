# TODO superseded by CalibrateSpectrumRegressShift

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from depiction.spectrum.peak_picking.basic_peak_picker import BasicPeakPicker
from depiction.calibration.models.linear_model import LinearModel
from depiction.calibration.models.polynomial_model import PolynomialModel
from depiction.calibration.deprecated.reference_distance_estimator import (
    ReferenceDistanceEstimator,
)
from depiction.spectrum.peak_filtering import PeakFilteringType

# TODO refactor this code once development is more advanced
class CalibrateSpectrum:
    """
    Computes the calibration model for a single spectrum.
    :param reference_mz: m/z values of the reference spectrum
    :param peak_picker: peak picker to use
    :param n_candidates: number of candidates to consider for each peak to consider
    :param model_type: either 'linear', 'poly_<degree>' or 'linear_siegelslopes'
    :param distance_limit: maximum distance between a reference peak and a sample peak to be considered
    """

    def __init__(
        self,
        reference_mz: NDArray[float],
        peak_picker: BasicPeakPicker,
        n_candidates: int = 3,
        model_type: str = "linear",
        distance_limit: float = 2.0,
    ) -> None:
        self._ref_estimator = ReferenceDistanceEstimator(reference_mz=reference_mz, n_candidates=n_candidates)
        self._model_type = model_type
        self._model_class = PolynomialModel if model_type.startswith("poly_") else LinearModel
        self._distance_limit = distance_limit
        self._peak_picker = peak_picker

    @staticmethod
    def fit_model(x: NDArray[float], y: NDArray[float], model_type: str) -> Union[LinearModel, PolynomialModel]:
        if len(x) < 3:
            # If there are not enough points, return a zero model.
            if model_type.startswith("poly_"):
                model_class = PolynomialModel
            elif model_type.startswith("linear"):
                model_class = LinearModel
            else:
                raise ValueError(f"Unknown {model_type=}")
            model = model_class.zero()
        elif model_type == "linear":
            model = LinearModel.fit_lsq(x_arr=x, y_arr=y)
        elif model_type.startswith("poly_"):
            degree = int(model_type.split("_")[1])
            model = PolynomialModel.fit_lsq(x_arr=x, y_arr=y, degree=degree)
        elif model_type == "linear_siegelslopes":
            model = LinearModel.fit_siegelslopes(x_arr=x, y_arr=y)
        else:
            raise ValueError(f"Unknown {model_type=}")
        return model

    def calibrate_spectrum(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> LinearModel | PolynomialModel:
        """Computes the calibration model for a single spectrum."""
        # get the mz-distance pairs
        problem_mz, problem_distance = self._get_mz_distance_pairs(mz_arr=mz_arr, int_arr=int_arr)

        # Fit the model
        return self.fit_model(x=problem_mz, y=problem_distance, model_type=self._model_type)

    def _get_mz_distance_pairs(
        self, mz_arr: NDArray[float], int_arr: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float]]:
        # TODO remove function later (if not used)
        mz_peaks = self._peak_picker.pick_peaks_mz(mz_arr=mz_arr, int_arr=int_arr)
        if len(mz_peaks) == 0:
            return np.array([]), np.array([])
        distances, closest_indices = self._ref_estimator.compute_distances_for_peaks(mz_peaks=mz_peaks)
        median_distance = np.median(distances[:, self._ref_estimator.closest_index])

        # Pick for every reference, the distance which is closest to the median distance.
        problem_mz = self._ref_estimator.reference_mz
        problem_distance = distances[
            np.arange(distances.shape[0]),
            np.argmin(np.abs(distances - median_distance), axis=1),
        ]

        # Remove all points that are too far away.
        within_range = np.abs(problem_distance) <= self._distance_limit
        return problem_mz[within_range], problem_distance[within_range]

    @classmethod
    def calibrate_from_config(
        cls,
        peak_picker: BasicPeakPicker,
        peak_filtering,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
        reference_mz_arr: NDArray[float],
        model_type: str,
        distance_limit: float,
        return_info: bool = False,
        prune_bad_limit: float = 5.0,
    ) -> Union[
        LinearModel,
        PolynomialModel,
        tuple[Union[LinearModel, PolynomialModel], dict[str, Any]],
    ]:
        # TODO experimental method, for quick testing/comparison
        #      needs to be properly integrated later (maybe it is also indicate of the changes which are needed still)
        problem_mz, problem_distance = cls.get_matches_from_config(
            peak_picker=peak_picker,
            peak_filtering=peak_filtering,
            mz_arr=mz_arr,
            int_arr=int_arr,
            reference_mz_arr=reference_mz_arr,
            distance_limit=distance_limit,
        )
        model, pruned_limit = cls.fit_calibration_model_for_problem(
            model_type=model_type,
            problem_mz=problem_mz,
            problem_distance=problem_distance,
            mz_arr=mz_arr,
            prune_bad_limit=prune_bad_limit,
        )

        # return the model
        if return_info:
            info = {"n_matches": len(problem_mz), "pruned_limit": pruned_limit}
            return model, info
        else:
            return model

    @classmethod
    def fit_calibration_model_for_problem(
        cls,
        model_type: str,
        problem_mz: NDArray[float],
        problem_distance: NDArray[float],
        mz_arr: NDArray[float],
        prune_bad_limit,
    ):
        # Fit the model
        model = cls.fit_model(x=problem_mz, y=problem_distance, model_type=model_type)
        pruned_limit = None
        model_class = PolynomialModel if model_type.startswith("poly_") else LinearModel
        if prune_bad_limit:
            # if this is not zero/None then we check 3 values for the model and the distance, if it's bigger than this
            # limit the model will be rejected
            mz_values = [mz_arr[0], mz_arr[len(mz_arr) // 2], mz_arr[-1]]
            distance_values = [model.predict(mz) for mz in mz_values]
            if np.any(np.abs(distance_values) > prune_bad_limit):
                model = model_class.zero()
                pruned_limit = np.max(np.abs(distance_values))
        # TODO it would also make sense to reject models based on general goodness-of-fit
        return model, pruned_limit

    @staticmethod
    def match_peaks_to_references(
        spectrum_mz_arr: NDArray[float],
        spectrum_int_arr: NDArray[float],
        peak_idx_arr: NDArray[int],
        reference_mz_arr: NDArray[float],
        distance_limit: float,
    ) -> tuple[NDArray[float], NDArray[float]]:
        # match each reference to its closest peak
        if len(peak_idx_arr) < 3:
            return np.array([]), np.array([])

        ref_estimator = ReferenceDistanceEstimator(reference_mz=reference_mz_arr, n_candidates=3)
        distances, closest_indices = ref_estimator.compute_distances_for_peaks(mz_peaks=spectrum_mz_arr[peak_idx_arr])
        median_distance = np.median(distances[:, ref_estimator.closest_index])
        # Pick for every reference, the distance which is closest to the median distance.
        problem_mz = ref_estimator.reference_mz
        problem_distance = distances[
            np.arange(distances.shape[0]),
            np.argmin(np.abs(distances - median_distance), axis=1),
        ]

        # drop points that are too far away
        within_range = np.abs(problem_distance) <= distance_limit
        if not np.any(within_range):
            return np.array([]), np.array([])
        else:
            return problem_mz[within_range], problem_distance[within_range]

    @classmethod
    def get_matches_from_config(
        cls,
        peak_picker: BasicPeakPicker,
        peak_filtering: PeakFilteringType,
        mz_arr: NDArray[float],
        int_arr: NDArray[float],
        reference_mz_arr: NDArray[float],
        distance_limit: float,
    ) -> tuple[NDArray[float], NDArray[float]]:
        # TODO experimental method, for quick testing/comparison
        #      needs to be properly integrated later (maybe it is also indicate of the changes which are needed still)
        # pick peaks
        spectrum_mz_arr = mz_arr
        spectrum_int_arr = int_arr
        peak_idx_arr = peak_picker.pick_peaks_index(mz_arr=spectrum_mz_arr, int_arr=spectrum_int_arr)

        # filter peaks
        if len(peak_idx_arr) >= 3:
            peak_idx_arr = peak_filtering.filter_index_peaks(
                spectrum_mz_arr=spectrum_mz_arr,
                spectrum_int_arr=spectrum_int_arr,
                peak_idx_arr=peak_idx_arr,
            )

        # match each reference to its closest peak
        return cls.match_peaks_to_references(
            spectrum_mz_arr=spectrum_mz_arr,
            spectrum_int_arr=spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
            reference_mz_arr=reference_mz_arr,
            distance_limit=distance_limit,
        )
