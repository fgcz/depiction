from typing import Optional

import numpy as np
import scipy.signal
import sklearn.metrics


class CalibrationModelLinear:
    def __init__(self, intercept: Optional[float] = None, slope: Optional[float] = None):
        self._intercept = intercept
        self._slope = slope

    def fit(self, mz_match_ref, mz_match_sample):
        """Fits a linear model to the provided m/z values."""
        # TODO consider robust regression like: siegelslopes
        self._slope, self._intercept, _, _, _ = scipy.stats.linregress(x=mz_match_ref, y=mz_match_sample)

    def predict(self, mz_arr: np.ndarray) -> np.ndarray:
        """Returns the predicted m/z values for the provided array."""
        return self._intercept + self._slope * mz_arr

    def fit_predict(
        self,
        mz_full_sample: np.ndarray,
        mz_match_ref: np.ndarray,
        mz_match_sample: np.ndarray,
    ) -> np.ndarray:
        """
        Using pairs of m/z values in mz_match_ref and mz_match_sample this method
        fits a linear model and then returns the m/z values in mz_full_sample
        """
        self.fit(mz_match_ref=mz_match_ref, mz_match_sample=mz_match_sample)
        return self.predict(mz_arr=mz_full_sample)

    @property
    def coefficients(self):
        return {"intercept": self._intercept, "slope": self._slope}

    def copy(self):
        return CalibrationModelLinear(intercept=self._intercept, slope=self._slope)


class EvaluateCalibration:
    def __init__(self, reference_mz: np.ndarray, max_assocation_distance: float):
        self._reference_mz = reference_mz
        self._max_assocation_distance = max_assocation_distance
        # TODO pass this as an argument once there are more options
        self._calibration_model = CalibrationModelLinear()

    def evaluate(self, mz_arr: np.ndarray, int_arr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Returns a new m/z array for the provided spectrum, where the m/z values are calibrated,
        and the coefficients of the model that was used.
        """
        # find matches
        ref_indices, sample_indices = self.compute_matches(mz_arr_sample=mz_arr, mz_arr_ref=self._reference_mz)

        # compute transformed m/z values
        model = self._calibration_model.copy()
        mz_arr = model.fit_predict(
            mz_full_sample=mz_arr,
            mz_match_ref=self._reference_mz[ref_indices],
            mz_match_sample=mz_arr[sample_indices],
        )
        return mz_arr, model.coefficients

    def compute_matches(self, mz_arr_sample: np.ndarray, mz_arr_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the matches between sample and reference arrays."""
        # TODO could be optimized in the future
        distances = sklearn.metrics.pairwise_distances(mz_arr_ref.reshape(-1, 1), mz_arr_sample.reshape(-1, 1))
        matches = distances.argmin(1)

        # remove matches which are above the maximal association distance
        matches[distances[np.arange(len(matches)), matches] > self._max_assocation_distance] = -1

        # construct source and sample pairs
        ref_indices = np.where(matches != -1)[0]
        sample_indices = matches[ref_indices]

        return ref_indices, sample_indices
