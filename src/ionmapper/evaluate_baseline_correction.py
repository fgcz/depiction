
import numpy as np
import pybaselines
from numpy.typing import NDArray


def _generic_baseline_correction(method):
    class GenericBaselineCorrection:
        def __init__(self, **kwargs) -> None:
            self._kwargs = kwargs

        def evaluate_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
            if len(int_arr) < 10:
                # TODO this avoids a bug in pybaselines, triggered with int_values of length 6 or less.
                print("Length of int_values is less than 10:", len(int_arr))
                return np.zeros_like(int_arr)
            return method(int_arr, **self._kwargs)[0]

        def subtract_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
            baseline = self.evaluate_baseline(mz_arr=mz_arr, int_arr=int_arr)
            return np.maximum(int_arr - baseline, 0)

    return GenericBaselineCorrection


EvaluateMWMVBaselineCorrection = _generic_baseline_correction(method=pybaselines.morphological.mwmv)

EvaluateFABCBaselineCorrection = _generic_baseline_correction(method=pybaselines.classification.fabc)
EvaluateCWTBRBaselineCorrection = _generic_baseline_correction(method=pybaselines.classification.cwt_br)
