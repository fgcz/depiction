from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Baseline(Protocol):
    def evaluate_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        pass

    def subtract_baseline(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        baseline = self.evaluate_baseline(mz_arr=mz_arr, int_arr=int_arr)
        return np.maximum(int_arr - baseline, 0)
