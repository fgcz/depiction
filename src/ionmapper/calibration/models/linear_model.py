from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import sklearn.linear_model
import scipy
from numpy.typing import NDArray
import scipy.stats


@dataclass
class LinearModel:
    coef: NDArray[float]

    def __post_init__(self):
        self.coef = np.asarray(self.coef)
        if self.coef.shape != (2,):
            raise ValueError(f"Invalid shape {self.coef=}")

    @property
    def intercept(self) -> float:
        return self.coef[0]

    @property
    def slope(self) -> float:
        return self.coef[1]

    @property
    def is_zero(self) -> bool:
        """Returns True if the model is a constant (and exact) zero function."""
        return self.coef[0] == 0 and self.coef[1] == 0

    def predict(self, x: NDArray[float]) -> float:
        x = np.atleast_1d(x)
        return self.coef[0] + self.coef[1] * x

    @classmethod
    def identity(cls) -> LinearModel:
        return cls([0, 1])

    @classmethod
    def zero(cls) -> LinearModel:
        return cls([0, 0])

    @classmethod
    def fit_lsq(cls, x_arr: NDArray[float], y_arr: NDArray[float]) -> LinearModel:
        """Fits a linear model to the given data using least squares regression."""
        model = sklearn.linear_model.LinearRegression()
        model.fit(x_arr[:, np.newaxis], y_arr[:, np.newaxis])
        return LinearModel(coef=[model.intercept_[0], model.coef_[0, 0]])

    @classmethod
    def fit_siegelslopes(cls, x_arr: NDArray[float], y_arr: NDArray[float]) -> LinearModel:
        """Fits a linear model to the given data using robust Siegel-Slopes regression."""
        slope, intercept = scipy.stats.siegelslopes(y=y_arr, x=x_arr)
        return LinearModel(coef=[intercept, slope])
