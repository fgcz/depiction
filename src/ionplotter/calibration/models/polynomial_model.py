from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PolynomialModel:
    """
    Attributes:
        coef: The coefficients of the polynomial model.
            Uses numpy convention, i.e. p(x) = p[0] * x**deg + ... + p[deg],
            i.e. the coefficient for the highest degree comes first,
            and the intercept comes last.
    """

    coef: NDArray[float]

    def __post_init__(self):
        self.coef = np.asarray(self.coef)

    @property
    def is_zero(self) -> bool:
        """Returns True if the model is the (exact) zero function."""
        return np.all(self.coef == 0)

    @property
    def degree(self) -> int:
        return len(self.coef) - 1

    def predict(self, x: NDArray[float]) -> NDArray[float]:
        x = np.atleast_1d(x)
        return np.polyval(self.coef, x)

    @classmethod
    def identity(cls, degree: int = 1) -> "PolynomialModel":
        return cls([0] * degree + [1])

    @classmethod
    def zero(cls, degree: int = 1) -> "PolynomialModel":
        return cls([0] * (degree + 1))

    @classmethod
    def fit_lsq(cls, x_arr: NDArray[float], y_arr: NDArray[float], degree: int) -> "PolynomialModel":
        """Fits a polynomial model of degree `degree` to the given data using least squares regression."""
        coef = np.polyfit(x_arr, y_arr, deg=degree)
        return PolynomialModel(coef=coef)
