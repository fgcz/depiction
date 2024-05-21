from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ConstantModel:
    """A constant function that can be fitted. (Mainly for testing purposes.)"""

    value: float

    @property
    def coef(self) -> NDArray[float]:
        return np.array([self.value])

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    def predict(self, x: np.ndarray) -> NDArray[float]:
        return np.full_like(x, self.value)

    @classmethod
    def identity(cls) -> ConstantModel:
        raise ValueError("Cannot create identity model for constant function.")

    @classmethod
    def zero(cls) -> ConstantModel:
        return cls(value=0)

    @classmethod
    def fit_mean(cls, x_arr: NDArray[float], y_arr: NDArray[float]) -> ConstantModel:
        return ConstantModel(value=np.mean(y_arr))

    @classmethod
    def fit_median(cls, x_arr: NDArray[float], y_arr: NDArray[float]) -> ConstantModel:
        return ConstantModel(value=np.median(y_arr))
