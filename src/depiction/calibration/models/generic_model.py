from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class GenericModel(Protocol):
    @property
    def coef(self) -> NDArray[np.float64]: ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @classmethod
    def identity(cls) -> "GenericModel": ...

    @classmethod
    def zero(cls) -> "GenericModel": ...
