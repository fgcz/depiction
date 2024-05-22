from typing import Protocol, runtime_checkable

from numpy.typing import NDArray


@runtime_checkable
class GenericModel(Protocol):
    @property
    def coef(self) -> NDArray[float]: ...

    def predict(self, x: NDArray[float]) -> NDArray[float]: ...

    @classmethod
    def identity(cls) -> "GenericModel": ...

    @classmethod
    def zero(cls) -> "GenericModel": ...
