from dataclasses import dataclass


@dataclass(frozen=True)
class PixelSize:
    size_x: float
    size_y: float
    unit: str
