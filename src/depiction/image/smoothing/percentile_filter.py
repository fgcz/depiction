from __future__ import annotations

import numpy as np
import scipy.ndimage
from dataclasses import dataclass
from enum import Enum
from numpy.typing import NDArray

from depiction.image.smoothing.base import ChannelWiseSmoothing


# TODO this is a prototype, not optimized and not tested!
# TODO if useful, this can be made a lot more generic

# Some consideration for the intuitions
# - Consider each channel separately, as we have no trust in the correspondences and this will be handled by the
#   regression. (+)
# - If we have a constant value in the window, we can just take that value. (+)
# - If we the center is higher than the surrounding,
#   - we can take a lower value if we are sure that this high intensity is essentially a noisy artifact
#   - we should not take a higher value
#   - but to make this robust we might want to consider using medians...


class KernelShape(Enum):
    Square = "square"
    Circle = "circle"


class KernelFunction(Enum):
    Min = "min"
    AbsMin = "abs_min"


@dataclass(frozen=True)
class PercentileFilter(ChannelWiseSmoothing):
    kernel_size: int = 5
    kernel_shape: KernelShape = KernelShape.Square
    percentile: float = 0

    def smooth_channel(
        self, image_2d: NDArray[np.float64], is_foreground: NDArray[np.bool_]
    ) -> tuple[NDArray[np.float64]]:
        if self.kernel_shape == KernelShape.Square:
            if self.percentile == 0:
                return _eval_abs_min(image_2d, self.kernel_size)
            else:
                return _eval_abs_percentile(image_2d, self.kernel_size, self.percentile)
        elif self.kernel_shape == KernelShape.Circle:
            circle = self.get_circle_kernel_mask()
            return scipy.ndimage.percentile_filter(
                image_2d,
                percentile=np.clip(self.percentile * 100, 0, 100),
                footprint=circle,
            )
        else:
            msg = f"Unknown kernel shape: {self.kernel_shape}"
            raise ValueError(msg)

    def get_circle_kernel_mask(self) -> NDArray[np.bool_]:
        footprint = np.zeros((self.kernel_size, self.kernel_size), dtype=int)
        for x, y in np.ndindex(footprint.shape):
            if (x - self.kernel_size // 2) ** 2 + (y - self.kernel_size // 2) ** 2 <= (self.kernel_size // 2) ** 2:
                footprint[x, y] = True
        return footprint


def _min_by_abs(arr):
    assert arr.ndim == 1
    return arr[np.abs(arr).argmin()]


def _eval_abs_min(array, ws: int):
    result = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            window = array[max(i - ws, 0) : i + ws + 1, max(j - ws, 0) : j + ws + 1]
            result[i, j] = _min_by_abs(window.ravel())
    return result


def _eval_abs_percentile(array, ws: int, percentile: float):
    result = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            window = array[max(i - ws, 0) : i + ws + 1, max(j - ws, 0) : j + ws + 1].ravel()
            result[i, j] = window[np.argsort(np.abs(window))[int(percentile * len(window))]]
    return result


def _eval_abs_percentile_circle(array, ws: int, percentile: float):
    result = np.zeros_like(array)
    radius = ws // 2
    height, width = array.shape[:2]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            window = []
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if di * di + dj * dj <= radius * radius:
                        ni, nj = (i + di) % height, (j + dj) % width
                        window.append(np.abs(array[ni, nj]))

            result[i, j] = np.percentile(window, percentile)

    return result
