from dataclasses import dataclass
from enum import Enum

import numpy as np
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper


# TODO this is a prototype, not optimized and not tested!
# TODO if useful, this can be made a lot more generic


class KernelShape(Enum):
    Square = "square"
    Circle = "circle"


class KernelFunction(Enum):
    Min = "min"
    AbsMin = "abs_min"


@dataclass(frozen=True)
class MinFilter:
    kernel_size: int = 5
    kernel_shape: KernelShape = KernelShape.Square

    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data = XarrayHelper.ensure_dense(image.data_spatial)
        if self.kernel_shape != KernelShape.Square:
            raise ValueError("Only square kernel is supported for now")

        smoothed_data = np.zeros_like(data.values)
        for c in range(smoothed_data.shape[2]):
            smoothed_data[:, :, c] = _eval_abs_min(data.values[:, :, c], self.kernel_size)

        return MultiChannelImage(
            DataArray(smoothed_data, dims=data.dims, coords=data.coords),
            is_foreground=image.fg_mask,
            is_foreground_label=image.is_foreground_label,
        )


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
