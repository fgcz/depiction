from dataclasses import dataclass
from enum import Enum

import numpy as np
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper


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
class MinFilter:
    kernel_size: int = 5
    kernel_shape: KernelShape = KernelShape.Square
    percentile: float = 0

    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data = XarrayHelper.ensure_dense(image.data_spatial)
        if self.kernel_shape != KernelShape.Square:
            raise ValueError("Only square kernel is supported for now")

        smoothed_data = np.zeros_like(data.values)
        for c in range(smoothed_data.shape[2]):
            if self.percentile == 0:
                smoothed_data[:, :, c] = _eval_abs_min(data.values[:, :, c], self.kernel_size)
            else:
                smoothed_data[:, :, c] = _eval_abs_percentile(data.values[:, :, c], self.kernel_size, self.percentile)

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


def _eval_abs_percentile(array, ws: int, percentile: float):
    result = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            window = array[max(i - ws, 0) : i + ws + 1, max(j - ws, 0) : j + ws + 1].ravel()
            result[i, j] = window[np.argsort(np.abs(window))[int(percentile * len(window))]]
    return result
