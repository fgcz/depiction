# TODO very experimental, it's not fully clear if it is the intended approach
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy
import xarray
import xarray as xr
from numpy.typing import NDArray

from depiction.image.smoothing.base import ChannelWiseSmoothing


@dataclass(frozen=True)
class SpatialSmoothingSparseAware(ChannelWiseSmoothing):
    """An experimental variant of 2D spatial smoothing that can handle nan values/boundaries.
    :param kernel_size: The size of the Gaussian kernel to use for smoothing.
    :param kernel_std: The standard deviation of the Gaussian kernel to use for smoothing.
    :param use_interpolation:
        If True, missing values (either explicit nan in sparse representation or nans in dense representation)
            are interpolated from the surrounding values,
        if False, they are preserved as nan.
    """

    kernel_size: int
    kernel_std: float
    use_interpolation: bool = False

    def smooth_channel(self, image_2d: NDArray[float], is_foreground: NDArray[float]) -> NDArray[float]:
        image_2d = image_2d.astype(float)

        # Get an initial kernel
        kernel = self.gaussian_kernel

        # Apply the kernel to the image.
        smoothed_image = scipy.signal.convolve(np.nan_to_num(image_2d), kernel, mode="same")

        # Apply the kernel counting the sum of the weights, so we can normalize the data.
        kernel_sum_image = scipy.signal.convolve(is_foreground.astype(float), kernel, mode="same")
        # Values are zero, when a pixel and all its neighbors are missing (but they are masked anyways).
        kernel_sum_image[np.abs(kernel_sum_image) < 1e-10] = 1

        # Normalize the image, and set the missing values to NaN.
        result_image = smoothed_image / kernel_sum_image

        if not self.use_interpolation:
            result_image[~is_foreground] = 0.0

        # Return the result.
        return result_image

    def update_is_foreground(self, data_result: xr.DataArray, is_foreground: xr.DataArray) -> xr.DataArray:
        return xarray.ones_like(is_foreground, dtype=bool) if self.use_interpolation else is_foreground

    @cached_property
    def gaussian_kernel(self) -> NDArray[float]:
        """Returns the gaussian kernel to use for smoothing. The kernel is normalized to sum to 1."""
        kernel_1d = scipy.signal.windows.gaussian(self.kernel_size, self.kernel_std)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)
