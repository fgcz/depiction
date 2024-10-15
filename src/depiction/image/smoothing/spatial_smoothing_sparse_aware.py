# TODO very experimental, it's not fully clear if it is the intended approach
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy
import xarray as xr
from numpy.typing import NDArray

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper


@dataclass(frozen=True)
class SpatialSmoothingSparseAware:
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

    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data_input = XarrayHelper.ensure_dense(image.data_spatial)
        is_foreground = XarrayHelper.ensure_dense(image.fg_mask)
        data_result = xr.apply_ufunc(
            self._smooth_dense,
            data_input,
            is_foreground,
            input_core_dims=[["y", "x"], ["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
        )
        if self.use_interpolation:
            is_foreground[:] = True
        return MultiChannelImage(
            data_result, is_foreground=is_foreground, is_foreground_label=image.is_foreground_label
        )

    def _smooth_dense(self, image_2d: NDArray[float], is_foreground: NDArray[float]) -> NDArray[float]:
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

    @cached_property
    def gaussian_kernel(self) -> NDArray[float]:
        """Returns the gaussian kernel to use for smoothing. The kernel is normalized to sum to 1."""
        kernel_1d = scipy.signal.windows.gaussian(self.kernel_size, self.kernel_std)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)
