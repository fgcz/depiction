# TODO very experimental, it's not fully clear if it is the intended approach
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy
import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray

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

    def smooth(self, image: DataArray, bg_value: float = 0.0) -> DataArray:
        image = image.transpose("y", "x", "c")
        image = image.astype(np.promote_types(image.dtype, np.obj2sctype(type(bg_value))))
        image = XarrayHelper.ensure_dense(image)
        image = xr.apply_ufunc(
            self._smooth_dense,
            image,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            kwargs={"bg_value": bg_value},
        )
        return image.transpose("y", "x", "c")

    def _smooth_dense(self, image_2d: NDArray[float], bg_value: float) -> NDArray[float]:
        """Applies the spatial smoothing to the provided 2D image."""
        if not np.issubdtype(image_2d.dtype, np.floating):
            raise ValueError("The input image must be a floating point array.")

        # Get an initial kernel, and mask of the missing values.
        kernel = self.gaussian_kernel
        is_missing = (image_2d == bg_value) | (np.isnan(bg_value) & np.isnan(image_2d))

        # Apply the kernel to the image.
        smoothed_image = scipy.signal.convolve(np.nan_to_num(image_2d), kernel, mode="same")

        # Apply the kernel counting the sum of the weights, so we can normalize the data.
        kernel_sum_image = scipy.signal.convolve((~is_missing).astype(float), kernel, mode="same")
        # Values are zero, when a pixel and all its neighbors are missing.
        kernel_sum_image[np.abs(kernel_sum_image) < 1e-10] = 1

        # TODO double check this does not mess up the scaling of the values

        # Normalize the image, and set the missing values to NaN.
        result_image = smoothed_image / kernel_sum_image

        if not self.use_interpolation:
            result_image[is_missing] = bg_value

        # Return the result.
        return result_image

    @cached_property
    def gaussian_kernel(self) -> NDArray[float]:
        """Returns the gaussian kernel to use for smoothing. The kernel is normalized to sum to 1."""
        kernel_1d = scipy.signal.windows.gaussian(self.kernel_size, self.kernel_std)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)
