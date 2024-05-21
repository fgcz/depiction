# TODO very experimental, it's not fully clear if it is the intended approach
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import scipy

from ionmapper.spatial_smoothing import SpatialSmoothing


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

    def smooth_sparse(self, sparse_values: NDArray[float], coordinates: NDArray[int]) -> NDArray[float]:
        """
        Applies the spatial smoothing to the provided image, and returns the smoothed image
        (again in the sparse format).
        Internally, this currently creates a dense image so it cannot be used to avoid memory limitations right now.
        """
        image_2d = SpatialSmoothing.flat_to_grid(
            sparse_values=sparse_values,
            coordinates=coordinates,
            background_value=np.nan,
        )
        smoothed = self.smooth_dense(image_2d=image_2d)
        return SpatialSmoothing.grid_to_flat(values_grid=smoothed, coordinates=coordinates)

    def smooth_dense(self, image_2d: NDArray[float]) -> NDArray[float]:
        """Applies the spatial smoothing to the provided 2D image."""
        if not np.issubdtype(image_2d.dtype, np.floating):
            raise ValueError("The input image must be a floating point array.")

        # Get an initial kernel, and mask of the missing values.
        kernel = self.gaussian_kernel
        is_missing = np.isnan(image_2d)

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
            result_image[is_missing] = np.nan

        # Return the result.
        return result_image

    def smooth_sparse_multi_channel(self, sparse_values: NDArray[float], coordinates: NDArray[int]) -> NDArray[float]:
        """
        Returns the result of a sparse array with multiple channels smoothed independently.
        :param sparse_values: a 2D array with shape (n_values, n_channels)
        :param coordinates: a 2D array with shape (n_values, 2)
        """
        _, n_channels = sparse_values.shape
        return np.stack(
            [
                self.smooth_sparse(sparse_values=sparse_values[:, i_channel], coordinates=coordinates)
                for i_channel in range(n_channels)
            ],
            axis=1,
        )

    def smooth_dense_multi_channel(self, image_2d: NDArray[float]) -> NDArray[float]:
        """
        Returns the result of a 2D array with multiple channels, with each channel smoothed independently.
        :param image_2d: a 3D array with shape (n_rows, n_columns, n_channels)
        """
        return np.stack(
            [self.smooth_dense(image_2d[:, :, i_channel]) for i_channel in range(image_2d.shape[2])],
            axis=2,
        )

    @cached_property
    def gaussian_kernel(self) -> NDArray[float]:
        """Returns the gaussian kernel to use for smoothing. The kernel is normalized to sum to 1."""
        kernel_1d = scipy.signal.windows.gaussian(self.kernel_size, self.kernel_std)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)
