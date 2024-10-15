from dataclasses import dataclass

import scipy.ndimage
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper


# TODO if useful, this can be made a lot more generic


class KernelShape:
    Square = "square"
    Circle = "circle"


@dataclass(frozen=True)
class MinFilter:
    kernel_size: int = 5
    kernel_shape: KernelShape = KernelShape.Square

    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data = XarrayHelper.ensure_dense(image.data_spatial)
        if self.kernel_shape != KernelShape.Square:
            raise ValueError("Only square kernel is supported for now")
        # TODO i want to try the circular kernel later, this impl is just for quick prototyping (as it does not support
        #      a circular window)
        smoothed = scipy.ndimage.minimum_filter(data, size=self.kernel_size, axes=(0, 1))
        return MultiChannelImage(
            DataArray(smoothed, dims=data.dims, coords=data.coords),
            is_foreground=image.fg_mask,
            is_foreground_label=image.is_foreground_label,
        )
