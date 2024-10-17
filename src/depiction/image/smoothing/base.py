from __future__ import annotations

import xarray
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper


class ChannelWiseSmoothing:
    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data_input = XarrayHelper.ensure_dense(image.data_spatial)
        is_foreground = XarrayHelper.ensure_dense(image.fg_mask)
        data_result = xarray.apply_ufunc(
            self.smooth_channel,
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

    def smooth_channel(self, image_2d: NDArray[float], is_foreground: NDArray[bool]) -> tuple[NDArray[float]]:
        raise NotImplementedError

    def update_is_foreground(self, data_result: DataArray, is_foreground: DataArray) -> DataArray:
        return is_foreground
