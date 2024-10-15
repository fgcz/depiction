from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import xarray as xr
from loguru import logger

from depiction.image import MultiChannelImage
from depiction.image.xarray_helper import XarrayHelper

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SmoothBilateralFilter:
    diameter: int = 5
    sigma_intensity: float = 5.0
    sigma_spatial: float = 20.0

    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage:
        data = XarrayHelper.ensure_dense(image.data_spatial)
        is_foreground = XarrayHelper.ensure_dense(image.fg_mask)
        dat = xr.apply_ufunc(
            self._smooth_dense_image,
            data,
            is_foreground,
            input_core_dims=[["y", "x"], ["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
        )
        return MultiChannelImage(dat, is_foreground=is_foreground, is_foreground_label=image.is_foreground_label)

    def _smooth_dense_image(self, image_2d: NDArray[float], is_foreground: NDArray[bool]) -> NDArray[float]:
        if not np.issubdtype(image_2d.dtype, np.floating):
            raise ValueError("The input image must be a floating point array.")

        # apply the bilateral filter
        logger.info("Applying bilateral filter")
        smoothed_image = cv2.bilateralFilter(
            np.nan_to_num(image_2d.astype(np.float32)),
            d=self.diameter,
            sigmaColor=self.sigma_intensity,
            sigmaSpace=self.sigma_spatial,
        )
        smoothed_image[~is_foreground] = 0
        return smoothed_image
