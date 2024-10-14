from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import scipy.ndimage
import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image.xarray_helper import XarrayHelper


@dataclass
class SpatialSmoothing:
    # TODO might need further definition in the future, in particular for n_channels support
    #      and maybe, support to specify the smoothing also in terms of real world units
    sigma: float | Sequence[float]
    background_fill_mode: Literal[None, "nearest"] = None

    def smooth(self, image: DataArray) -> DataArray:
        image = image.transpose("y", "x", "c")
        image = XarrayHelper.ensure_dense(image)
        image = self._fill_background(image)
        image = xr.apply_ufunc(
            self._smooth_single_channel,
            image,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
        )
        return image.transpose("y", "x", "c")

    def _smooth_single_channel(self, values: NDArray[float]) -> NDArray[float]:
        print(type(values))
        return scipy.ndimage.gaussian_filter(values, sigma=self.sigma)

    def _fill_background(self, values: NDArray[float]) -> NDArray[float]:
        return xr.apply_ufunc(
            self._fill_background_single_channel,
            values,
            input_core_dims=[["y", "x"]],
            output_core_dims=[["y", "x"]],
            vectorize=True,
        )

    def _fill_background_single_channel(self, values: NDArray[float]) -> NDArray[float]:
        if self.background_fill_mode == "nearest":
            bg_mask = values == 0
            indices = scipy.ndimage.distance_transform_edt(
                bg_mask,
                return_distances=False,
                return_indices=True,
            )
            return values[tuple(indices)]
        elif self.background_fill_mode:
            raise ValueError(f"Unknown fill_background: {self.background_fill_mode}")
        else:
            return values
