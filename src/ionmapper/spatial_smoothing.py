from dataclasses import dataclass
from typing import Sequence, Literal

import numpy as np
from numpy.typing import NDArray
import scipy.ndimage


@dataclass
class SpatialSmoothing:
    # TODO might need further definition in the future, in particular for n_channels support
    #      and maybe, support to specify the smoothing also in terms of real world units
    sigma: float | Sequence[float]
    background_fill_mode: Literal[None, "nearest"] = None
    background_value: float = 0.0

    def smooth_values(self, values: NDArray[float], coordinates: NDArray[int]) -> NDArray[float]:
        values_spatial = self.flat_to_grid(
            values=values,
            coordinates=coordinates,
            background_value=self.background_value,
        )
        values_spatial = self.fill_background(values_spatial=values_spatial)
        smoothed_spatial = scipy.ndimage.gaussian_filter(values_spatial, sigma=self.sigma)
        return self.grid_to_flat(values_grid=smoothed_spatial, coordinates=coordinates)

    @staticmethod
    def flat_to_grid(values: NDArray[float], coordinates: NDArray[int], background_value: float) -> NDArray[float]:
        """
        values: shape (n_values,) or (n_values, n_channels)
        coordinates: shape (n_values, n_dims)
        """
        coordinates_min = coordinates.min(axis=0)
        coordinates_extent = coordinates.max(axis=0) - coordinates_min + 1
        coordinates_shifted = coordinates - coordinates_min

        dtype = np.promote_types(values.dtype, np.obj2sctype(type(background_value)))
        values_grid = np.full(coordinates_extent, fill_value=background_value, dtype=dtype)
        values_grid[tuple(coordinates_shifted.T)] = values

        return values_grid

    @staticmethod
    def grid_to_flat(values_grid: NDArray[float], coordinates: NDArray[int]) -> NDArray[float]:
        coordinates_min = coordinates.min(axis=0)
        coordinates_offset = coordinates - coordinates_min
        return values_grid[tuple(coordinates_offset.T)]

    def fill_background(self, values_spatial: NDArray[float]) -> NDArray[float]:
        if self.background_fill_mode == "nearest":
            if np.isnan(self.background_value):
                bg_mask = np.isnan(values_spatial)
            else:
                bg_mask = values_spatial == self.background_value
            indices = scipy.ndimage.distance_transform_edt(
                bg_mask,
                return_distances=False,
                return_indices=True,
            )
            return values_spatial[tuple(indices)]
        elif self.background_fill_mode:
            raise ValueError(f"Unknown fill_background: {self.background_fill_mode}")
        else:
            return values_spatial
