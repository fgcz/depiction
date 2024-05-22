from dataclasses import dataclass
from typing import Literal
from collections.abc import Sequence

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

    def smooth_sparse(self, sparse_values: NDArray[float], coordinates: NDArray[int]) -> NDArray[float]:
        """Returns a spatially-smoothed copy of the values, provided in flat/sparse format.
        :param sparse_values: shape (n_values,) or (n_values, n_channels)
        :param coordinates: shape (n_values, 2)
        """
        values_spatial = self.flat_to_grid(
            sparse_values=sparse_values,
            coordinates=coordinates,
            background_value=self.background_value,
        )
        values_spatial = self.fill_background(values_spatial=values_spatial)
        smoothed_spatial = self.smooth_dense(image_2d=values_spatial)
        return self.grid_to_flat(values_grid=smoothed_spatial, coordinates=coordinates)

    def smooth_dense(self, image_2d: NDArray[float]) -> NDArray[float]:
        """Returns a spatially-smoothed copy of the values, provided as a grid.
        # TODO define and test properly
        :param image_2d: shape (n_rows, n_cols) or (n_rows, n_cols, n_channels)
        """
        return scipy.ndimage.gaussian_filter(image_2d, sigma=self.sigma)

    # TODO extract into dedicated module
    @staticmethod
    def flat_to_grid(sparse_values: NDArray[float], coordinates: NDArray[int], background_value: float) -> NDArray[float]:
        """
        values: shape (n_values,) or (n_values, n_channels)
        coordinates: shape (n_values, n_dims)
        """
        coordinates_min = coordinates.min(axis=0)
        coordinates_extent = coordinates.max(axis=0) - coordinates_min + 1
        coordinates_shifted = coordinates - coordinates_min

        dtype = np.promote_types(sparse_values.dtype, np.obj2sctype(type(background_value)))
        values_grid = np.full(coordinates_extent, fill_value=background_value, dtype=dtype)
        values_grid[tuple(coordinates_shifted.T)] = sparse_values

        return values_grid

    # TODO extract into dedicated module
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
