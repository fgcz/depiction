from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray


@dataclass
class StratifiedGrid:
    cells_x: int
    cells_y: int

    @cached_property
    def edges_x(self) -> NDArray[int]:
        """Grid edges along x-axis in unit interval domain."""
        eps = 1e-12
        return np.linspace(-eps, 1 + eps, self.cells_x + 1)

    @cached_property
    def edges_y(self) -> NDArray[int]:
        """Grid edges along y-axis in unit interval domain."""
        eps = 1e-12
        return np.linspace(-eps, 1 + eps, self.cells_y + 1)

    def grid_limits_unit(self, index: int) -> tuple[float, float, float, float]:
        """Return the limits of the grid cell corresponding to the given index in the unit interval domain.
        Returns a tuple of (min_x, min_y, max_x, max_y).
        """
        i_y, i_x = divmod(index, self.cells_x)
        return self.edges_x[i_x], self.edges_y[i_y], self.edges_x[i_x + 1], self.edges_y[i_y + 1]

    def grid_limits_scaled(self, index: int, array: DataArray) -> tuple[float, float, float, float]:
        """Return the limits of the grid cell corresponding to the given index in the scaled domain.
        Returns a tuple of (min_x, min_y, max_x, max_y).
        """
        min_x, min_y, max_x, max_y = self.grid_limits_unit(index)
        array_x_min, array_x_max = array.x.values.min(), array.x.values.max()
        array_y_min, array_y_max = array.y.values.min(), array.y.values.max()
        return (
            min_x * (array_x_max - array_x_min) + array_x_min,
            min_y * (array_y_max - array_y_min) + array_y_min,
            max_x * (array_x_max - array_x_min) + array_x_min,
            max_y * (array_y_max - array_y_min) + array_y_min,
        )

    def assign_points(self, array: DataArray) -> dict[int, NDArray[int]]:
        """Assigns points to the grid cells, returning a dictionary of cell index to point indices.
        This method expects array to have a dimension called i which indicates the index of the point which will be
        part of the return value, and coordinates x and y which are used to determine the grid cell assignment.
        """
        if "i" not in array.dims:
            raise ValueError("DataArray must have a dimension 'i'.")
        if ("x" not in array.coords) or ("y" not in array.coords):
            raise ValueError("DataArray must have coordinates 'x' and 'y'.")

        coords_x = array.coords["x"].values
        coords_y = array.coords["y"].values
        print(coords_x)

        assignments = {}
        for i_cell in range(self.cells_x * self.cells_y):
            min_x, min_y, max_x, max_y = self.grid_limits_scaled(i_cell, array)
            assignments[i_cell] = np.where(
                (min_x <= coords_x) & (coords_x < max_x) & (min_y <= coords_y) & (coords_y < max_y)
            )[0]
        return assignments
