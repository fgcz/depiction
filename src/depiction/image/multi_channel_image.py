from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray
from xarray import DataArray

from depiction.image.sparse_representation import SparseRepresentation


class MultiChannelImage:
    def __init__(self, data: DataArray) -> None:
        self._data = data.transpose("y", "x", "c")
        if "bg_value" not in self._data.attrs:
            raise ValueError("The bg_value attribute must be set.")

    @classmethod
    def from_numpy_sparse(
        cls, values: np.ndarray, coordinates: np.ndarray, channel_names: list[str] | None, bg_value: float = 0.0
    ) -> MultiChannelImage:
        """Creates a MultiChannelImage instance from numpy arrays providing values and coordinates."""
        data = SparseRepresentation.sparse_to_dense(
            sparse_values=DataArray(values, dims=("i", "c")),
            coordinates=DataArray(coordinates, dims=("i", "d")),
            bg_value=bg_value,
        )
        data.attrs["bg_value"] = bg_value
        channel_names = list(channel_names) if channel_names is not None else None
        if channel_names:
            data.coords["c"] = channel_names
        return cls(data=data)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._data.sizes["c"]

    # TODO sparse_values, sparse_coordinates - these are currently widely used which i guess is a problem

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the values."""
        return self._data.dtype

    @property
    def bg_value(self) -> int | float:
        """Returns the background value."""
        return self._data.attrs["bg_value"]

    @cached_property
    def bg_mask(self) -> DataArray:
        return ((self._data == self.bg_value) | (self._data.isnull() & np.isnan(self.bg_value))).all(dim="c")

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns width and height of the image."""
        # TODO reconsider this method (adding it now for compatibility)
        return self._data.sizes["x"], self._data.sizes["y"]

    @property
    def channel_names(self) -> list[str]:
        """Returns the names of the channels."""
        # TODO consider renaming to `channels`
        return [str(c) for c in self._data.coords["c"].values.tolist()]

    @property
    def data_spatial(self) -> DataArray:
        """Returns the underlying data, in its spatial form, i.e. dimensions (y, x, c)."""
        return self._data

    @property
    def data_flat(self) -> DataArray:
        """Returns the underlying data, in its flat form, i.e. dimensions (i, c), omitting any background values."""
        return self._data.where(~self.bg_mask).stack(i=("y", "x")).dropna(dim="i")

    # TODO from_dense_array

    # TODO get_single_channel_dense_array

    def retain_channels(self, channel_indices: Sequence[int]) -> MultiChannelImage:
        """Returns a copy with only the specified channels retained."""
        return MultiChannelImage(data=self._data.isel(c=channel_indices))

    # TODO save_single_channel_image... does it belong here or into plotter?

    # TODO to_dense_xarray
    # TODO from_dense_xarray

    def write_hdf5(self, path: Path) -> None:
        """Writes the image to a HDF5 file (actually NETCDF4)."""
        self._data.to_netcdf(path, format="NETCDF4")

    @classmethod
    def read_hdf5(cls, path: Path) -> MultiChannelImage:
        """Reads a MultiChannelImage from a HDF5 file (assuming it contains NETCDF data)."""
        return cls(data=xarray.open_dataarray(path))

    # TODO is_valid_hdf5

    # TODO combine_in_parallel, combine_sequentially: consider moving this somewhere else

    def with_channel_names(self, channel_names: Sequence[str]) -> MultiChannelImage:
        """Returns a copy with the specified channel names."""
        return MultiChannelImage(data=self._data.assign_coords(c=channel_names))

    def __str__(self) -> str:
        # TODO indicate sparse vs dense repr
        size_y = self._data.sizes["y"]
        size_x = self._data.sizes["x"]
        return f"MultiChannelImage(size_y={size_y}, size_x={size_x}, n_channels={self.n_channels})"
