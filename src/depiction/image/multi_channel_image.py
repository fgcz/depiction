from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray
from depiction.image.sparse_representation import SparseRepresentation
from numpy.typing import NDArray
from xarray import DataArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class MultiChannelImage:
    """Represents a multi-channel 2D image, internally backed by a `xarray.DataArray`."""

    def __init__(self, data: DataArray) -> None:
        self._data = data.transpose("y", "x", "c")
        if "bg_value" not in self._data.attrs:
            raise ValueError("The bg_value attribute must be set.")

    @classmethod
    def from_sparse(
        cls,
        values: NDArray[float] | DataArray,
        coordinates: NDArray[int] | DataArray,
        channel_names: list[str] | None,
        bg_value: float = 0.0,
    ) -> MultiChannelImage:
        """Creates a MultiChannelImage instance from sparse arrays providing values and coordinates.
        :param values: The sparse values (n_nonzero, n_channels) (or a DataArray with dims (i, c)).
        :param coordinates: The coordinates of the non-background values (n_nonzero, 2)
            (or a DataArray with dims (i, d)).
        :param channel_names: The names of the channels.
        :param bg_value: The background value.
        """
        data = SparseRepresentation.sparse_to_dense(
            sparse_values=cls._validate_sparse_values(values),
            coordinates=cls._validate_coordinates(coordinates),
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

    @property
    def n_nonzero(self) -> int:
        """Number of non-zero values."""
        # TODO efficient impl
        return (~self.bg_mask).sum().item()

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values."""
        return self._data.dtype

    @property
    def bg_value(self) -> int | float:
        """The background value."""
        return self._data.attrs["bg_value"]

    @cached_property
    def bg_mask(self) -> DataArray:
        """A boolean mask indicating the background values as `True` and non-background values as `False`."""
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

    @property
    def coordinates_flat(self) -> DataArray:
        """Returns the coordinates of the non-background values."""
        orig_coords = self.data_flat.coords
        return DataArray(
            np.stack((orig_coords["y"].values, orig_coords["x"].values), axis=0),
            dims=("d", "i"),
            coords={"d": ["y", "x"], "i": orig_coords["i"]},
        )

    # TODO from_dense_array

    def retain_channels(
        self, indices: Sequence[int] | None = None, coords: Sequence[Any] | None = None
    ) -> MultiChannelImage:
        """Returns a copy with only the specified channels retained."""
        if (indices is not None) == (coords is not None):
            raise ValueError("Exactly one of indices and coords must be specified.")
        data = self._data.isel(c=indices) if indices is not None else self._data.sel(c=coords)
        return MultiChannelImage(data=data)

    # TODO save_single_channel_image... does it belong here or into plotter?

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

    def __repr__(self) -> str:
        return f"MultiChannelImage(data={self._data!r})"

    @staticmethod
    def _validate_sparse_values(values: NDArray[float] | DataArray) -> DataArray:
        """Converts the sparse values to a DataArray, if necessary."""
        if hasattr(values, "coords"):
            return values.transpose("i", "c")
        else:
            if values.ndim != 2:
                raise ValueError("Values must be a 2D array.")
            return DataArray(values, dims=("i", "c"))

    @staticmethod
    def _validate_coordinates(coordinates: NDArray[int] | DataArray) -> DataArray:
        """Converts the coordinates to a DataArray, if necessary."""
        if hasattr(coordinates, "coords"):
            return coordinates.trnaspose("i", "d")
        else:
            if coordinates.ndim != 2:
                raise ValueError("Coordinates must be a 2D array.")
            return DataArray(coordinates, dims=("i", "d"))
