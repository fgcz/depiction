from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image.image_channel_stats import ImageChannelStats
from depiction.image.sparse_representation import SparseRepresentation
from depiction.persistence.format_ome_tiff import OmeTiff

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# TODO would it be clever or stupid to call is_foreground "alpha" channel?


class MultiChannelImage:
    """Represents a multi-channel 2D image, internally backed by a `xarray.DataArray`.

    The API is generally designed to be immutable, i.e. methods modifying the image return a new instance.
    The image is internally represented in a dense layout, with the background/foreground being explicitly stored in
    a `is_foreground` channel that is not part of the `n_channels` count but will be exported.
    This is to make the conversion to and from sparse representation sane.
    """

    def __init__(self, data: DataArray, is_foreground: DataArray, is_foreground_label: str = "is_foreground") -> None:
        self._data = data.transpose("y", "x", "c")
        self._is_foreground = is_foreground.transpose("y", "x")
        self._is_foreground_label = is_foreground_label
        if "bg_value" in self._data.attrs:
            warnings.warn("bg_value is deprecated, use is_foreground instead", DeprecationWarning)
        if (
            self._data.sizes["x"] != self._is_foreground.sizes["x"]
            or self._data.sizes["y"] != self._is_foreground.sizes["y"]
        ):
            raise ValueError("Data and is_foreground must have the same dimensions")
        if np.not_equal(self._data.coords["y"], self._is_foreground.coords["y"]).any():
            raise ValueError("Inconsistent y coordinates between data and is_foreground.")
        if np.not_equal(self._data.coords["x"], self._is_foreground.coords["x"]).any():
            raise ValueError("Inconsistent x coordinates between data and is_foreground.")

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
        data, is_foreground = SparseRepresentation.sparse_to_dense_v2(
            sparse_values=cls._validate_sparse_values(values),
            coordinates=cls._validate_coordinates(coordinates),
            bg_value=bg_value,
        )
        channel_names = list(channel_names) if channel_names is not None else None
        if channel_names:
            data.coords["c"] = channel_names
        return cls(data=data, is_foreground=is_foreground)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._data.sizes["c"]

    @property
    def n_nonzero(self) -> int:
        """Number of non-zero values."""
        return self._is_foreground.sum().item()

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values."""
        return self._data.dtype

    @property
    def fg_mask(self) -> DataArray:
        """A boolean mask indicating the foreground values as `True` and non-foreground values as `False`."""
        return self._is_foreground

    @property
    def bg_mask(self) -> DataArray:
        """A boolean mask indicating the background values as `True` and non-background values as `False`."""
        return ~self._is_foreground

    @property
    def fg_mask_flat(self) -> DataArray:
        """A boolean mask indicating the foreground values as `True` and non-foreground values as `False`."""
        return self._is_foreground.stack(i=("y", "x")).dropna(dim="i")

    @property
    def bg_mask_flat(self) -> DataArray:
        """A boolean mask indicating the background values as `True` and non-background values as `False`."""
        return ~self.fg_mask_flat

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns width and height of the image."""
        # TODO reconsider this method (adding it now for compatibility)
        return self._data.sizes["x"], self._data.sizes["y"]

    @property
    def channel_names(self) -> list[str]:
        """Returns the names of the channels."""
        return [str(c) for c in self._data.coords["c"].values.tolist()]

    @property
    def data_spatial(self) -> DataArray:
        """Returns the underlying data, in its spatial form, i.e. dimensions (y, x, c)."""
        return self._data

    @property
    def data_flat(self) -> DataArray:
        """Returns the underlying data, in its flat form, i.e. dimensions (i, c), omitting any background values."""
        return self._data.stack(i=("y", "x")).isel(i=self.fg_mask_flat)

    @property
    def coordinates_flat(self) -> DataArray:
        """Returns the coordinates of the non-background values."""
        orig_coords = self.data_flat.coords
        return DataArray(
            np.stack((orig_coords["y"].values, orig_coords["x"].values), axis=0),
            dims=("d", "i"),
            coords={"d": ["y", "x"], "i": orig_coords["i"]},
        )

    def recompute_is_foreground(self) -> MultiChannelImage:
        # TODO to be defined
        raise NotImplementedError

    # TODO from_dense_array

    # TODO rename to sel_channels
    def retain_channels(
        self, indices: Sequence[int] | None = None, coords: Sequence[Any] | None = None
    ) -> MultiChannelImage:
        """Returns a copy with only the specified channels retained."""
        if (indices is not None) == (coords is not None):
            raise ValueError("Exactly one of indices and coords must be specified.")
        data = self._data.isel(c=indices) if indices is not None else self._data.sel(c=coords)
        return MultiChannelImage(
            data=data, is_foreground=self._is_foreground, is_foreground_label=self._is_foreground_label
        )

    # TODO rename to dropsel_channels
    def drop_channels(self, *, coords: Sequence[Any], allow_missing: bool) -> MultiChannelImage:
        """Returns a copy with the specified channels dropped."""
        data = self._data.drop_sel(c=coords, errors="ignore" if allow_missing else "raise")
        return MultiChannelImage(
            data=data, is_foreground=self._is_foreground, is_foreground_label=self._is_foreground_label
        )

    # TODO save_single_channel_image... does it belong here or into plotter?

    def write_hdf5(self, path: Path) -> None:
        """Writes the image to a HDF5 file (actually NETCDF4)."""
        self._data.to_netcdf(path, format="NETCDF4")

    @classmethod
    def read_hdf5(
        cls, path: Path, group: str | None = None, is_foreground_label: str = "is_foreground"
    ) -> MultiChannelImage:
        """Reads a MultiChannelImage from a HDF5 file (assuming it contains NETCDF data).

        :param path: The path to the HDF5 file.
        :param group: The group within the HDF5 file, by default None.
        :param is_foreground_label: The label for the is_foreground channel, by default "is_foreground".
        """
        data_store = xarray.open_dataarray(path, group=group)
        is_foreground = data_store.sel(c=is_foreground_label)
        data = data_store.drop_sel(c=is_foreground_label)
        return cls(data=data, is_foreground=is_foreground, is_foreground_label=is_foreground_label)

    # TODO is_valid_hdf5
    # TODO combine_in_parallel, combine_sequentially: consider moving this somewhere else

    @classmethod
    def read_ome_tiff(cls, path: Path, bg_value: float = 0.0) -> MultiChannelImage:
        """Reads a MultiChannelImage from a OME-TIFF file."""
        data = OmeTiff.read(path)
        return MultiChannelImage(data=data, is_foreground=cls._compute_is_foreground(data=data, bg_value=bg_value))

    def with_channel_names(self, channel_names: Sequence[str]) -> MultiChannelImage:
        """Returns a copy with the specified channel names."""
        return MultiChannelImage(
            data=self._data.assign_coords(c=channel_names),
            is_foreground=self._is_foreground,
            is_foreground_label=self._is_foreground_label,
        )

    @cached_property
    def channel_stats(self) -> ImageChannelStats:
        """Returns an object providing channel statistics."""
        return ImageChannelStats(image=self)

    def append_channels(self, other: MultiChannelImage) -> MultiChannelImage:
        """Returns a copy with the channels from the other image appended."""
        common_channels = set(self.channel_names) & set(other.channel_names)
        if common_channels:
            msg = f"Channels {common_channels} are present in both images."
            raise ValueError(msg)
        data = xarray.concat([self._data, other._data], dim="c")
        return MultiChannelImage(
            data=data, is_foreground=self._is_foreground, is_foreground_label=self._is_foreground_label
        )

    def get_z_scaled(self) -> MultiChannelImage:
        """Returns a copy of self with each feature z-scaled."""
        eps = 1e-12
        with xarray.set_options(keep_attrs=True):
            return MultiChannelImage(
                data=(self._data - self.channel_stats.mean + eps) / (self.channel_stats.std + eps),
                is_foreground=self._is_foreground,
                is_foreground_label=self._is_foreground_label,
            )

    # TODO reconsider:there is actually a problem, whether it should use bg_mask only or also replace individual values
    #     since both could be necessary it should be implemented in a sane and maintainable manner
    #    def replace_bg_value(self, new_bg_value: float) -> MultiChannelImage:
    #        """Returns a copy with the background value replaced, i.e. changing all occurrences of the current background
    #        value to the new background value and setting the new background value in the attributes."""
    #        data = self._data.where(~self.bg_mask, new_bg_value)
    #        data.attrs["bg_value"] = new_bg_value
    #        return MultiChannelImage(data=data)

    # def crop_bounding_box(self) -> MultiChannelImage:
    #    #TODO correctly implement this
    #    present_values = np.where(~self.bg_mask)
    #    min_y, max_y = present_values[0].min(), present_values[0].max()
    #    min_x, max_x = present_values[1].min(), present_values[1].max()
    #    data = self._data.isel(y=slice(min_y, max_y + 1), x=slice(min_x, max_x + 1))
    #    return MultiChannelImage(data=data)

    def __str__(self) -> str:
        # TODO indicate sparse vs dense repr
        size_y = self._data.sizes["y"]
        size_x = self._data.sizes["x"]
        return f"MultiChannelImage(size_y={size_y}, size_x={size_x}, n_channels={self.n_channels})"

    def __repr__(self) -> str:
        return f"MultiChannelImage(data={self._data!r})"

    @classmethod
    def _compute_is_foreground(cls, data: DataArray, bg_value: float = np.nan) -> DataArray:
        """Computes the foreground mask from the data."""
        if np.isnan(bg_value):
            return ~data.isnull()
        else:
            return data != bg_value

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
            return coordinates.transpose("i", "d")
        else:
            if coordinates.ndim != 2:
                raise ValueError("Coordinates must be a 2D array.")
            return DataArray(coordinates, dims=("i", "d"))
