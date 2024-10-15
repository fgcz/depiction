from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray
from numpy.typing import NDArray
from xarray import DataArray

from depiction.image.image_channel_stats import ImageChannelStats
from depiction.image.multi_channel_image_persistence import MultiChannelImagePersistence
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
        if "bg_value" in data.attrs:
            # TODO remove this warning at a later time
            warnings.warn("bg_value is deprecated, use is_foreground instead", DeprecationWarning)

        # Assign the data
        self._data = data.transpose("y", "x", "c").drop_attrs()
        self._is_foreground = is_foreground.transpose("y", "x").drop_vars("c", errors="ignore").drop_attrs()
        self._is_foreground_label = is_foreground_label

        # Validate the input
        self._assert_data_and_foreground_dimensions()
        self._assert_data_and_foreground_coords()
        self._assert_foreground_is_boolean()
        self._assert_data_channel_names_present()

    def _assert_data_channel_names_present(self) -> None:
        """Asserts that the data has channel names and that they are strings."""
        if "c" not in self._data.coords:
            raise ValueError("Data must have a 'c' coordinate for channel names.")
        if self._data.sizes["c"] > 0 and not isinstance(self._data.c[0].item(), str):
            raise ValueError(f"Channel names must be strings, but type is: {type(self._data.c[0].item())}.")

    def _assert_data_and_foreground_coords(self) -> None:
        if np.not_equal(self._data.coords["y"], self._is_foreground.coords["y"]).any():
            raise ValueError("Inconsistent y coordinate values between data and is_foreground.")
        if np.not_equal(self._data.coords["x"], self._is_foreground.coords["x"]).any():
            raise ValueError("Inconsistent x coordinate values between data and is_foreground.")

    def _assert_data_and_foreground_dimensions(self) -> None:
        if (
            self._data.sizes["x"] != self._is_foreground.sizes["x"]
            or self._data.sizes["y"] != self._is_foreground.sizes["y"]
        ):
            msg = (
                "'data' and 'is_foreground' must have the same dimensions, but "
                f"data[y,x] = {self._data.sizes['y'], self._data.sizes['x']}, "
                f"is_foreground[y,x] = {self._is_foreground.sizes['y'], self._is_foreground.sizes['x']}."
            )
            raise ValueError(msg)

    def _assert_foreground_is_boolean(self) -> None:
        if self._is_foreground.dtype != np.bool_:
            raise ValueError(f"is_foreground must be a boolean array, but has dtype {self._is_foreground.dtype}.")

    @classmethod
    def from_spatial(
        cls, data: DataArray, bg_value: float = 0, is_foreground_label: str = "is_foreground"
    ) -> MultiChannelImage:
        # TODO improve this method
        is_fg = cls._compute_is_foreground(data=data, bg_value=bg_value)
        return cls(data=data, is_foreground=is_fg, is_foreground_label=is_foreground_label)

    @classmethod
    def from_flat(
        cls,
        values: DataArray,
        coordinates: DataArray | None,
        channel_names: list[str] | bool = False,
        bg_value: float = 0.0,
    ):
        coordinates = cls._extract_flat_coordinates(values) if coordinates is None else coordinates
        if channel_names:
            if "c" in values.coords:
                msg = (
                    "Either provide channel names as coordinate in values or as argument, but not both. "
                    "Use .drop_vars('c') to remove the `c` coordinate, "
                    "or use `.assign_coords(c=channel_names)` to add channel names directly."
                )
                raise ValueError(msg)
            elif isinstance(channel_names, bool):
                channel_names = [str(i) for i in range(values.sizes["c"])]
            else:
                channel_names = [str(name) for name in channel_names]
            values = values.assign_coords(c=channel_names)

        data, is_foreground = SparseRepresentation.flat_to_spatial(
            sparse_values=values.transpose("i", "c"),
            coordinates=cls._validate_coordinates(coordinates),
            bg_value=bg_value,
        )
        return cls(data=data, is_foreground=is_foreground)

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
        # TODO delete method
        warnings.warn("from_sparse is deprecated, use from_flat instead", DeprecationWarning)
        data, is_foreground = SparseRepresentation.flat_to_spatial(
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
    def is_foreground_label(self) -> str:
        """The label for the is_foreground channel when persisting."""
        return self._is_foreground_label

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
        # TODO make consistent
        # return DataArray(
        #    np.stack((orig_coords["x"].values, orig_coords["y"].values), axis=0),
        #    dims=("d", "i"),
        #    coords={"d": ["x", "y"], "i": orig_coords["i"]},
        # )
        return DataArray(
            np.stack((orig_coords["y"].values, orig_coords["x"].values), axis=0),
            dims=("d", "i"),
            coords={"d": ["y", "x"], "i": orig_coords["i"]},
        )

    def recompute_is_foreground(self, bg_value: float = 0.0) -> MultiChannelImage:
        """Returns a copy of self with a recomputed is_foreground mask, based on the provided bg value."""
        is_foreground = self._compute_is_foreground(data=self._data, bg_value=bg_value)
        return MultiChannelImage(
            data=self._data, is_foreground=is_foreground, is_foreground_label=self._is_foreground_label
        )

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

    def write_hdf5(self, path: Path, mode: Literal["a", "w"] = "w", group: str | None = None) -> None:
        """Writes the image to a HDF5 file (actually NETCDF4)."""
        return MultiChannelImagePersistence(image=self).write_hdf5(path=path, mode=mode, group=group)

    @classmethod
    def read_hdf5(
        cls, path: Path, group: str | None = None, is_foreground_label: str = "is_foreground"
    ) -> MultiChannelImage:
        """Reads a MultiChannelImage from a HDF5 file (assuming it contains NETCDF data).

        :param path: The path to the HDF5 file.
        :param group: The group within the HDF5 file, by default None.
        :param is_foreground_label: The label for the is_foreground channel, by default "is_foreground".
        """
        return MultiChannelImagePersistence.read_hdf5(path=path, group=group, is_foreground_label=is_foreground_label)

    # TODO combine_in_parallel, combine_sequentially: consider moving this somewhere else

    @classmethod
    def read_ome_tiff(cls, path: Path, bg_value: float = 0.0) -> MultiChannelImage:
        """Reads a MultiChannelImage from a OME-TIFF file."""
        data = OmeTiff.read(path)
        return MultiChannelImage(data=data, is_foreground=cls._compute_is_foreground(data=data, bg_value=bg_value))

    def with_channel_names(self, channel_names: Sequence[str]) -> MultiChannelImage:
        """Returns a copy with the specified channel names."""
        # TODO too specific! it would be better to have a "rename_channels" method instead that allows specifying only some
        #      or, do a "select" like in polars
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
            return ~data.isnull().all(dim="c")
        else:
            return (data != bg_value).any(dim="c")

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
        if not hasattr(coordinates, "coords"):
            return DataArray(coordinates, dims=("i", "d"), coords={"d": ["x", "y"]})
        else:
            coordinates = coordinates.transpose("i", "d").sortby("d")
            if not coordinates.coords["d"].values.tolist() == ["x", "y"]:
                raise ValueError("Coordinates must have dimensions 'x' and 'y'.")
            return coordinates

    @classmethod
    def _extract_flat_coordinates(cls, values: DataArray) -> DataArray:
        return DataArray(
            np.stack([values.coords["x"].values, values.coords["y"].values], axis=1),
            dims=("i", "d"),
            coords={"d": ["x", "y"]},
        )
