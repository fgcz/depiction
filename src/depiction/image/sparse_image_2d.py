from __future__ import annotations

# file is deprecated, use MultiChannelImage instead!

import warnings

warnings.warn("This file is deprecated, use MultiChannelImage instead!", DeprecationWarning)

import math
from typing import Callable, TYPE_CHECKING

import matplotlib.pyplot
import numpy as np
import seaborn
from matplotlib import pyplot as plt
import xarray

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Sequence
    import h5py


class SparseImage2d:
    """
    A sparse representation of a 2d image.
    The coordinate system starts at bottom-left, with the x-axis pointing to the right and the y-axis pointing upwards.
    :param values: (n_nonzero, n_channels) the values of the non-zero pixels
    :param coordinates: (n_nonzero, 2) the coordinates of the non-zero pixels
    """

    def __init__(
        self,
        values: NDArray[float],
        coordinates: NDArray[int],
        channel_names: list[str] | None = None,
    ) -> None:
        self._values = values
        self._coordinates = coordinates
        self._channel_names = channel_names
        self._validate()

    def _validate(self) -> None:
        """Validates the fields of this class and raises a ``ValueError`` if any violations are found."""
        if self._values.ndim != 2:
            raise ValueError(f"values must have two dimensions, but it has shape {self._values.shape}")
        if self._coordinates.ndim != 2:
            raise ValueError(f"coordinates must have two dimensions, but it has shape {self._coordinates.shape}")
        if self._values.shape[0] != self._coordinates.shape[0]:
            raise ValueError(
                f"values and coordinates must have the same number of rows, but have shapes {self._values.shape} "
                f"and {self._coordinates.shape}"
            )
        if self._coordinates.shape[1] != 2:
            raise ValueError(
                f"coordinates must have two columns (2d coordinates), but have shape {self._coordinates.shape}"
            )
        if self._channel_names is not None and len(self._channel_names) != self.n_channels:
            raise ValueError(
                f"channel_names must have length {self.n_channels}, but has length {len(self._channel_names)}"
            )

    @property
    def n_nonzero(self) -> int:
        """
        Number of non-zero pixels. This checks the static information, expecting that only non-zero pixels are
        provided to the constructor.
        """
        return self._values.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._values.shape[1]

    @property
    def sparse_values(self) -> NDArray[float]:
        """Returns the underlying values."""
        view = self._values.view()
        view.flags.writeable = False
        return view

    @property
    def sparse_coordinates(self) -> NDArray[int]:
        """Returns the underlying coordinates."""
        view = self._coordinates.view()
        view.flags.writeable = False
        return view

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the values."""
        return self._values.dtype

    @property
    def offset(self) -> NDArray[int]:
        """Returns the offset of the coordinates, i.e. in a dense representation usually these will be skipped."""
        return self._coordinates.min(axis=0)

    @property
    def dimensions(self) -> tuple[int, int]:
        height = self._coordinates[:, 1].max() - self._coordinates[:, 1].min() + 1
        width = self._coordinates[:, 0].max() - self._coordinates[:, 0].min() + 1
        return width, height

    @property
    def channel_names(self) -> list[str]:
        """Returns the names of the channels, either as specified or 'Channel 0', 'Channel 1', ..."""
        if self._channel_names is None:
            return [f"Channel {i}" for i in range(self.n_channels)]
        else:
            # TODO unit test that it is indeed a list and not a numpy array (which can happen with pandas dataframes and has different semantics)
            return list(self._channel_names)

    def get_dense_array(self, bg_value: float | int = 0) -> NDArray[float]:
        """
        Returns a dense representation of the image.
        The result will be casted, if necessary to accommodate the bg_value, e.g. int to float when bg_value is nan.
        :param bg_value: the value to use for the background (i.e. where no pixel information is present)
        Returns:
            - dense_values: (n_rows, n_cols, n_channels) the values of the pixels (i.e. [-y, x, channel])
        """
        dense_coordinates = self._coordinates - self.offset
        # be aware: rows=y, cols=x
        shape = (
            dense_coordinates[:, 1].max() + 1,
            dense_coordinates[:, 0].max() + 1,
            self.n_channels,
        )
        dtype = np.promote_types(self.dtype, np.obj2sctype(type(bg_value)))
        dense_values = np.full(shape, bg_value, dtype=dtype)
        dense_values[
            # flipped y-axis (convention)
            dense_coordinates[:, 1].max() - dense_coordinates[:, 1],
            dense_coordinates[:, 0],
        ] = self._values
        return dense_values

    @classmethod
    def from_dense_array(
        cls,
        dense_values: NDArray[float],
        offset: NDArray[int],
        channel_names: list[str] | None = None,
    ) -> SparseImage2d:
        """Creates an instance of SparseImage2d from a dense array and an offset.
        Note that the current implementation will not attempt to remove zero values from the dense array, and treat
        them as values present.
        :param dense_values: (n_rows, n_cols, n_channels) the values of the pixels
        :param offset: (2,) the offset of the coordinates
        :param channel_names: (n_channels,) the names of the channels (optional)
        """
        n_rows, n_cols, n_channels = dense_values.shape
        x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        coordinates = np.stack([x, y], axis=-1).reshape(-1, 2) + offset[:2].astype(int)
        values = dense_values.reshape(-1, n_channels)
        return cls(values=values, coordinates=coordinates, channel_names=channel_names)

    def get_single_channel_dense_array(self, i_channel: int, bg_value: float = 0.0) -> NDArray[float]:
        """
        Returns a dense representation of the image for a single channel.
        :param i_channel: the index of the channel to return
        :param bg_value: the value to use for the background (i.e. where no pixel information is present)
        Returns:
            - dense_values: (n_rows, n_cols) the values of the pixels
        """
        single_channel_im = SparseImage2d(values=self._values[:, [i_channel]], coordinates=self._coordinates)
        dense_values = single_channel_im.get_dense_array(bg_value=bg_value)
        return dense_values[:, :, 0]

    def retain_channels(self, channel_indices: Sequence[int]) -> SparseImage2d:
        """Returns a new instance with only the specified channels retained."""
        channel_indices = np.asarray(channel_indices)
        return SparseImage2d(
            values=self._values[:, channel_indices],
            coordinates=self._coordinates,
            channel_names=[self.channel_names[i] for i in channel_indices],
        )

    def save_single_channel_image(
        self,
        i_channel: int,
        path: str,
        cmap: str = "mako",
        transform_int: Callable | None = None,
    ) -> None:
        """
        Saves a single channel as an image.
        :param i_channel: the index of the channel to save
        :param path: the path to save the image to, e.g. "image.png"
        :param cmap: the color map to use, e.g. "mako"
        :param transform_int: a function to transform the values to integers, e.g. ``lambda x: x * 1000``
        """
        cmap = seaborn.color_palette(cmap, as_cmap=True)
        # TODO this could be extracted to a visualization module
        dense_values = self.get_single_channel_dense_array(i_channel)
        if transform_int is not None:
            dense_values = transform_int(dense_values)
        matplotlib.pyplot.imsave(path, dense_values, cmap=cmap, origin="lower")

    # TODO extract the grid logic into a grid layout class, since it would also be useful for cases where one wants to
    #      plot a sparse image and a non-sparse one together

    def plot_single_channel_image(
        self,
        i_channel: int,
        ax: plt.Axes | None = None,
        cmap: str = "mako",
        transform_int: Callable | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_fn: Callable | None = None,
        vmax_fn: Callable | None = None,
        interpolation: str | None = None,
        mask_background: bool = False,
    ) -> None:
        if ax is None:
            ax = plt.gca()

        ax.set_title(self.channel_names[i_channel])
        ax.set_xticks([])
        ax.set_yticks([])

        if vmin_fn is not None:
            vmin = vmin_fn(self._values[:, i_channel])
        if vmax_fn is not None:
            vmax = vmax_fn(self._values[:, i_channel])

        bg_value = np.nan if mask_background else 0.0
        dense_values = self.get_single_channel_dense_array(i_channel, bg_value=bg_value)
        if transform_int is not None:
            dense_values = transform_int(dense_values)
        if mask_background:
            dense_values = np.ma.masked_invalid(dense_values)

        ax.imshow(
            dense_values,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

    def plot_channels_grid(
        self,
        n_per_row: int = 5,
        cmap: str = "mako",
        transform_int: Callable | None = None,
        single_im_width: float = 2.0,
        single_im_height: float | None = None,
        # TODO consider if this can be handled better (i.e. the parameter interface mainly)
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_fn: Callable | None = None,
        vmax_fn: Callable | None = None,
        interpolation: str | None = None,
        mask_background: bool = False,
        axs: NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        # determine plots layout
        n_channels = self.n_channels
        n_rows = math.ceil(n_channels / n_per_row)
        n_cols = min(n_channels, n_per_row)

        # determine the figure size
        im_width, im_height = self.dimensions
        aspect_ratio = im_width / im_height

        # set up the grid
        if single_im_height is None:
            single_im_height = single_im_width

        if axs is None:
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    n_cols * single_im_width,
                    n_rows * single_im_height * aspect_ratio,
                ),
                squeeze=False,
            )
        else:
            fig = axs.ravel()[0].get_figure()

        # create the plots
        for i_channel in range(n_channels):
            # Get the axis
            i_row = i_channel // n_per_row
            i_col = i_channel % n_per_row
            ax = axs[i_row, i_col]

            self.plot_single_channel_image(
                i_channel=i_channel,
                ax=ax,
                cmap=cmap,
                transform_int=transform_int,
                vmin=vmin,
                vmax=vmax,
                vmin_fn=vmin_fn,
                vmax_fn=vmax_fn,
                interpolation=interpolation,
                mask_background=mask_background,
            )

        # remove redundant axes
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                if i_row * n_per_row + i_col >= n_channels:
                    axs[i_row, i_col].set_axis_off()

        return fig, axs

    def to_dense_xarray(self, bg_value: float = 0.0) -> xarray.DataArray:
        """Returns a dense representation of the image as an `xarray.DataArray`."""
        return xarray.DataArray(
            self.get_dense_array(bg_value=bg_value), dims=("y", "x", "c"), coords={"c": self.channel_names}
        )

    @classmethod
    def from_dense_xarray(cls, xarray_data: xarray.DataArray) -> SparseImage2d:
        xarray_data.transpose(("y", "x", "c"))
        raise NotImplementedError("This method is not implemented yet.")

    def save_to_hdf5(self, group: h5py.Group) -> None:
        """Saves this instance to an HDF5 group."""
        group.create_dataset("values", data=self._values)
        group.create_dataset("coordinates", data=self._coordinates)
        group.attrs["_type"] = "SparseImage2d"
        group.attrs["channel_names"] = self._channel_names

    @classmethod
    def load_from_hdf5(cls, group: h5py.Group) -> SparseImage2d:
        """Loads an instance from an HDF5 group."""
        return cls(
            values=np.asarray(group["values"]),
            coordinates=np.asarray(group["coordinates"]),
            channel_names=list(group.attrs["channel_names"]),
        )

    @classmethod
    def is_valid_hdf5(cls, group: h5py.Group) -> bool:
        """Returns whether the given group is a valid HDF5 group for this class."""
        if hasattr(group, "attrs") and "_type" not in group.attrs:
            return False
        return group.attrs["_type"] == "SparseImage2d"

    @classmethod
    def combine_in_parallel(cls, images: list[SparseImage2d]) -> SparseImage2d:
        """Combines the individual images in parallel, i.e. it assumes that every image has the same dimensions and
        number of channels.

        Images A, B with channels 1, 2 would be passed as [A, B] and the result would be a single
        image with channels of order A1, B1, A2, B2. Channel names will be copied from the individual images.
        Coordinates also have to be identical for all images."""
        # TODO consider if there should be a function here or in general to add a prefix to sparse images (but low prio)

        n_nonzero = images[0].n_nonzero
        n_channels = images[0].n_channels

        # check number of channels per image to be equal
        if len({image.n_channels for image in images}) != 1:
            raise ValueError(f"Images have different numbers of channels: {[image.n_channels for image in images]}")

        # check if image sizes are identical
        if len({image.dimensions for image in images}) != 1:
            raise ValueError(f"Images have different dimensions: {[image.dimensions for image in images]}")
        if len({image.n_nonzero for image in images}) != 1:
            raise ValueError(
                f"Images have different numbers of nonzero pixels: {[image.n_nonzero for image in images]}"
            )

        # check if images have different dtypes
        if len({image.dtype for image in images}) != 1:
            dtypes = {image.dtype for image in images}
            raise ValueError(f"Images have different dtypes: {dtypes}")

        # check coordinates
        coordinates = images[0].sparse_coordinates
        if any(not np.array_equal(coordinates, image.sparse_coordinates) for image in images[1:]):
            raise ValueError("Coordinates are not identical for all images.")

        # create the result
        values = np.zeros((n_nonzero, len(images) * n_channels), dtype=images[0].dtype)
        channel_names = []
        for i_channel in range(n_channels):
            for i_image, image in enumerate(images):
                values[:, i_channel * len(images) + i_image] = image.sparse_values[:, i_channel]
                channel_names.append(image.channel_names[i_channel])

        return cls(values=values, coordinates=coordinates, channel_names=channel_names)

    def combine_sequentially(self, other: SparseImage2d) -> SparseImage2d:
        """Returns a new instance with the channels of the other image appended to the channels of this image."""
        # TODO add all the checks
        # TODO unit test
        channel_names_new = self.channel_names + other.channel_names
        values_new = np.concatenate([self.sparse_values, other.sparse_values], axis=1)
        return SparseImage2d(
            values=values_new,
            coordinates=self.sparse_coordinates,
            channel_names=channel_names_new,
        )

    def with_channel_names(self, channel_names: list[str]) -> SparseImage2d:
        """Returns a new instance with the given channel names."""
        return SparseImage2d(
            values=self._values,
            coordinates=self._coordinates,
            channel_names=channel_names,
        )

    def __str__(self) -> str:
        return (
            "SparseImage2d with "
            f"n_nonzero={self.n_nonzero}, n_channels={self.n_channels}, offset={tuple(self.offset)}"
        )
