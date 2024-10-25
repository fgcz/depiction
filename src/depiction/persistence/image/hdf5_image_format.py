from __future__ import annotations

import xarray
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from depiction.image.container.alpha_channel import AlphaChannel

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage


# TODO currently the files are not closed, which you can observe e.g. in a notebook when the original files has been
#      replaced


class Hdf5ImageFormat:
    """Implements a HDF5 based persistence format for `MultiChannelImage`."""

    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image
        self._alpha_channel = AlphaChannel(label=image.is_foreground_label)

    def write_hdf5(self, path: Path, mode: Literal["a", "w"] = "w", group: str | None = None) -> None:
        """Writes the image to a HDF5 file (actually NETCDF4).

        :param path: The path to the file to write to.
        :param mode: The mode to open the file in. Either 'a' for append, in which case a group should be specified
            so multiple images can be stored in the same file, or 'w' for write, in which case the file will be created
            or overwritten if it already exists.
        :param group: The group to write the image to. If `None`, the image will be written to the root group.
        """
        data_array = self._image.data_spatial
        is_fg_array = self._image.fg_mask

        if not isinstance(data_array.coords["c"][0].item(), str):
            # TODO this really should be validated against in the constructor, and the static methods need to set it
            #   TODO FIXME later
            data_array = data_array.assign_coords(c=self._image.channel_names)

        combined_array = self._alpha_channel.stack(data_array=data_array, is_fg_array=is_fg_array)
        combined_array.attrs["is_foreground_label"] = self._image.is_foreground_label
        # TODO engine should not be necessary, but using it for debugging
        combined_array.to_netcdf(path, mode=mode, group=group, format="NETCDF4", engine="netcdf4")

    @classmethod
    def read_hdf5(cls, path: Path, group: str | None = None) -> MultiChannelImage:
        """Reads a `MultiChannelImage` from a HDF5 file (actually NETCDF4).

        :param path: The path to the file to read from.
        :param group: The group to read the image from. If `None`, the image will be read from the root group.
        """
        from depiction.image.multi_channel_image import MultiChannelImage

        combined_array = xarray.open_dataarray(path, group=group)
        is_foreground_label = combined_array.attrs.get("is_foreground_label", "is_foreground")
        data_array, is_fg_array = AlphaChannel(label=is_foreground_label).split(combined=combined_array)
        return MultiChannelImage(data=data_array, is_foreground=is_fg_array, is_foreground_label=is_foreground_label)

    # TODO is_valid_hdf5
