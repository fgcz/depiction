from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray

from depiction.image.container.alpha_channel import AlphaChannel

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage


# TODO currently the files are not closed, which you can observe e.g. in a notebook when the original files has been
#      replaced


class MultiChannelImagePersistence:
    """Implements the persistence layer logic for MultiChannelImage."""

    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image
        self._alpha_channel = AlphaChannel(label=image.is_foreground_label)

    def write_hdf5(self, path: Path, mode: Literal["a", "w"] = "w", group: str | None = None) -> None:
        data_array = self._image.data_spatial
        is_fg_array = self._image.fg_mask

        if not isinstance(data_array.coords["c"][0].item(), str):
            # TODO this really should be validated against in the constructor, and the static methods need to set it
            #   TODO FIXME later
            data_array = data_array.assign_coords(c=self._image.channel_names)

        combined_array = self._alpha_channel.stack(data_array=data_array, is_fg_array=is_fg_array)
        # TODO engine should not be necessary, but using it for debugging
        combined_array.to_netcdf(path, mode=mode, group=group, format="NETCDF4", engine="netcdf4")

    @classmethod
    def read_hdf5(
        cls, path: Path, group: str | None = None, is_foreground_label: str = "is_foreground"
    ) -> MultiChannelImage:
        from depiction.image.multi_channel_image import MultiChannelImage

        combined_array = xarray.open_dataarray(path, group=group)
        data_array, is_fg_array = AlphaChannel(label=is_foreground_label).split(combined=combined_array)
        return MultiChannelImage(data=data_array, is_foreground=is_fg_array, is_foreground_label=is_foreground_label)

    # TODO is_valid_hdf5
