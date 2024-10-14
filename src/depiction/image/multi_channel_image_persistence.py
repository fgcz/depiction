from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage


class MultiChannelImagePersistence:
    """Implements the persistence layer logic for MultiChannelImage."""

    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image

    def write_hdf5(self, path: Path, mode: Literal["a", "w"] = "w", group: str | None = None) -> None:
        data_array = self._image.data_spatial
        is_fg_array = self._image.fg_mask
        is_fg_label = self._image.is_foreground_label

        if not isinstance(data_array.coords["c"][0].item(), str):
            # TODO this really should be validated against in the constructor, and the static methods need to set it
            #   TODO FIXME later
            data_array = data_array.assign_coords(c=self._image.channel_names)

        combined_array = self._stack_for_persistence(
            data_array=data_array, is_fg_array=is_fg_array, is_fg_label=is_fg_label
        )
        # TODO engine should not be necessary, but using it for debugging
        combined_array.to_netcdf(path, mode=mode, group=group, format="NETCDF4", engine="netcdf4")

    @classmethod
    def read_hdf5(
        cls, path: Path, group: str | None = None, is_foreground_label: str = "is_foreground"
    ) -> MultiChannelImage:
        from depiction.image.multi_channel_image import MultiChannelImage

        combined_array = xarray.open_dataarray(path, group=group)
        data_array, is_fg_array = cls._split_from_combined(combined=combined_array, is_fg_label=is_foreground_label)
        return MultiChannelImage(data=data_array, is_foreground=is_fg_array, is_foreground_label=is_foreground_label)

    @classmethod
    def _stack_for_persistence(
        cls, data_array: xarray.DataArray, is_fg_array: xarray.DataArray, is_fg_label: str
    ) -> xarray.DataArray:
        return xarray.concat(
            [data_array, is_fg_array.expand_dims("c", axis=-1).assign_coords(c=[is_fg_label])], dim="c"
        )

    @classmethod
    def _split_from_combined(
        cls, combined: xarray.DataArray, is_fg_label: str
    ) -> tuple[xarray.DataArray, xarray.DataArray]:
        data_array = combined.drop_sel(c=is_fg_label)
        is_fg_array = combined.sel(c=is_fg_label).drop_vars("c").astype(bool)
        return data_array, is_fg_array

    # TODO is_valid_hdf5
