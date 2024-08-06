from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

from depiction.image.horizontal_concat import horizontal_concat
from depiction.image.multi_channel_image import MultiChannelImage


# TODO properly document (y, x) vs (x, y) and the min_coords in get_single_image and get_single_images


class MultiChannelImageConcatenation:
    """Represents a concatenation of multiple multi-channel images, with potentially different shapes
    (but same number of channels per image and same background value).

    This is done by concatenating the images in the spatial domain, so it is possible to obtain a single
    multi-channel image as well as a list of individual multi-channel images.
    """

    def __init__(self, data: MultiChannelImage) -> None:
        self._data = data

    @cached_property
    def n_individual_images(self) -> int:
        """Number of individual images."""
        return int(self._data.retain_channels(coords=["image_index"]).data_flat.max().values + 1)

    def get_combined_image(self) -> MultiChannelImage:
        return self._data.drop_channels(coords=["image_index"], allow_missing=False)

    def get_combined_image_index(self) -> MultiChannelImage:
        return self._data.retain_channels(coords=["image_index"])

    def get_single_image(self, index: int, min_coords: tuple[int, int] = (0, 0)) -> MultiChannelImage:
        # perform the selection in flat representation for sanity
        # all_values = self._data.data_flat.drop_sel(c="image_index", allow_missing=False)
        all_values = self._data.drop_channels(coords=["image_index"], allow_missing=False).data_flat
        all_coords = self._data.coordinates_flat

        # determine the indices in flat representation, corresponding to the requested image
        sel_indices = np.where(self._data.data_flat.sel(c="image_index").values == index)[0]

        # select the values and coordinates
        sel_values = all_values.isel(i=sel_indices)
        sel_coords = all_coords.isel(i=sel_indices)

        # readjust the coordinates
        sel_coords = sel_coords - sel_coords.min(axis=1) + np.array(min_coords)[:, None]

        # create the individual image
        return MultiChannelImage.from_sparse(
            values=sel_values,
            coordinates=sel_coords,
            channel_names=sel_values.coords["c"].values.tolist(),
            bg_value=sel_values.bg_value,
        )

    def get_single_images(self) -> list[MultiChannelImage]:
        return [self.get_single_image(index=index) for index in range(self.n_individual_images)]

    @classmethod
    def read_hdf5(cls, path: Path) -> MultiChannelImageConcatenation:
        return cls(data=MultiChannelImage.read_hdf5(path=path))

    def write_hdf5(self, path: Path) -> None:
        self._data.write_hdf5(path=path)

    @classmethod
    def concat_images(cls, images: list[MultiChannelImage]) -> MultiChannelImageConcatenation:
        """Returns the horizontal concatenation of the provided images."""
        # TODO consider introducing a padding step as it would be more precise
        data = horizontal_concat(images=images, add_index=True, index_channel="image_index")
        return cls(data=data)
