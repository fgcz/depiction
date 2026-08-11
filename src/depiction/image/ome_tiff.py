# TODO figure out if format specific exporters should actually be moved to a different path
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter

from depiction_io.pixel_size import PixelSize

if TYPE_CHECKING:
    from depiction.image import MultiChannelImage


class OmeTiff:
    """Writer for OME-TIFF files to simplify our use case.
    The goal of this class is to not provide the full functionality of the OME-TIFF format as that would be excessive,
    but rather centralize the format handling logic in one place.
    """

    @classmethod
    def write(cls, image: xarray.DataArray, path: Path) -> None:
        """Writes the image to a OME-TIFF file at the specified path.
        The image must have the dimensions c, y, x and an attribute "pixel_size" holding the pixel size information,
        or `None` when the acquisition declared none -- bioio then omits `PhysicalSizeX`/`PhysicalSizeY` rather than
        stating a size that was never measured.
        """
        channel_names = list(image.coords["c"].values)
        image_export = image.transpose("c", "y", "x")
        pixel_size = image.attrs["pixel_size"]
        pixel_sizes = PhysicalPixelSizes(
            Z=None,
            Y=None if pixel_size is None else pixel_size.size_y,
            X=None if pixel_size is None else pixel_size.size_x,
        )
        OmeTiffWriter.save(
            image_export.data, path, channel_names=channel_names, physical_pixel_sizes=[pixel_sizes], dim_order="CYX"
        )

    @classmethod
    def write_image(cls, image: MultiChannelImage, path: Path, pixel_size: PixelSize | None) -> None:
        """Writes the image to an OME-TIFF file at the specified path."""
        # TODO make possible to attach metadata to MultiChannelImage
        data = image.data_spatial.copy()
        data.attrs["pixel_size"] = pixel_size
        cls.write(image=data, path=path)

    @classmethod
    def read(cls, path: Path) -> xarray.DataArray:
        """Reads an OME-TIFF file from the specified path and returns the image as a xarray.DataArray."""
        image = BioImage(path)
        data = xarray.DataArray(
            image.data,
            dims=[d.lower() for d in image.dims.order],
            coords={"c": image.channel_names},
        )
        data = data.squeeze(["t", "z"])
        # ome-types leaves both attributes `None` when the file declares no physical size, so a
        # partially declared one is the only case where anything is thrown away here -- and a
        # `PixelSize` with a `None` side is worse than admitting the size is unknown.
        sizes = image.physical_pixel_sizes
        data.attrs["pixel_size"] = (
            None if sizes.X is None or sizes.Y is None else PixelSize(size_x=sizes.X, size_y=sizes.Y, unit="micrometer")
        )
        return data

    @classmethod
    def read_image(cls, path: Path, bg_value: float = 0.0) -> MultiChannelImage:
        """Reads an OME-TIFF file from the specified path and returns the image as a MultiChannelImage."""
        from depiction.image import MultiChannelImage

        return MultiChannelImage.from_spatial(data=OmeTiff.read(path), bg_value=bg_value)
