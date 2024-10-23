# TODO figure out if format specific exporters should actually be moved to a different path

# TODO figure out the ideal extension i.e. tif vs tiff!
from pathlib import Path

import xarray
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from bioio_base.types import PhysicalPixelSizes

from depiction.persistence.pixel_size import PixelSize


class OmeTiff:
    """Writer for OME-TIFF files to simplify our use case.
    The goal of this class is to not provide the full functionality of the OME-TIFF format as that would be excessive,
    but rather centralize the format handling logic in one place.
    """

    @classmethod
    def write(cls, image: xarray.DataArray, path: Path) -> None:
        """Writes the image to a OME-TIFF file at the specified path.
        The image must have the dimensions c, y, x and an attribute "pixel_size" with the pixel size information.
        """
        channel_names = list(image.coords["c"].values)
        image_export = image.transpose("c", "y", "x")
        ps_x, ps_y = image.attrs["pixel_size"].size_x, image.attrs["pixel_size"].size_y
        pixel_sizes = PhysicalPixelSizes(Z=None, Y=ps_y, X=ps_x)
        OmeTiffWriter.save(
            image_export.data, path, channel_names=channel_names, physical_pixel_sizes=[pixel_sizes], dim_order="CYX"
        )

    @classmethod
    def read(cls, path: Path) -> xarray.DataArray:
        image = BioImage(path)
        data = xarray.DataArray(
            image.data,
            dims=[d.lower() for d in image.dims.order],
            coords={"c": image.channel_names},
        )
        data = data.squeeze(["t", "z"])
        data.attrs["pixel_size"] = PixelSize(
            size_x=image.physical_pixel_sizes.X, size_y=image.physical_pixel_sizes.Y, unit="micrometer"
        )
        return data
