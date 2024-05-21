# TODO figure out if format specific exporters should actually be moved to a different path
# TODO figure out the ideal extension i.e. tif vs tiff!
from pathlib import Path
from typing import Any
import pyometiff
import xarray
from ionplotter.persistence.pixel_size import PixelSize


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
        pixel_size = image.attrs["pixel_size"]
        channel_names = list(image.coords["c"].values)

        metadata = {
            **cls._get_image_resolution_metadata(pixel_size),
            **cls._get_channel_metadata(channel_names),
        }

        # add dimensions for the OME-TIFF format
        image_export = image.expand_dims(dim=["z", "t"])

        # export
        writer = pyometiff.OMETIFFWriter(
            fpath=path,
            dimension_order="ZTCYX",
            array=image_export.transpose("z", "t", "c", "y", "x").values,
            metadata=metadata,
            explicit_tiffdata=True,
        )
        writer.write()

    @staticmethod
    def _get_image_resolution_metadata(pixel_size: PixelSize) -> dict[str, Any]:
        assert pixel_size.unit == "micrometer"
        return {
            "PhysicalSizeX": int(pixel_size.size_x),
            "PhysicalSizeY": int(pixel_size.size_y),
        }

    @staticmethod
    def _get_channel_metadata(channel_names: list[str]) -> dict[str, Any]:
        return {
            "Channels": {
                f"{index}": {
                    "Name": str(name),
                    # TODO additional metadata e.g.
                    # "SamplesPerPixel": 1,
                }
                for index, name in enumerate(channel_names)
            },
        }
