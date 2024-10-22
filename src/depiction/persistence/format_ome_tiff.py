# TODO figure out if format specific exporters should actually be moved to a different path
# TODO figure out the ideal extension i.e. tif vs tiff!
from pathlib import Path
from typing import Any

import ome_types
import tifffile
import xarray

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
        image_export = image.expand_dims("s").transpose("c", "y", "x", "s")

        # export
        tifffile.imwrite(
            path,
            image_export.data,
            metadata={"axes": "CYXS", "channel_names": channel_names},
        )

        """
        pixel_size = image.attrs["pixel_size"]
        #metadata = {
        #    **cls._get_image_resolution_metadata(pixel_size),
        #    **cls._get_channel_metadata(channel_names),
        #}

        # add storage dimensions for the OME-TIFF format
        #image_export = image.expand_dims("s").transpose("c", "y", "x", "s")
        image_export = image.expand_dims(["z", "t"]).transpose( "x", "y", "z", "c", "t")
        #image_export = image.expand_dims("s").transpose("c", "y", "x", "c", "t")

        # create OME-XML metadata
        pixels_data = ome_types.model.Pixels(
            # TODO check "PixelType"
            type="float",
            size_c=image_export.sizes["c"],
            size_t=1,
            size_z=1,
            size_x=image_export.sizes["x"],
            size_y=image_export.sizes["y"],
            dimension_order="XYZCT",
            channels=[{"name": name} for name in channel_names],
            physical_size_x=pixel_size.size_x,
            physical_size_y=pixel_size.size_y,
        )
        image_data = ome_types.model.Image(pixels=pixels_data)
        ome_data = ome_types.model.OME(images=[image_data])
        ome_xml = ome_types.to_xml(ome_data)

        # export
        tifffile.imwrite(
            path,
            image_export.data.T,
            description=ome_xml,
            metadata=None,
            ome=True,
            photometric="minisblack",
            #metadata={"axes": "CYXS", **cls._get_channel_metadata(channel_names)},
        )"""

    @classmethod
    def read(cls, path: Path) -> xarray.DataArray:
        file = tifffile.TiffFile(path)
        if not file.is_ome:
            raise ValueError("File is not an OME-TIFF file")

        metadata_xml = file.ome_metadata
        ome_data = ome_types.from_xml(metadata_xml)

        if len(ome_data.images) != 1:
            raise ValueError("Only single image OME-TIFF files are supported")

        image_data = ome_data.images[0]
        np_array = file.asarray(squeeze=False).T[0]
        array = (
            xarray.DataArray(
                np_array,
                dims=[char.lower() for char in image_data.pixels.dimension_order.value],
                coords={"c": [channel.name for channel in image_data.pixels.channels]},
            )
            .squeeze(["z", "t"])
            .transpose("c", "y", "x")
        )

        array.attrs["pixel_size"] = PixelSize(
            size_x=image_data.pixels.physical_size_x,
            size_y=image_data.pixels.physical_size_y,
            unit="micrometer",
        )
        return array

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
