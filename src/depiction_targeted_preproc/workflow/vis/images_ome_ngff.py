import cyclopts
from bioio_ome_zarr.writers import Channel, OMEZarrWriter
from depiction.image import MultiChannelImage
from depiction_io.imzml.metadata import Metadata
from pathlib import Path

app = cyclopts.App()

#: OMERO requires a colour per channel and there is nothing in an MSI channel to derive one from,
#: so every channel is white -- the same default bioio uses when it generates channel metadata.
_CHANNEL_COLOR = "ffffff"


@app.default
def vis_images_ome_ngff(
    input_netcdf_path: Path,
    input_raw_metadata_path: Path,
    output_zarr_path: Path,
) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    raw_metadata = Metadata.model_validate_json(input_raw_metadata_path.read_text())
    data = image.data_spatial.transpose("c", "y", "x").data

    # NGFF requires a `scale` transformation on every dataset, so unlike OME-TIFF it cannot leave
    # an unknown pixel size out. The unit carries the distinction instead: a scale in micrometer
    # when the acquisition declared a raster, and an unitless scale of 1 -- which reads as pixels,
    # not as a 1 um raster -- when it did not.
    pixel_size = raw_metadata.pixel_size
    if pixel_size is None:
        physical_pixel_size, axes_units = None, None
    else:
        physical_pixel_size = [1.0, pixel_size.size_y, pixel_size.size_x]
        axes_units = [None, pixel_size.unit, pixel_size.unit]

    writer = OMEZarrWriter(
        output_zarr_path,
        # A single resolution level: these images are small, and nothing downstream reads a pyramid.
        level_shapes=data.shape,
        dtype=data.dtype,
        image_name="Image:0",
        channels=[Channel(label=name, color=_CHANNEL_COLOR) for name in image.channel_names],
        # Spelled out because the default for a 3D array is z/y/x, which would label the channel
        # axis as a third spatial one and give the image a depth it does not have.
        axes_names=["c", "y", "x"],
        axes_types=["channel", "space", "space"],
        axes_units=axes_units,
        physical_pixel_size=physical_pixel_size,
    )
    writer.write_full_volume(data)


if __name__ == "__main__":
    app()
