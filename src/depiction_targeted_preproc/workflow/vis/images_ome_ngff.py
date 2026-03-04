import cyclopts
from bioio.writers import OmeZarrWriter
from bioio_base.types import PhysicalPixelSizes
from depiction.image import MultiChannelImage
from depiction_io.imzml.metadata import Metadata
from pathlib import Path

app = cyclopts.App()


@app.default
def vis_images_ome_ngff(
    input_netcdf_path: Path,
    input_raw_metadata_path: Path,
    output_zarr_path: Path,
) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    raw_metadata = Metadata.model_validate_json(input_raw_metadata_path.read_text())

    ps_x, ps_y = raw_metadata.pixel_size.size_x, raw_metadata.pixel_size.size_y
    pixel_sizes = PhysicalPixelSizes(Z=None, Y=ps_y, X=ps_x)
    OmeZarrWriter(output_zarr_path).write_image(
        image.data_spatial.transpose("c", "y", "x").data,
        "Image:0",
        channel_names=image.channel_names,
        # TODO at least in the napari plugin, setting the channel_colors to None leads to a rather bad user experience,
        #      maybe we could set these values
        channel_colors=None,
        physical_pixel_sizes=pixel_sizes,
        dimension_order="CYX",
    )


if __name__ == "__main__":
    app()
