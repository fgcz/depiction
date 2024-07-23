from pathlib import Path

import cyclopts
from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence.format_ome_tiff import OmeTiff

app = cyclopts.App()


@app.default()
def ometiff_to_hdf5(
    input_ometiff: Path,
    output_hdf5: Path,
) -> None:
    """Writes input_ometiff to output_hdf5 using our MultiChannelImage representation."""
    data = OmeTiff.read(input_ometiff)
    if "pixel_size" in data.attrs:
        # TODO this is quite broken and should be fixed in the future, but currently the pixel size cannot
        # be persisted
        del data.attrs["pixel_size"]
    image = MultiChannelImage(data)
    image.write_hdf5(output_hdf5)


if __name__ == "__main__":
    app()
