import cyclopts
from pathlib import Path

from depiction.image.ome_tiff import OmeTiff

app = cyclopts.App()


@app.default()
def ometiff_to_hdf5(
    input_ometiff: Path,
    output_hdf5: Path,
) -> None:
    """Writes input_ometiff to output_hdf5 using our MultiChannelImage representation."""
    image = OmeTiff.read_image(input_ometiff, bg_value=0.0)
    image.write_hdf5(output_hdf5)


if __name__ == "__main__":
    app()
