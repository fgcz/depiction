from pathlib import Path

import cyclopts

from depiction.image import MultiChannelImage
from depiction.persistence.image.ome_tiff import OmeTiff
from depiction.persistence.imzml.metadata import Metadata

app = cyclopts.App()


@app.default
def vis_images_ome_tiff(
    input_netcdf_path: Path,
    input_raw_metadata_path: Path,
    output_ometiff_path: Path,
) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    raw_metadata = Metadata.model_validate_json(input_raw_metadata_path.read_text())
    OmeTiff.write_image(image=image, path=output_ometiff_path, pixel_size=raw_metadata.pixel_size)


if __name__ == "__main__":
    app()
