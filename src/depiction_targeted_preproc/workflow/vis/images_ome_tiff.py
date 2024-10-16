from pathlib import Path

import cyclopts

from depiction.image import MultiChannelImage
from depiction.persistence.format_ome_tiff import OmeTiff
from depiction.persistence.imzml.extract_metadata import Metadata

app = cyclopts.App()


@app.default
def vis_images_ome_tiff(
    input_netcdf_path: Path,
    input_raw_metadata_path: Path,
    output_ometiff_path: Path,
) -> None:
    # load the netcdf data
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    data = image.data_spatial

    # add image resolution
    raw_metadata = Metadata.model_validate_json(input_raw_metadata_path.read_text())
    # TODO test the implication on save/load
    data.attrs["pixel_size"] = raw_metadata.pixel_size

    # perform the export
    OmeTiff.write(image=data, path=output_ometiff_path)


if __name__ == "__main__":
    app()
