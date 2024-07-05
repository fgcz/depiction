from pathlib import Path
from typing import Annotated

import typer
import xarray

from depiction.persistence.imzml.extract_metadata import Metadata
from depiction.persistence.format_ome_tiff import OmeTiff


def vis_images_ome_tiff(
    input_netcdf_path: Annotated[Path, typer.Option()],
    input_raw_metadata_path: Annotated[Path, typer.Option()],
    output_ometiff_path: Annotated[Path, typer.Option()],
) -> None:
    # load the netcdf data
    image = xarray.open_dataset(input_netcdf_path).to_array("var").squeeze("var")

    # add image resolution
    raw_metadata = Metadata.model_validate_json(input_raw_metadata_path.read_text())
    # TODO test the implication on save/load
    image.attrs["pixel_size"] = raw_metadata.pixel_size

    # perform the export
    OmeTiff.write(image=image, path=output_ometiff_path)


def main() -> None:
    typer.run(vis_images_ome_tiff)


if __name__ == "__main__":
    main()
