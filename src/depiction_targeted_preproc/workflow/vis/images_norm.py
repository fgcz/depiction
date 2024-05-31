from typing import Annotated

import typer
import xarray

from depiction.image.image_normalization import ImageNormalization, ImageNormalizationVariant


def vis_images_norm(
    input_hdf5_path: Annotated[str, typer.Option()],
    output_hdf5_path: Annotated[str, typer.Option()],
) -> None:
    image_orig = xarray.open_dataset(input_hdf5_path).to_array("var").squeeze("var")
    image_norm = ImageNormalization().normalize_xarray(image_orig, variant=ImageNormalizationVariant.VEC_NORM)
    image_norm.to_netcdf(output_hdf5_path)


def main():
    typer.run(vis_images_norm)


if __name__ == "__main__":
    main()
