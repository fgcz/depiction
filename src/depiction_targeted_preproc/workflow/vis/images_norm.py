from pathlib import Path

import cyclopts

from depiction.image import MultiChannelImage
from depiction.image.image_normalization import ImageNormalization, ImageNormalizationVariant

app = cyclopts.App()


@app.default
def vis_images_norm(input_hdf5_path: Path, output_hdf5_path: Path) -> None:
    image_orig = MultiChannelImage.read_hdf5(input_hdf5_path)
    image_norm = ImageNormalization().normalize_image(image_orig, variant=ImageNormalizationVariant.VEC_NORM)
    image_norm.write_hdf5(output_hdf5_path)


if __name__ == "__main__":
    app()
