from pathlib import Path

import cyclopts

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.multi_channel_image_concatenation import MultiChannelImageConcatenation

app = cyclopts.App()


@app.default
def concatenate_images(output_hdf5: Path, paths: list[Path]) -> None:
    images = [MultiChannelImage.read_hdf5(path) for path in paths]
    concat = MultiChannelImageConcatenation.concat_images(images=images)
    concat.write_hdf5(path=output_hdf5)


if __name__ == "__main__":
    app()
