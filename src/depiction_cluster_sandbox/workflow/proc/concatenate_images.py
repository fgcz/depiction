from pathlib import Path
import cyclopts
from depiction.image.horizontal_concat import horizontal_concat
from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def concatenate_images(output_hdf5: Path, paths: list[Path]) -> None:
    images = [MultiChannelImage.read_hdf5(path) for path in paths]
    concatenated = horizontal_concat(images, add_index=True)
    concatenated.write_hdf5(output_hdf5)


if __name__ == "__main__":
    app()
