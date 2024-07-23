import cyclopts
from pathlib import Path
import matplotlib.pyplot as plt
from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def render_single_channel_png(
    input_hdf5: Path,
    output_png: Path,
    channel_index: int = 0,
) -> None:
    image = MultiChannelImage.read_hdf5(input_hdf5)
    image = image.retain_channels(indices=[channel_index])

    plt.figure()
    image.data_spatial.squeeze().plot.imshow(yincrease=False, ax=plt.gca(), x="x", y="y", cmap="tab10")
    plt.savefig(output_png)


if __name__ == "__main__":
    app()
