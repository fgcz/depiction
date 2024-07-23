from pathlib import Path

import cyclopts
from depiction.image.multi_channel_image import MultiChannelImage
from matplotlib import pyplot as plt
from umap import UMAP

app = cyclopts.App()


@app.default
def render_umap_png(input_hdf5: Path, output_png: Path) -> None:
    image = MultiChannelImage.read_hdf5(path=input_hdf5)

    # TODO use UMAP(random_state=...) but this will prevent parallelization
    umap = UMAP().fit_transform(image.data_flat.drop_sel(c="cluster").values.T)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(umap[:, 0], umap[:, 1], c=image.data_flat.sel(c="cluster").values.ravel(), s=1, cmap="tab10")
    axs[0].axis("off")
    axs[0].set_title("Clusters")

    axs[1].scatter(umap[:, 0], umap[:, 1], c=image.data_flat.sel(c="image_index").values.ravel(), s=1, cmap="tab10")
    axs[1].axis("off")
    axs[1].set_title("Image IDs")

    plt.savefig(output_png, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app()
