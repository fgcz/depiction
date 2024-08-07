from pathlib import Path
import matplotlib.pyplot as plt
import cyclopts

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.multi_channel_image_concatenation import MultiChannelImageConcatenation

app = cyclopts.App()


@app.default
def visualize_cluster_umap_coefs(
    input_umap_hdf5_path: Path,
    input_cluster_hdf5_path: Path,
    output_png_path: Path,
    channel: str = "cluster",
) -> None:
    # load the input data
    umap_image = MultiChannelImage.read_hdf5(path=input_umap_hdf5_path).retain_channels(coords=["umap_x", "umap_y"])
    cluster_image = MultiChannelImage.read_hdf5(path=input_cluster_hdf5_path)
    combined_image = umap_image.append_channels(cluster_image)

    # read the umap coords and labels to visualize
    if channel in ("cluster", "image_index"):
        labels = combined_image.data_flat.sel(c=channel).values.ravel()
        umap_coords = combined_image.data_flat.sel(c=["umap_x", "umap_y"]).values.T
    else:
        raise ValueError(f"Unknown channel: {channel}")

    # scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=labels, s=1, cmap="tab10")
    ax.set_title(f"UMAP of {channel}")
    ax.axis("off")

    # save the figure
    plt.savefig(output_png_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app()
