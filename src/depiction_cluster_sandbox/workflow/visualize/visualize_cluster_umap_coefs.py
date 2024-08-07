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
    umap_image = MultiChannelImageConcatenation.read_hdf5(path=input_umap_hdf5_path, allow_individual=True)
    cluster_image = MultiChannelImage.read_hdf5(path=input_cluster_hdf5_path)

    # read the labels to visualize
    if channel == "cluster":
        labels = cluster_image.data_flat.sel(c="cluster").values.ravel()
    elif channel == "image_index":
        # TODO multi-channel-image currently has a conceptional hole when it comes to dealing with background vs foreground
        #      TODO how to fix it...
        index_image = umap_image.get_combined_image_index()
        joined = index_image.append_channels(cluster_image)
        labels = joined.data_flat.sel(c="image_index").values.ravel()
    else:
        raise ValueError(f"Unknown channel: {channel}")

    # read the umap coordinates
    umap_coords = umap_image.get_combined_image().data_flat.values.T

    # scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=labels, s=1, cmap="tab10")
    ax.set_title(f"UMAP of {channel}")
    ax.axis("off")

    # save the figure
    plt.savefig(output_png_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app()
