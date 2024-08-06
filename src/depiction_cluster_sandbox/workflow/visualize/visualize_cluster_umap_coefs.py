from pathlib import Path
import matplotlib.pyplot as plt
import cyclopts

from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def visualize_cluster_umap_coefs(
    input_umap_hdf5_path: Path,
    input_cluster_hdf5_path: Path,
    output_png_path: Path,
) -> None:
    # load the input data
    umap_image = MultiChannelImage.read_hdf5(path=input_umap_hdf5_path)
    cluster_image = MultiChannelImage.read_hdf5(path=input_cluster_hdf5_path)

    # read the cluster labels
    cluster_labels = cluster_image.data_flat.sel(c="cluster").values.ravel()

    # read the umap coordinates
    umap_coords = umap_image.data_flat.values.T

    # scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=cluster_labels, s=1, cmap="tab10")

    # save the figure
    plt.savefig(output_png_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app()
