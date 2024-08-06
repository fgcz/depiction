from pathlib import Path

import cyclopts
from umap import UMAP

from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def compute_image_umap_coefs(
    input_image_path: Path,
    output_image_path: Path,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> None:
    input_image = MultiChannelImage.read_hdf5(path=input_image_path)

    # compute the umap transformation into 2D
    umap = UMAP(n_components=2, n_jobs=n_jobs, random_state=random_state)
    values = umap.fit_transform(input_image.data_flat.values.T)

    # create a multi-channel image with these values (n_nonzero, 2)
    umap_image = MultiChannelImage.from_sparse(
        values=values,
        coordinates=input_image.coordinates_flat,
        channel_names=["umap_x", "umap_y"],
        bg_value=0.0,
    )

    # write it to the output path
    umap_image.write_hdf5(path=output_image_path)


if __name__ == "__main__":
    app()
