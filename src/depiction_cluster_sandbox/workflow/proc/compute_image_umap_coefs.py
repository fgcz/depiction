from pathlib import Path

import cyclopts
from umap import UMAP

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.multi_channel_image_concatenation import MultiChannelImageConcatenation

app = cyclopts.App()

# TODO consider how to test individual steps, by checking several assumptions
#  - extra coordinates preserved
#  - umap multi-channel image generated at output path


@app.default
def compute_image_umap_coefs(
    input_image_path: Path,
    output_image_path: Path,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> None:
    input_image_conc = MultiChannelImageConcatenation.read_hdf5(path=input_image_path, allow_individual=True)
    input_image = input_image_conc.get_combined_image()

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
    umap_image_conc = input_image_conc.with_replaced_combined_image(image=umap_image)
    umap_image_conc.write_hdf5(path=output_image_path)


if __name__ == "__main__":
    app()
