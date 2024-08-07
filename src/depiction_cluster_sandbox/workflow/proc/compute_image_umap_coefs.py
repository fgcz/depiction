from pathlib import Path

import cyclopts
from loguru import logger
from umap import UMAP

from depiction.image.feature_selection import FeatureSelectionIQR, retain_features
from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.multi_channel_image_concatenation import MultiChannelImageConcatenation

app = cyclopts.App()

# TODO consider how to test individual steps, by checking several assumptions
#  - extra coordinates preserved
#  - umap multi-channel image generated at output path


# TODO make customizable
feature_selection = FeatureSelectionIQR.validate({"n_features": 30})


@app.default
def compute_image_umap_coefs(
    input_image_path: Path,
    output_image_path: Path,
    n_jobs: int = -1,
    random_state: int | None = None,
    enable_feature_selection: bool = False,
) -> None:
    input_image_conc = MultiChannelImageConcatenation.read_hdf5(path=input_image_path, allow_individual=True)
    input_image = input_image_conc.get_combined_image()
    if enable_feature_selection:
        logger.info(f"Feature selection requested: {feature_selection}")
        input_image = retain_features(feature_selection=feature_selection, image=input_image)

    # compute the umap transformation into 2D
    logger.info(f"Computing UMAP for input image with shape {input_image.dimensions}")
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
    umap_image_conc = input_image_conc.relabel_combined_image(image=umap_image)
    umap_image_conc.write_hdf5(path=output_image_path)


if __name__ == "__main__":
    app()
