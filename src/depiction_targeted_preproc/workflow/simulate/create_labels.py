from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from matplotlib import pyplot as plt
from typer import Option

from depiction.tools.simulate import GenerateLabelImage
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, SimulateParameters


def generate_circles(generator: GenerateLabelImage, params: SimulateParameters):
    rng = np.random.default_rng(0)
    # TODO more customizable
    n_circles_per_channel = np.clip(rng.normal(7, 6, params.n_labels).astype(int), 0, 10)
    return [
        circle
        for i in range(30)
        for circle in generator.sample_circles(radius_mean=20, channel_indices=[i] * n_circles_per_channel[i])
    ]


def simulate_create_labels(
    config_path: Annotated[Path, Option()],
    output_image_path: Annotated[Path, Option()],
    output_overview_image_path: Annotated[Path, Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    generator = GenerateLabelImage(
        image_height=config.simulate.image_height,
        image_width=config.simulate.image_width,
        n_labels=config.simulate.n_labels,
    )
    circles = generate_circles(generator, config.simulate)
    generator.add_circles(circles)
    label_image = generator.render()

    label_image.write_hdf5(output_image_path)

    # create the overview image
    plt.figure(figsize=(10, 8))
    label_image.data_spatial.plot.imshow(x="x", y="y", col="c", col_wrap=10, cmap="gray")
    plt.savefig(output_overview_image_path, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(simulate_create_labels)
