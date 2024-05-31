from pathlib import Path
import polars as pl
import typer
from typing import Annotated

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile
from depiction.tools.generate_ion_image import GenerateIonImage
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def vis_images(
    imzml_path: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    mass_list_path: Annotated[Path, typer.Option()],
    output_hdf5_path: Annotated[Path, typer.Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    mass_list_df = pl.read_csv(mass_list_path)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs, task_size=None)

    gen_image = GenerateIonImage(parallel_config=parallel_config)
    image = gen_image.generate_ion_images_for_file(
        input_file=ImzmlReadFile(imzml_path),
        mz_values=mass_list_df["mass"],
        tol=mass_list_df["tol"],
        channel_names=mass_list_df["label"],
    )
    image.write_hdf5(output_hdf5_path)

    # if output_stats_path:
    #    stats = ImageChannelStatistics.compute_xarray(xarray)
    #    stats.write_csv(output_stats_path)


def main() -> None:
    typer.run(vis_images)


if __name__ == "__main__":
    main()
