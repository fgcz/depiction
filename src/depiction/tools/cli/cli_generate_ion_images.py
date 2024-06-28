from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from depiction.parallel_ops import ParallelConfig
from depiction.persistence import ImzmlReadFile
from depiction.tools.generate_ion_image import GenerateIonImage


# TODO for the cli module we should figure out a sane way to configure the default n_jobs value


def generate_ion_images(
    imzml_path: Annotated[Path, typer.Option()],
    mass_list_path: Annotated[Path, typer.Option()],
    output_hdf5_path: Annotated[Path, typer.Option()],
    n_jobs: Annotated[int, typer.Option()] = 16,
) -> None:
    parallel_config = ParallelConfig(n_jobs=n_jobs)
    gen_image = GenerateIonImage(parallel_config=parallel_config)

    mass_list_df = pl.read_csv(mass_list_path)

    image = gen_image.generate_ion_images_for_file(
        input_file=ImzmlReadFile(imzml_path),
        mz_values=mass_list_df["mass"],
        tol=mass_list_df["tol"],
        channel_names=mass_list_df["label"],
    )
    image.write_hdf5(output_hdf5_path)


if __name__ == "__main__":
    typer.run(generate_ion_images)
