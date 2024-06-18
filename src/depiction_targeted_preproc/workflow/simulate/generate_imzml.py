from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence import ImzmlWriteFile, ImzmlModeEnum
from depiction.tools.simulate import SyntheticMSIDataGenerator
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def simulate_generate_imzml(
    input_image_path: Annotated[Path, Option()],
    input_mass_list_path: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
    output_imzml_path: Annotated[Path, Option()],
) -> None:
    write_file = ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
    label_image = MultiChannelImage.read_hdf5(input_image_path)
    mass_list = pl.read_csv(input_mass_list_path)
    config = PipelineParameters.parse_yaml(config_path)

    mz_arr = SyntheticMSIDataGenerator.get_mz_arr(
        mass_list["mass"].min() - 50, mass_list["mass"].max() + 50, config.simulate.bin_width_ppm
    )
    gen = SyntheticMSIDataGenerator()
    gen.generate_imzml_for_labels(
        write_file=write_file,
        label_image=label_image,
        label_masses=np.asarray(mass_list["mass"]),
        n_isotopes=3,
        mz_arr=mz_arr,
        baseline_max_intensity=1.0,
        background_noise_strength=0.05,
    )


if __name__ == "__main__":
    typer.run(simulate_generate_imzml)
