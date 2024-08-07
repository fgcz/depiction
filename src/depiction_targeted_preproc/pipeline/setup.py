import shutil
from pathlib import Path

import polars as pl
import yaml
from loguru import logger

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def initial_setup(
    input_imzml: Path,
    input_mass_list: Path,
    params_file: Path,
    dir: Path,
    force: bool = False,
    mass_list_filename: str = "mass_list.raw.csv",
) -> None:
    # TODO replace by the following function (after cleaning up the table handling into a dedicated pipeline step)
    if not force and (dir / "raw.imzML").exists():
        logger.info("Skipping initial setup, directory already exists: {dir}", dir=dir)
    else:
        logger.info("Setting up directory: {dir}", dir=dir)
        dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(input_imzml, dir / "raw.imzML")
        shutil.copy(input_imzml.with_suffix(".ibd"), dir / "raw.ibd")
        shutil.copy(input_mass_list, dir / mass_list_filename)
        shutil.copy(params_file, dir / "pipeline_params.yml")


def setup_workdir(params: PipelineParameters, input_imzml_file: Path, input_panel_file: Path, work_dir: Path) -> None:
    # Set up the directory
    sample_name = input_imzml_file.stem
    sample_dir = work_dir / sample_name
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Copy the .imzML and .ibd files
    shutil.copy(input_imzml_file, sample_dir / "raw.imzML")
    shutil.copy(input_imzml_file.with_suffix(".ibd"), sample_dir / "raw.ibd")

    # Copy the panel file
    copy_standardized_table(input_panel_file, sample_dir / "mass_list.raw.csv")

    # Write the pipeline parameters
    params_file = sample_dir / "pipeline_params.yml"
    with params_file.open("w") as file:
        yaml.dump(params.dict(), file)


def copy_standardized_table(input_csv: Path, output_csv: Path):
    input_df = pl.read_csv(input_csv)
    write_standardized_table(input_df, output_csv)


def write_standardized_table(input_df: pl.DataFrame, output_csv: Path) -> None:
    # TODO this is a total hack for a quick setup
    mapping = {}
    for column in input_df.columns:
        if column.lower() in ["marker", "label"]:
            mapping[column] = "label"
        elif column.lower() in ["mass", "m/z", "pc-mt (m+h)+"]:
            mapping[column] = "mass"
        elif column.lower() in ["tol"]:
            mapping[column] = "tol"
    output_df = input_df.rename(mapping)

    if "tol" not in output_df:
        # TODO make configurable
        output_df = output_df.with_columns([pl.Series("tol", [0.2] * len(output_df))])

    output_df.write_csv(output_csv)
