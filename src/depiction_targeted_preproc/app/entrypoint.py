import shutil
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import polars as pl
import typer
import yaml
from depiction_targeted_preproc.example.run import snakemake_invoke, get_result_files, export_results, \
    RESULT_FILE_MAPPING

from depiction.misc.find_file_util import find_one_by_extension
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters, \
    PipelineArtifact, PipelineParametersPreset


def entrypoint(
    script_version: Annotated[str, typer.Option(..., help="The version of the script to run.")] = "",
    input_dir: Annotated[Path, typer.Option(..., help="The input directory.")] = Path("/data/input"),
    output_dir: Annotated[Path, typer.Option(..., help="The output directory.")] = Path("/data/output"),
) -> None:
    if script_version != "":
        raise ValueError(f"Unknown script version: {script_version}")

    # TODO extend in the future
    # Find input files
    input_imzml_file = find_one_by_extension(input_dir=input_dir, extension=".imzML")
    input_panel_file = find_one_by_extension(input_dir=input_dir, extension=".csv")
    workunit_yaml_file = input_dir / "workunit.yaml"

    if not input_imzml_file or not input_panel_file:
        raise RuntimeError(
            f"Input files were not found: input_imzml_file={input_imzml_file}, input_panel_file={input_panel_file}"
        )
    if not workunit_yaml_file.exists():
        raise RuntimeError(f"Workunit yaml file not found: {workunit_yaml_file}")

    # Parse the params
    params = parse_parameters(workunit_yaml_file)

    # Set up the workdir for the Snakemake workflow
    work_dir = output_dir / "work"
    setup_workdir(params=params, input_imzml_file=Path(input_imzml_file), input_panel_file=Path(input_panel_file),
                  work_dir=work_dir)

    # Execute the snakemake workflow
    sample_name = Path(input_imzml_file).stem
    result_files = get_result_files(params=params, work_dir=work_dir, sample_name=sample_name)
    snakemake_invoke(work_dir=work_dir, result_files=result_files)

    # Export the results
    export_results(work_dir=work_dir, output_dir=output_dir, sample_name=sample_name,
                   requested_artifacts=params.requested_artifacts, result_file_mapping=RESULT_FILE_MAPPING)

    # Zip the results
    zip_results(output_dir=output_dir, sample_name=sample_name)


def zip_results(output_dir: Path, sample_name: str) -> None:
    with ZipFile(output_dir / f"{sample_name}.zip", "w") as zipf:
        for file in (output_dir / sample_name).rglob("*"):
            if file.is_file() and file.name != f"{sample_name}.zip":
                zipf.write(file, file.relative_to(output_dir))


def setup_workdir(params: PipelineParameters, input_imzml_file: Path, input_panel_file: Path, work_dir: Path) -> None:
    # Set up the directory
    sample_name = input_imzml_file.stem
    sample_dir = work_dir / sample_name
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Copy the imzML
    shutil.copy(input_imzml_file, sample_dir / "raw.imzML")
    shutil.copy(input_imzml_file.with_suffix(".ibd"), sample_dir / "raw.ibd")

    # Copy the panel file
    copy_standardized_table(input_panel_file, sample_dir / "images_default_mass_list.csv")
    # shutil.copy(input_panel_file, sample_dir / "images_default_mass_list.csv")

    # Write the pipeline parameters
    params_file = sample_dir / "pipeline_params.yml"
    with params_file.open("w") as file:
        yaml.dump(params.dict(), file)


def copy_standardized_table(input_csv: Path, output_csv: Path):
    # TODO this is a total hack for a quick setu
    input_df = pl.read_csv(input_csv)
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


def parse_parameters(yaml_file: Path) -> PipelineParameters:
    data = yaml.safe_load(yaml_file.read_text())

    # Find the correct preset
    preset_name = Path(data["application"]["parameters"]["config_preset"]).name
    preset_path = Path(__file__).parent / "config_presets" / f"{preset_name}.yml"
    # Load the presets
    preset = PipelineParametersPreset.validate(yaml.safe_load(preset_path.read_text()))

    # Add n_jobs and requested_artifacts information to build a PipelineParameters
    # TODO passing this as strings is technically not correct, but it's the only way that currently works
    #      when writing the yaml
    requested_artifacts = []
    if data["application"]["parameters"]["output_activate_calibrated_imzml"]:
        requested_artifacts.append(PipelineArtifact.CALIB_IMZML)
        # requested_artifacts.append("CALIB_IMZML")
    if data["application"]["parameters"]["output_activate_calibrated_ometiff"]:
        requested_artifacts.append(PipelineArtifact.CALIB_IMAGES)
        # requested_artifacts.append("CALIB_IMAGES")
    if data["application"]["parameters"]["output_activate_calibration_qc"]:
        requested_artifacts.append(PipelineArtifact.CALIB_QC)
        # requested_artifacts.append("CALIB_QC")
    return PipelineParameters.from_preset_and_settings(
        preset=preset,
        requested_artifacts=requested_artifacts,
        n_jobs=32
    )


def main() -> None:
    """Provides the CLI around `entrypoint`."""
    typer.run(entrypoint)


if __name__ == "__main__":
    main()
